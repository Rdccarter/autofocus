from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np


from .focus_metric import Roi, astigmatic_error_signal, roi_total_intensity, fit_gaussian_psf
from .interfaces import CameraInterface, StageInterface
from .zhuang import (
    ZhuangEllipticityParams,
    ZhuangLookupTable,
    ZhuangParams,
    build_lookup_table,
    fit_zhuang_axis,
    fit_zhuang_ellipticity,
    second_moment_error_from_ellipticity,
    zhuang_ellipticity,
    zhuang_usable_range,
)


@dataclass(slots=True)
class FocusCalibration:
    """Maps astigmatic error to physical Z *offset* (command delta).

    The mapping follows a local linear approximation around focus:
    z_offset_um ~= error_to_um * (error - error_at_focus)

    Important: this calibration is intentionally relative (how much to move and
    in which direction), not an absolute Z-position model.

    Note: `error_at_focus` is derived from calibration samples and is interpreted
    in the local sweep frame used by the fitter. With symmetric sweeps centered
    near focus this approximates true best-focus error well. Strongly asymmetric
    sweeps can bias this estimate; prefer centered bidirectional sweeps.
    """

    error_at_focus: float
    error_to_um: float
    error_min: float | None = None
    error_max: float | None = None

    def error_to_z_offset_um(self, error: float) -> float:
        return (error - self.error_at_focus) * self.error_to_um

    def is_error_in_range(self, error: float, margin: float = 0.0) -> bool:
        if self.error_min is None or self.error_max is None:
            return True
        return (self.error_min - margin) <= error <= (self.error_max + margin)


@dataclass(slots=True)
class CalibrationMetadata:
    objective: str = ""
    roi_size: tuple[int, int] | None = None
    exposure_ms: float | None = None
    wavelength_nm: float | None = None
    camera_roi: tuple[int, int, int, int] | None = None
    camera_binning: str | None = None
    stage_type: str | None = None
    temperature_c: float | None = None
    created_at_unix_s: float = 0.0


@dataclass(slots=True)
class CalibrationSample:
    z_um: float
    error: float
    weight: float = 1.0


@dataclass(slots=True)
class CalibrationFitReport:
    calibration: FocusCalibration
    intercept_um: float
    r2: float
    rmse_um: float
    n_samples: int
    n_inliers: int
    robust: bool
    recommended_z_range_um: tuple[float, float]
    hysteresis_um: float




def _sanitize_calibration_samples(samples: list[CalibrationSample]) -> list[CalibrationSample]:
    if len(samples) < 2:
        raise ValueError("Need at least two calibration samples")

    sanitized: list[CalibrationSample] = []
    for s in samples:
        if (not math.isfinite(float(s.z_um))) or (not math.isfinite(float(s.error))) or (not math.isfinite(float(s.weight))):
            continue
        sanitized.append(s)

    if len(sanitized) < 2:
        raise ValueError("Need at least two finite calibration samples")

    unique_errors = len({round(float(s.error), 12) for s in sanitized})
    if unique_errors < 2:
        raise ValueError("Calibration samples are degenerate")

    return sanitized
def _weighted_linear_fit(samples: list[CalibrationSample]) -> tuple[float, float]:
    if len(samples) < 2:
        raise ValueError("Need at least two calibration samples")

    sum_w = sum(max(0.0, s.weight) for s in samples)
    if sum_w <= 0:
        raise ValueError("Calibration sample weights must contain positive mass")

    sum_e = sum(max(0.0, s.weight) * s.error for s in samples)
    sum_z = sum(max(0.0, s.weight) * s.z_um for s in samples)
    sum_ee = sum(max(0.0, s.weight) * s.error * s.error for s in samples)
    sum_ez = sum(max(0.0, s.weight) * s.error * s.z_um for s in samples)

    denom = sum_w * sum_ee - sum_e * sum_e
    if denom == 0.0:
        raise ValueError("Calibration samples are degenerate")

    slope = (sum_w * sum_ez - sum_e * sum_z) / denom
    intercept = (sum_z - slope * sum_e) / sum_w
    if slope == 0.0:
        raise ValueError("Calibration slope is zero")
    return slope, intercept




def _weighted_z_reference(samples: list[CalibrationSample]) -> float:
    sum_w = sum(max(0.0, s.weight) for s in samples)
    if sum_w <= 0:
        raise ValueError("Calibration sample weights must contain positive mass")
    return sum(max(0.0, s.weight) * s.z_um for s in samples) / sum_w


def _center_samples_on_reference(
    samples: list[CalibrationSample],
    z_reference_um: float,
) -> list[CalibrationSample]:
    return [
        CalibrationSample(
            z_um=s.z_um - z_reference_um,
            error=s.error,
            weight=s.weight,
        )
        for s in samples
    ]


def _robust_seed_fit(samples: list[CalibrationSample], *, max_pairs: int = 500) -> tuple[float, float]:
    if len(samples) < 2:
        raise ValueError("Need at least two calibration samples")

    # Deterministic subsample: spread evenly across error-sorted samples to
    # avoid O(n²) blowup while still covering the full error range.  Using a
    # deterministic selection keeps the seed (and therefore the inlier set and
    # final calibration) reproducible across runs on the same data.
    if len(samples) > 80:
        sorted_by_error = sorted(samples, key=lambda s: s.error)
        stride = max(1, len(sorted_by_error) // 80)
        subset = sorted_by_error[::stride]
    else:
        subset = list(samples)

    best: tuple[float, float] | None = None
    best_med = float("inf")
    pairs_checked = 0
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            if pairs_checked >= max_pairs:
                break
            e0 = subset[i].error
            e1 = subset[j].error
            if e1 == e0:
                continue
            slope = (subset[j].z_um - subset[i].z_um) / (e1 - e0)
            if slope == 0.0:
                continue
            intercept = subset[i].z_um - slope * e0
            residuals = [abs(s.z_um - (slope * s.error + intercept)) for s in samples]
            residuals.sort()
            med = residuals[len(residuals) // 2]
            if med < best_med:
                best_med = med
                best = (slope, intercept)
            pairs_checked += 1
        if pairs_checked >= max_pairs:
            break
    if best is None:
        return _weighted_linear_fit(samples)
    return best




def _estimate_bidirectional_hysteresis(samples: list[CalibrationSample]) -> float:
    z_to_errors: dict[float, list[float]] = {}
    for s in samples:
        key = round(float(s.z_um), 3)
        z_to_errors.setdefault(key, []).append(float(s.error))
    hysteresis_deltas = [max(v) - min(v) for v in z_to_errors.values() if len(v) > 1]
    return max(hysteresis_deltas) if hysteresis_deltas else 0.0

def _fit_report(
    samples: list[CalibrationSample],
    slope: float,
    intercept: float,
    *,
    robust: bool,
    n_inliers: int,
    metric_samples: list[CalibrationSample] | None = None,
) -> CalibrationFitReport:
    error_at_focus = -intercept / slope
    errors = [s.error for s in samples]
    cal = FocusCalibration(
        error_at_focus=error_at_focus,
        error_to_um=slope,
        error_min=min(errors),
        error_max=max(errors),
    )

    metric_samples = samples if metric_samples is None else metric_samples
    weights = [max(0.0, s.weight) for s in metric_samples]
    w_sum = sum(weights)
    if w_sum <= 0:
        raise ValueError("Calibration sample weights must contain positive mass")

    z_mean = sum(w * s.z_um for w, s in zip(weights, metric_samples)) / w_sum
    ss_res = 0.0
    ss_tot = 0.0
    for w, s in zip(weights, metric_samples):
        pred = slope * s.error + intercept
        ss_res += w * ((s.z_um - pred) ** 2)
        ss_tot += w * ((s.z_um - z_mean) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    rmse = (ss_res / w_sum) ** 0.5

    return CalibrationFitReport(
        calibration=cal,
        intercept_um=intercept,
        r2=r2,
        rmse_um=rmse,
        n_samples=len(samples),
        n_inliers=n_inliers,
        robust=robust,
        recommended_z_range_um=(min(s.z_um for s in samples), max(s.z_um for s in samples)),
        hysteresis_um=_estimate_bidirectional_hysteresis(samples),
    )


def fit_linear_calibration_with_report(
    samples: list[CalibrationSample],
    *,
    robust: bool = False,
    outlier_threshold_um: float = 0.2,
) -> CalibrationFitReport:
    """Fit a relative offset model and return quality metrics.

    Even when sample `z_um` values are absolute stage positions, fitting is
    performed in a centered local frame so the resulting calibration remains a
    command-delta mapping usable across targets at different absolute Z levels.
    """

    samples = _sanitize_calibration_samples(samples)

    # Fit in a local Z frame to avoid large absolute-stage offsets skewing
    # intercept-derived error_at_focus estimates. This keeps slope unchanged.
    # Caveat: if the sweep is strongly asymmetric around true focus, the local
    # reference can introduce small bias in error_at_focus (symmetric sweeps are
    # recommended and are the GUI default).
    z_reference_um = _weighted_z_reference(samples)
    centered_samples = _center_samples_on_reference(samples, z_reference_um)

    slope, intercept = _weighted_linear_fit(centered_samples)
    n_inliers = len(centered_samples)
    # The samples used for R²/RMSE metrics must match the samples used for
    # the fit; otherwise the quality gate is unreliable and calibrations
    # that look good by R² can still produce bad control behavior.
    fit_inliers = centered_samples

    if robust:
        seed_slope, seed_intercept = _robust_seed_fit(centered_samples)
        inliers: list[CalibrationSample] = []
        for s in centered_samples:
            pred = seed_slope * s.error + seed_intercept
            if abs(s.z_um - pred) <= outlier_threshold_um:
                inliers.append(s)
        if len(inliers) >= 2:
            slope, intercept = _weighted_linear_fit(inliers)
            n_inliers = len(inliers)
            fit_inliers = inliers

    return _fit_report(
        centered_samples,
        slope,
        intercept,
        robust=robust,
        n_inliers=n_inliers,
        metric_samples=fit_inliers,
    )


def fit_linear_calibration(
    samples: list[CalibrationSample],
    *,
    robust: bool = False,
    outlier_threshold_um: float = 0.2,
) -> FocusCalibration:
    """Fit z = slope*error + intercept via weighted ordinary least squares."""

    report = fit_linear_calibration_with_report(
        samples,
        robust=robust,
        outlier_threshold_um=outlier_threshold_um,
    )
    return report.calibration


def auto_calibrate(
    camera: CameraInterface,
    stage: StageInterface,
    roi: Roi,
    *,
    z_min_um: float,
    z_max_um: float,
    n_steps: int,
    bidirectional: bool = True,
    settle_s: float = 0.05,
    should_stop: Callable[[], bool] | None = None,
    on_step: Callable[[int, int, float, float | None, bool], None] | None = None,
) -> list[CalibrationSample]:
    """Collect calibration samples from a deterministic stage sweep."""

    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if z_max_um <= z_min_um:
        raise ValueError("z_max_um must be greater than z_min_um")

    step = (z_max_um - z_min_um) / float(n_steps - 1)
    forward_targets = [z_min_um + i * step for i in range(n_steps)]
    targets = forward_targets
    if bidirectional:
        targets = forward_targets + list(reversed(forward_targets))

    out: list[CalibrationSample] = []
    failed_moves: list[tuple[float, Exception]] = []
    total_steps = len(targets)
    for i, target_z in enumerate(targets):
        if should_stop is not None and should_stop():
            raise RuntimeError("Calibration cancelled by user")

        step_index = i + 1
        try:
            stage.move_z_um(target_z)
        except Exception as exc:
            failed_moves.append((target_z, exc))
            if on_step is not None:
                on_step(step_index, total_steps, target_z, None, False)
            continue

        # Allow the stage to settle before capturing to avoid recording
        # frames while the piezo is still ringing.  Without this delay the
        # recorded Z and the PSF shape can be mismatched, making calibration
        # non-reproducible between runs.
        if settle_s > 0:
            time.sleep(settle_s)

        # Read the actual stage position BEFORE capturing so the recorded Z
        # corresponds to the frame we are about to acquire (not a later
        # position if the stage drifts or overshoots).
        measured_z = target_z
        try:
            measured_z = float(stage.get_z_um())
        except Exception:
            pass

        frame = camera.get_frame()
        err = astigmatic_error_signal(frame.image, roi)
        weight = roi_total_intensity(frame.image, roi)

        if on_step is not None:
            on_step(step_index, total_steps, target_z, measured_z, True)

        out.append(CalibrationSample(z_um=measured_z, error=err, weight=max(0.0, weight)))

    if len(out) < 2:
        if failed_moves:
            first_z, first_exc = failed_moves[0]
            raise RuntimeError(
                "Calibration sweep could not collect enough valid points: "
                f"{len(out)} succeeded, {len(failed_moves)} failed. "
                f"First failed move at z={first_z:+0.3f} um: {first_exc}"
            ) from first_exc
        raise RuntimeError(
            "Calibration sweep could not collect enough valid points; "
            "need at least 2 successful stage positions."
        )

    return out


def save_calibration_samples_csv(path: str | Path, samples: list[CalibrationSample]) -> None:
    """Write calibration sweep samples for later GUI/model reuse."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["z_um", "error", "weight"])
        writer.writeheader()
        for s in samples:
            writer.writerow({"z_um": s.z_um, "error": s.error, "weight": s.weight})


def load_calibration_samples_csv(path: str | Path) -> list[CalibrationSample]:
    """Read calibration sweep samples previously exported from the GUI."""

    in_path = Path(path)
    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: list[CalibrationSample] = []
        for row in reader:
            out.append(
                CalibrationSample(
                    z_um=float(row["z_um"]),
                    error=float(row["error"]),
                    weight=float(row.get("weight", "1.0")),
                )
            )
    return out


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    var_y = sum((y - y_mean) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / ((var_x * var_y) ** 0.5)


def calibration_quality_issues(
    samples: list[CalibrationSample],
    report: CalibrationFitReport,
    *,
    min_abs_corr: float = 0.2,
    min_error_span: float = 0.01,
    focus_margin_fraction: float = 0.1,
    max_bidirectional_hysteresis: float = 0.02,
) -> list[str]:
    """Return human-readable issues when a sweep is not safely usable for control."""

    if len(samples) < 2:
        return ["need at least 2 samples"]

    errors = [s.error for s in samples]
    z_vals = [s.z_um for s in samples]
    min_err = min(errors)
    max_err = max(errors)
    err_span = max_err - min_err

    issues: list[str] = []

    if err_span < min_error_span:
        issues.append(
            f"error span too small ({err_span:0.4f}); increase Z range or improve ROI SNR"
        )

    abs_corr = abs(_pearson_corr(z_vals, errors))
    if abs_corr < min_abs_corr:
        # Astigmatic curves are often locally non-linear around lobe transitions.
        # Keep this as advisory text while relying on fit+range checks for gating.
        issues.append(
            f"error-vs-Z is weakly correlated (|corr|={abs_corr:0.3f}); keep ROI centered and reduce sweep range around focus"
        )

    # For bidirectional sweeps (up/down), the same Z is sampled twice. Ensure
    # the error signal is reasonably consistent to catch backlash/hysteresis.
    z_to_errors: dict[float, list[float]] = {}
    for s in samples:
        key = round(float(s.z_um), 3)
        z_to_errors.setdefault(key, []).append(float(s.error))
    hysteresis_deltas = [max(v) - min(v) for v in z_to_errors.values() if len(v) > 1]
    if hysteresis_deltas and (max(hysteresis_deltas) > max_bidirectional_hysteresis):
        issues.append(
            "up/down sweep mismatch is high (possible backlash or stage settling issue); "
            "reduce step size, slow sweep, or tighten stage settling"
        )

    err0 = report.calibration.error_at_focus
    nearest_err_dist = min(abs(err - err0) for err in errors)
    # Be tolerant to slightly out-of-range fitted centers: astigmatic curves can
    # be asymmetric/noisy near the lobe crossover even when focus is bracketed.
    margin = max(0.02, err_span * focus_margin_fraction)
    near_focus_tolerance = max(0.02, err_span * 0.25)
    if (err0 < (min_err - margin) or err0 > (max_err + margin)) and (
        nearest_err_dist > near_focus_tolerance
    ):
        issues.append(
            "fitted focus lies outside sampled error range; sweep likely does not bracket focus"
        )

    return issues


def validate_calibration_sign(
    calibration: FocusCalibration,
    *,
    expected_positive_slope: bool = True,
) -> None:
    """Raise if fitted slope sign does not match the expected setup convention."""

    if expected_positive_slope and calibration.error_to_um <= 0:
        raise ValueError("Calibration slope sign is inverted for expected-positive setup")
    if (not expected_positive_slope) and calibration.error_to_um >= 0:
        raise ValueError("Calibration slope sign is inverted for expected-negative setup")


# ---------------------------------------------------------------------------
# Zhuang-model calibration (physics-based, wider range, more accurate)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ZhuangCalibrationSample:
    """Calibration sample with Gaussian PSF fit data for Zhuang model."""
    z_um: float
    error: float          # second-moment error signal
    weight: float
    sigma_x: float        # fitted Gaussian sigma_x (pixels)
    sigma_y: float        # fitted Gaussian sigma_y (pixels)
    ellipticity: float    # sigma_x / sigma_y
    fit_r2: float         # Gaussian fit R²


@dataclass(slots=True)
class ZhuangFocusCalibration:
    """Focus calibration using the Zhuang/Huang defocus model.

    Provides error_to_z_offset_um() for the control loop, using a precomputed
    lookup table derived from the physics-based model.  Falls back to the
    linear model interface so it's a drop-in replacement for FocusCalibration.
    """
    params: ZhuangEllipticityParams
    params_x: ZhuangParams
    params_y: ZhuangParams
    lookup: ZhuangLookupTable
    pixel_size_um: float = 1.0  # px→um conversion for sigma values

    # These two fields maintain interface compatibility with FocusCalibration
    error_at_focus: float = 0.0
    error_to_um: float = 1.0  # linear slope approximation for diagnostics

    def error_to_z_offset_um(self, error: float) -> float:
        """Map second-moment error signal to z offset using Zhuang lookup."""
        return self.lookup.error_to_z_offset_um(error)

    @property
    def usable_range_um(self) -> tuple[float, float] | None:
        return zhuang_usable_range(self.params)


@dataclass(slots=True)
class ZhuangCalibrationReport:
    """Quality report for a Zhuang model fit."""
    calibration: ZhuangFocusCalibration
    r2_x: float
    r2_y: float
    chi2_x: float
    chi2_y: float
    n_samples: int
    n_good_fits: int
    usable_range_um: tuple[float, float] | None
    linear_slope: float  # approximate slope at focus for comparison


def auto_calibrate_zhuang(
    camera: CameraInterface,
    stage: StageInterface,
    roi: Roi,
    *,
    z_min_um: float,
    z_max_um: float,
    n_steps: int,
    bidirectional: bool = True,
    settle_s: float = 0.05,
    theta: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_step: Callable[[int, int, float, float | None, bool], None] | None = None,
) -> list[ZhuangCalibrationSample]:
    """Collect calibration samples with full Gaussian PSF fitting.

    Like auto_calibrate() but also fits a 2D Gaussian at each Z step to
    extract sigma_x and sigma_y for the Zhuang defocus model.
    """
    from .focus_metric import extract_roi

    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if z_max_um <= z_min_um:
        raise ValueError("z_max_um must be greater than z_min_um")

    step = (z_max_um - z_min_um) / float(n_steps - 1)
    forward_targets = [z_min_um + i * step for i in range(n_steps)]
    targets = forward_targets
    if bidirectional:
        targets = forward_targets + list(reversed(forward_targets))

    out: list[ZhuangCalibrationSample] = []
    failed_moves: list[tuple[float, Exception]] = []
    total_steps = len(targets)

    for i, target_z in enumerate(targets):
        if should_stop is not None and should_stop():
            raise RuntimeError("Calibration cancelled by user")

        step_index = i + 1
        try:
            stage.move_z_um(target_z)
        except Exception as exc:
            failed_moves.append((target_z, exc))
            if on_step is not None:
                on_step(step_index, total_steps, target_z, None, False)
            continue

        if settle_s > 0:
            time.sleep(settle_s)

        measured_z = target_z
        try:
            measured_z = float(stage.get_z_um())
        except Exception:
            pass

        frame = camera.get_frame()
        err = astigmatic_error_signal(frame.image, roi)
        weight = roi_total_intensity(frame.image, roi)

        # Full Gaussian fit for Zhuang model
        patch = extract_roi(frame.image, roi)
        gauss = fit_gaussian_psf(patch, theta=theta)

        if gauss is not None and gauss.r_squared > 0.3:
            sx = gauss.sigma_x
            sy = gauss.sigma_y
            ell = gauss.ellipticity
            fit_r2 = gauss.r_squared
        else:
            sx = sy = ell = fit_r2 = math.nan

        if on_step is not None:
            on_step(step_index, total_steps, target_z, measured_z, True)

        out.append(ZhuangCalibrationSample(
            z_um=measured_z, error=err, weight=max(0.0, weight),
            sigma_x=sx, sigma_y=sy, ellipticity=ell, fit_r2=fit_r2,
        ))

    if len(out) < 2:
        if failed_moves:
            first_z, first_exc = failed_moves[0]
            raise RuntimeError(
                f"Calibration sweep failed: {len(out)} succeeded, "
                f"{len(failed_moves)} failed. First: z={first_z:+.3f} um: {first_exc}"
            ) from first_exc
        raise RuntimeError("Calibration sweep could not collect enough valid points.")

    return out


def fit_zhuang_calibration(
    samples: list[ZhuangCalibrationSample],
    pixel_size_um: float = 1.0,
    min_fit_r2: float = 0.5,
) -> ZhuangCalibrationReport:
    """Fit Zhuang defocus model to calibration sweep data.

    Args:
        samples: From auto_calibrate_zhuang()
        pixel_size_um: Camera pixel size in um (to convert sigma from px to um)
        min_fit_r2: Minimum R² of individual Gaussian fits to include

    Returns:
        ZhuangCalibrationReport with the calibration and quality metrics.
    """
    # Filter to samples with good Gaussian fits
    good = [s for s in samples if math.isfinite(s.sigma_x) and s.fit_r2 >= min_fit_r2]

    if len(good) < 10:
        raise ValueError(
            f"Only {len(good)} samples had good Gaussian fits (R² >= {min_fit_r2}); "
            f"need at least 10. Check ROI placement and PSF quality."
        )

    z = np.array([s.z_um for s in good])
    sx = np.array([s.sigma_x for s in good]) * pixel_size_um
    sy = np.array([s.sigma_y for s in good]) * pixel_size_um
    ell = sx / sy

    # Center Z for fitting
    z_center = float(np.mean(z))
    z_local = z - z_center

    # Fit individual axes
    params_x, dparams_x, (chi2_x, r2_x) = fit_zhuang_axis(z_local, sx)
    params_y, dparams_y, (chi2_y, r2_y) = fit_zhuang_axis(z_local, sy)

    # Build combined ellipticity model
    ell_params = ZhuangEllipticityParams.from_axis_params(params_x, params_y)

    # Refine ellipticity model by fitting directly to ellipticity data
    ell_params, _, _ = fit_zhuang_ellipticity(z_local, ell, initial=ell_params)

    # Build lookup table for fast runtime use
    lookup = build_lookup_table(ell_params)

    # Compute approximate linear slope at focus for diagnostics
    dz = 0.01
    e_plus = second_moment_error_from_ellipticity(
        float(zhuang_ellipticity(dz, ell_params))
    )
    e_minus = second_moment_error_from_ellipticity(
        float(zhuang_ellipticity(-dz, ell_params))
    )
    linear_slope = 2 * dz / (e_plus - e_minus) if e_plus != e_minus else 1.0

    cal = ZhuangFocusCalibration(
        params=ell_params,
        params_x=params_x,
        params_y=params_y,
        lookup=lookup,
        pixel_size_um=pixel_size_um,
        error_at_focus=0.0,
        error_to_um=linear_slope,
    )

    return ZhuangCalibrationReport(
        calibration=cal,
        r2_x=r2_x,
        r2_y=r2_y,
        chi2_x=chi2_x,
        chi2_y=chi2_y,
        n_samples=len(samples),
        n_good_fits=len(good),
        usable_range_um=zhuang_usable_range(ell_params),
        linear_slope=linear_slope,
    )



def save_zhuang_calibration_samples_csv(
    path: str | Path,
    samples: list[ZhuangCalibrationSample],
) -> None:
    """Write Zhuang calibration sweep samples to CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["z_um", "error", "weight", "sigma_x", "sigma_y", "ellipticity", "fit_r2"]
        )
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "z_um": s.z_um, "error": s.error, "weight": s.weight,
                "sigma_x": s.sigma_x, "sigma_y": s.sigma_y,
                "ellipticity": s.ellipticity, "fit_r2": s.fit_r2,
            })


def load_zhuang_calibration_samples_csv(path: str | Path) -> list[ZhuangCalibrationSample]:
    """Read Zhuang calibration sweep samples from CSV."""
    in_path = Path(path)
    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: list[ZhuangCalibrationSample] = []
        for row in reader:
            out.append(ZhuangCalibrationSample(
                z_um=float(row["z_um"]),
                error=float(row["error"]),
                weight=float(row.get("weight", "1.0")),
                sigma_x=float(row.get("sigma_x", "nan")),
                sigma_y=float(row.get("sigma_y", "nan")),
                ellipticity=float(row.get("ellipticity", "nan")),
                fit_r2=float(row.get("fit_r2", "nan")),
            ))
    return out


def fit_calibration_model(
    samples: list[CalibrationSample],
    *,
    model: str = "linear",
    robust: bool = False,
    outlier_threshold_um: float = 0.2,
) -> CalibrationFitReport:
    """Fit calibration with selectable model (linear/poly2/piecewise).

    Non-linear modes currently use a local-linear approximation around focus
    for runtime control compatibility and report residual quality gates.
    """
    if model == "linear":
        return fit_linear_calibration_with_report(samples, robust=robust, outlier_threshold_um=outlier_threshold_um)

    if model == "poly2":
        ss = _sanitize_calibration_samples(samples)
        z_reference_um = _weighted_z_reference(ss)
        centered = _center_samples_on_reference(ss, z_reference_um)
        n = len(centered)
        sum_w = sum(max(0.0, s.weight) for s in ss)
        if sum_w <= 0:
            raise ValueError("Calibration sample weights must contain positive mass")

        # Weighted quadratic model in local (centered) Z frame.
        e = [s.error for s in centered]
        z = [s.z_um for s in centered]
        w = [max(0.0, s.weight) for s in centered]
        X = np.vstack([np.array(e) ** 2, np.array(e), np.ones(n)]).T
        W = np.diag(np.array(w))
        beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ np.array(z))
        a, b, c = [float(v) for v in beta]

        # Estimate focus error e0 from z(e)=0 roots; choose root nearest weighted median error.
        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                raise ValueError("Polynomial calibration is degenerate near focus")
            e0 = -c / b
        else:
            disc = (b * b) - (4.0 * a * c)
            if disc < 0.0:
                e0 = sorted(e)[n // 2]
            else:
                sqrt_disc = math.sqrt(disc)
                r1 = (-b + sqrt_disc) / (2.0 * a)
                r2 = (-b - sqrt_disc) / (2.0 * a)
                e_med = sorted(e)[n // 2]
                e0 = r1 if abs(r1 - e_med) <= abs(r2 - e_med) else r2

        # Linearize around estimated focus so runtime control stays compatible
        # with the linear mapping interface.
        slope = 2.0 * a * e0 + b
        if slope == 0.0:
            slope = b if b != 0 else 1e-6
        intercept = -slope * e0
        return _fit_report(centered, slope, intercept, robust=robust, n_inliers=len(centered))

    if model == "piecewise":
        ss = _sanitize_calibration_samples(samples)
        z_reference_um = _weighted_z_reference(ss)
        centered = sorted(_center_samples_on_reference(ss, z_reference_um), key=lambda s: s.error)
        mid = len(centered) // 2
        left = centered[: max(2, mid)]
        right = centered[max(0, mid - 1):]
        l_slope, l_int = _weighted_linear_fit(left)
        r_slope, r_int = _weighted_linear_fit(right)

        # Estimate focus error from line intersection and collapse to one local slope.
        if abs(l_slope - r_slope) < 1e-12:
            e0 = centered[mid].error
        else:
            e0 = (r_int - l_int) / (l_slope - r_slope)

        slope = 0.5 * (l_slope + r_slope)
        z0 = 0.5 * ((l_slope * e0 + l_int) + (r_slope * e0 + r_int))
        intercept = z0 - slope * e0
        return _fit_report(centered, slope, intercept, robust=robust, n_inliers=len(centered))

    raise ValueError(f"Unsupported model: {model}")


def save_calibration_metadata_json(path: str | Path, metadata: CalibrationMetadata) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": metadata.objective,
        "roi_size": list(metadata.roi_size) if metadata.roi_size is not None else None,
        "exposure_ms": metadata.exposure_ms,
        "wavelength_nm": metadata.wavelength_nm,
        "camera_roi": list(metadata.camera_roi) if metadata.camera_roi is not None else None,
        "camera_binning": metadata.camera_binning,
        "stage_type": metadata.stage_type,
        "temperature_c": metadata.temperature_c,
        "created_at_unix_s": metadata.created_at_unix_s or time.time(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration_metadata_json(path: str | Path) -> CalibrationMetadata:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibrationMetadata(
        objective=str(payload.get("objective", "")),
        roi_size=tuple(payload["roi_size"]) if payload.get("roi_size") is not None else None,
        exposure_ms=payload.get("exposure_ms"),
        wavelength_nm=payload.get("wavelength_nm"),
        camera_roi=tuple(payload["camera_roi"]) if payload.get("camera_roi") is not None else None,
        camera_binning=payload.get("camera_binning"),
        stage_type=payload.get("stage_type"),
        temperature_c=payload.get("temperature_c"),
        created_at_unix_s=float(payload.get("created_at_unix_s", 0.0)),
    )
