from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .interfaces import Image2D


@dataclass(slots=True)
class Roi:
    """Axis-aligned region of interest in image coordinates."""

    x: int
    y: int
    width: int
    height: int

    def clamp(self, image_shape: tuple[int, int]) -> "Roi":
        h, w = image_shape
        x = min(max(0, self.x), max(0, w - 1))
        y = min(max(0, self.y), max(0, h - 1))
        width = min(self.width, w - x)
        height = min(self.height, h - y)
        if width <= 0 or height <= 0:
            raise ValueError("ROI does not intersect image")
        return Roi(x=x, y=y, width=width, height=height)


def _coerce_image_2d(image: Any) -> Image2D:
    if hasattr(image, "tolist") and callable(image.tolist):
        image = image.tolist()
    if isinstance(image, tuple):
        image = list(image)
    if not isinstance(image, list):
        raise TypeError("Image must be a 2D list/tuple or expose tolist()")

    out: Image2D = []
    width: int | None = None
    for row in image:
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list):
            raise TypeError("Image rows must be list/tuple")
        if width is None:
            width = len(row)
            if width == 0:
                raise ValueError("Empty image")
        elif len(row) != width:
            raise ValueError("Image rows must have equal length")
        out.append([float(v) for v in row])

    if not out:
        raise ValueError("Empty image")
    return out


def _image_shape(image: Image2D) -> tuple[int, int]:
    if not image or not image[0]:
        raise ValueError("Empty image")
    return len(image), len(image[0])


def extract_roi(image: Image2D, roi: Roi) -> Image2D:
    safe_image = _coerce_image_2d(image)
    h, w = _image_shape(safe_image)
    safe_roi = roi.clamp((h, w))
    return [
        row[safe_roi.x : safe_roi.x + safe_roi.width]
        for row in safe_image[safe_roi.y : safe_roi.y + safe_roi.height]
    ]


def _astigmatic_error_signal_numpy(patch: Image2D) -> float:
    try:
        import numpy as np
    except Exception:
        return _astigmatic_error_signal_python(patch)

    arr = np.asarray(patch, dtype=float)

    # Background-subtract so uniform illumination doesn't dilute the
    # intensity-weighted second moments.  Without this, background pixels
    # compress the (var_x - var_y)/(var_x + var_y) ratio toward zero,
    # making the error signal ~20× weaker than the Zhuang model expects.
    bg = float(np.percentile(arr, 10))
    arr = np.maximum(arr - bg, 0.0)

    total = float(arr.sum())
    if total <= 0:
        return 0.0

    y_idx, x_idx = np.indices(arr.shape, dtype=float)
    cx = float((x_idx * arr).sum() / total)
    cy = float((y_idx * arr).sum() / total)

    var_x = float((((x_idx - cx) ** 2) * arr).sum() / total)
    var_y = float((((y_idx - cy) ** 2) * arr).sum() / total)

    denom = var_x + var_y
    if denom == 0:
        return 0.0
    return (var_x - var_y) / denom


def _astigmatic_error_signal_python(patch: Image2D) -> float:
    # Flatten to estimate background (10th percentile).
    all_vals = sorted(v for row in patch for v in row)
    if not all_vals:
        return 0.0
    bg = all_vals[max(0, len(all_vals) // 10)]

    total = 0.0
    for row in patch:
        for val in row:
            total += max(0.0, val - bg)
    if total <= 0:
        return 0.0

    sum_x = 0.0
    sum_y = 0.0
    for y, row in enumerate(patch):
        for x, val in enumerate(row):
            w = max(0.0, val - bg)
            sum_x += x * w
            sum_y += y * w

    cx = sum_x / total
    cy = sum_y / total

    var_x = 0.0
    var_y = 0.0
    for y, row in enumerate(patch):
        for x, val in enumerate(row):
            w = max(0.0, val - bg)
            var_x += ((x - cx) ** 2) * w
            var_y += ((y - cy) ** 2) * w

    var_x /= total
    var_y /= total

    denom = var_x + var_y
    if denom == 0:
        return 0.0
    return (var_x - var_y) / denom


def _centroid_python(patch: Image2D) -> tuple[float, float]:
    """Return (cx, cy) intensity-weighted centroid of the patch."""
    total = sum(sum(row) for row in patch)
    if total <= 0:
        h = len(patch)
        w = len(patch[0]) if patch else 0
        return (w - 1) / 2.0, (h - 1) / 2.0
    sum_x = 0.0
    sum_y = 0.0
    for y, row in enumerate(patch):
        for x, val in enumerate(row):
            sum_x += x * val
            sum_y += y * val
    return sum_x / total, sum_y / total


def centroid_near_edge(image: Image2D, roi: Roi, margin_px: float) -> bool:
    """Return True if the intensity centroid is within *margin_px* of the ROI boundary.

    This is a guard against truncated PSFs: when the bead drifts near the ROI
    edge, the second-moment error signal becomes biased and can drive runaway
    corrections.
    """
    if margin_px <= 0:
        return False
    patch = extract_roi(image, roi)
    h = len(patch)
    w = len(patch[0]) if patch else 0
    if h == 0 or w == 0:
        return True
    cx, cy = _centroid_python(patch)
    if cx < margin_px or cx > (w - 1) - margin_px:
        return True
    if cy < margin_px or cy > (h - 1) - margin_px:
        return True
    return False


def roi_total_intensity(image: Image2D, roi: Roi) -> float:
    patch = extract_roi(image, roi)
    return float(sum(sum(row) for row in patch))


def astigmatic_error_signal(image: Image2D, roi: Roi) -> float:
    """Return focus error based on anisotropic second moments.

    Uses a NumPy-accelerated path when NumPy is available; otherwise falls back
    to a pure-Python implementation.
    """

    patch = extract_roi(image, roi)
    return _astigmatic_error_signal_numpy(patch)


# ---------------------------------------------------------------------------
# Gaussian PSF fitting for calibration sweeps
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GaussianPsfResult:
    """Result of a 2D Gaussian PSF fit.

    Parameters follow the convention of Huang et al. (2008):
      sigma_x, sigma_y are the 1/e^2 half-widths along the principal axes.
      ellipticity = sigma_x / sigma_y
    """
    x: float
    y: float
    sigma_x: float
    sigma_y: float
    amplitude: float
    offset: float
    theta: float  # rotation angle (radians)
    ellipticity: float  # sigma_x / sigma_y
    r_squared: float


def _gaussian_2d(coords: tuple, x0: float, y0: float, sigma_x: float, sigma_y: float,
                 amplitude: float, offset: float, theta: float) -> Any:
    """Rotated 2D Gaussian for scipy curve_fit."""
    xv, yv = coords
    cos_t = float(__import__('math').cos(theta))
    sin_t = float(__import__('math').sin(theta))
    xr = cos_t * (xv - x0) + sin_t * (yv - y0)
    yr = -sin_t * (xv - x0) + cos_t * (yv - y0)
    try:
        import numpy as np
        g = amplitude * np.exp(-0.5 * ((xr / sigma_x)**2 + (yr / sigma_y)**2)) + offset
    except Exception:
        import math
        g = [[amplitude * math.exp(-0.5 * ((xr[i][j] / sigma_x)**2 + (yr[i][j] / sigma_y)**2)) + offset
              for j in range(len(xr[0]))] for i in range(len(xr))]
    return g


def fit_gaussian_psf(patch: Image2D, theta: float | None = None) -> GaussianPsfResult | None:
    """Fit a rotated 2D Gaussian to an image patch.

    This is used during calibration sweeps (not real-time control) to extract
    sigma_x and sigma_y for the Zhuang defocus model.

    Args:
        patch: 2D image data (ROI crop)
        theta: Fixed rotation angle. If None, theta is fitted freely.

    Returns:
        GaussianPsfResult or None if fitting fails.
    """
    try:
        import numpy as np
        import scipy.optimize as opt
    except ImportError:
        return None

    patch = _coerce_image_2d(patch)
    arr = np.asarray(patch, dtype=float)
    h, w = arr.shape

    if h < 3 or w < 3:
        return None

    # Initial guess from moments
    total = arr.sum()
    if total <= 0:
        return None

    y_idx, x_idx = np.indices(arr.shape, dtype=float)
    x0 = float((x_idx * arr).sum() / total)
    y0 = float((y_idx * arr).sum() / total)
    var_x = float(((x_idx - x0)**2 * arr).sum() / total)
    var_y = float(((y_idx - y0)**2 * arr).sum() / total)

    offset_guess = float(np.percentile(arr, 10))
    amp_guess = float(arr.max()) - offset_guess
    sx_guess = max(0.5, float(np.sqrt(var_x)))
    sy_guess = max(0.5, float(np.sqrt(var_y)))

    coords = (x_idx, y_idx)

    try:
        if theta is not None:
            # Fixed theta: fit 6 parameters
            def model_fixed(coords, x0, y0, sx, sy, amp, off):
                return np.asarray(_gaussian_2d(coords, x0, y0, sx, sy, amp, off, theta)).ravel()

            p0 = [x0, y0, sx_guess, sy_guess, amp_guess, offset_guess]
            bounds_lo = [0, 0, 0.3, 0.3, 0, -np.inf]
            bounds_hi = [w, h, w, h, np.inf, np.inf]
            popt, _ = opt.curve_fit(model_fixed, coords, arr.ravel(), p0=p0,
                                    bounds=(bounds_lo, bounds_hi), maxfev=10000)
            x_fit, y_fit, sx_fit, sy_fit, amp_fit, off_fit = popt
            theta_fit = theta
        else:
            # Free theta: fit 7 parameters
            def model_free(coords, x0, y0, sx, sy, amp, off, th):
                return np.asarray(_gaussian_2d(coords, x0, y0, sx, sy, amp, off, th)).ravel()

            p0 = [x0, y0, sx_guess, sy_guess, amp_guess, offset_guess, 0.0]
            bounds_lo = [0, 0, 0.3, 0.3, 0, -np.inf, -np.pi]
            bounds_hi = [w, h, w, h, np.inf, np.inf, np.pi]
            popt, _ = opt.curve_fit(model_free, coords, arr.ravel(), p0=p0,
                                    bounds=(bounds_lo, bounds_hi), maxfev=10000)
            x_fit, y_fit, sx_fit, sy_fit, amp_fit, off_fit, theta_fit = popt

    except (RuntimeError, ValueError):
        return None

    # Ensure sigma_x >= sigma_y by convention (swap + rotate if needed)
    if sx_fit < sy_fit:
        sx_fit, sy_fit = sy_fit, sx_fit
        theta_fit += np.pi / 2

    # R^2
    model_vals = np.asarray(_gaussian_2d(coords, x_fit, y_fit, sx_fit, sy_fit,
                                          amp_fit, off_fit, theta_fit))
    ss_res = float(np.sum((arr - model_vals)**2))
    ss_tot = float(np.sum((arr - np.mean(arr))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ell = sx_fit / sy_fit if sy_fit > 0 else float("inf")

    return GaussianPsfResult(
        x=x_fit, y=y_fit,
        sigma_x=sx_fit, sigma_y=sy_fit,
        amplitude=amp_fit, offset=off_fit,
        theta=theta_fit, ellipticity=ell,
        r_squared=r2,
    )
