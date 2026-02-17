from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .calibration import FocusCalibration, ZhuangFocusCalibration
from .focus_metric import Roi, astigmatic_error_signal, centroid_near_edge, roi_total_intensity
from .interfaces import CameraFrame, CameraInterface, StageInterface

# Either calibration type works â€” both provide error_to_z_offset_um().
CalibrationLike = FocusCalibration | ZhuangFocusCalibration


class AutofocusState(str, Enum):
    IDLE = "IDLE"
    CALIBRATED_READY = "CALIBRATED_READY"
    LOCKING = "LOCKING"
    LOCKED = "LOCKED"
    DEGRADED = "DEGRADED"
    RECOVERY = "RECOVERY"
    FAULT = "FAULT"


@dataclass(slots=True)
class AutofocusConfig:
    roi: Roi
    loop_hz: float = 30.0
    # Cap effective control-step dt to avoid large corrective jumps after
    # temporary stalls (GUI pauses, GC, device hiccups).
    max_dt_s: float = 0.2
    # PID control gains, in units of um of stage command per um equivalent error.
    kp: float = 0.6
    ki: float = 0.15
    kd: float = 0.0
    max_step_um: float = 0.25
    integral_limit_um: float = 2.0
    stage_min_um: float | None = None
    stage_max_um: float | None = None
    # Safety clamp around initial lock position to avoid runaway absolute jumps.
    max_abs_excursion_um: float | None = 5.0
    # Freeze control updates when ROI total intensity drops below threshold.
    min_roi_intensity: float | None = None
    # Additional quality gates before applying corrections.
    min_roi_variance: float | None = None
    max_saturated_fraction: float | None = None
    saturation_level: float = 65535.0
    # Exponential moving average smoothing factor for the error signal.
    # 0.0 = no filtering (raw error used), 1.0 = ignore new measurements.
    error_alpha: float = 0.0
    # Derivative filtering coefficient (EMA on d(error)/dt).
    derivative_alpha: float = 0.7
    # Compensate stage response delay by decaying D term based on command age.
    stage_latency_s: float = 0.0
    # Reject frames when PSF centroid is within this many pixels of the ROI
    # boundary, to avoid biased second moments from a truncated PSF.
    edge_margin_px: float = 0.0
    # Do not issue stage moves smaller than this threshold (um) to reduce
    # high-frequency dithering/oscillation near focus.
    command_deadband_um: float = 0.005
    # Slew limit in um/s for commanded target changes.
    max_slew_rate_um_per_s: float | None = None
    # Setpoint locking + optional guarded recentering.
    lock_setpoint: bool = True
    recenter_alpha: float = 0.0
    # Guardrail: reject control updates when measured error is outside the
    # calibration domain. This prevents lookup extrapolation/clamping from
    # producing large wrong-way moves when changing ROI/target.
    calibration_error_margin: float = 0.02


@dataclass(slots=True)
class AutofocusSample:
    timestamp_s: float
    error: float
    error_um: float
    stage_z_um: float
    commanded_z_um: float
    roi_total_intensity: float
    control_applied: bool
    confidence_ok: bool
    state: AutofocusState
    loop_latency_ms: float


class AstigmaticAutofocusController:
    """Closed-loop focus controller for a single astigmatic PSF target."""

    def __init__(
        self,
        camera: CameraInterface,
        stage: StageInterface,
        config: AutofocusConfig,
        calibration: CalibrationLike,
        initial_integral_um: float = 0.0,
    ) -> None:
        self._camera = camera
        self._stage = stage
        self._config = config
        self._validate_config()
        self._calibration = calibration
        self._integral_um = initial_integral_um
        self._filtered_error_um: float | None = None
        self._last_frame_ts: float | None = None
        self._z_lock_center_um: float | None = None
        self._last_error_um: float | None = None
        self._filtered_derivative: float = 0.0
        self._last_command_time_s: float | None = None
        self._last_commanded_z_um: float | None = None
        self._setpoint_error: float | None = None
        self._state = AutofocusState.CALIBRATED_READY
        self._degraded_count = 0
        self._config_lock = threading.RLock()

    @property
    def loop_hz(self) -> float:
        with self._config_lock:
            return self._config.loop_hz

    @property
    def calibration(self) -> CalibrationLike:
        return self._calibration

    @calibration.setter
    def calibration(self, value: CalibrationLike) -> None:
        self._calibration = value
        self._state = AutofocusState.CALIBRATED_READY

    def get_config_snapshot(self) -> AutofocusConfig:
        with self._config_lock:
            return copy.deepcopy(self._config)

    def update_config(self, **kwargs) -> None:
        with self._config_lock:
            for key, value in kwargs.items():
                if not hasattr(self._config, key):
                    raise AttributeError(f"Unknown AutofocusConfig field: {key}")
                setattr(self._config, key, value)
            self._validate_config()

    def update_roi(self, roi: Roi) -> None:
        self.update_config(roi=roi)
        self.reset_lock_state()

    def reset_lock_state(self) -> None:
        """Clear control memory so lock is re-acquired cleanly after ROI changes."""
        self._integral_um = 0.0
        self._filtered_error_um = None
        self._last_error_um = None
        self._filtered_derivative = 0.0
        self._z_lock_center_um = None
        self._last_commanded_z_um = None
        self._last_command_time_s = None
        self._setpoint_error = None
        self._last_frame_ts = None
        self._degraded_count = 0
        self._state = AutofocusState.CALIBRATED_READY

    def _validate_config(self) -> None:
        if self._config.loop_hz <= 0:
            raise ValueError("loop_hz must be > 0")
        if self._config.max_dt_s <= 0:
            raise ValueError("max_dt_s must be > 0")
        if self._config.max_step_um < 0:
            raise ValueError("max_step_um must be >= 0")
        if self._config.integral_limit_um < 0:
            raise ValueError("integral_limit_um must be >= 0")
        if not 0.0 <= self._config.error_alpha <= 1.0:
            raise ValueError("error_alpha must be in [0.0, 1.0]")
        if not 0.0 <= self._config.derivative_alpha <= 1.0:
            raise ValueError("derivative_alpha must be in [0.0, 1.0]")
        if self._config.edge_margin_px < 0:
            raise ValueError("edge_margin_px must be >= 0")
        if self._config.max_abs_excursion_um is not None and self._config.max_abs_excursion_um < 0:
            raise ValueError("max_abs_excursion_um must be >= 0 when provided")
        if self._config.command_deadband_um < 0:
            raise ValueError("command_deadband_um must be >= 0")
        if self._config.max_slew_rate_um_per_s is not None and self._config.max_slew_rate_um_per_s <= 0:
            raise ValueError("max_slew_rate_um_per_s must be > 0 when provided")
        if self._config.calibration_error_margin < 0:
            raise ValueError("calibration_error_margin must be >= 0")

    def _is_error_in_calibration_domain(self, error: float, config: AutofocusConfig) -> bool:
        checker = getattr(self._calibration, "is_error_in_range", None)
        if callable(checker):
            try:
                return bool(checker(error, margin=config.calibration_error_margin))
            except Exception:
                pass

        lookup = getattr(self._calibration, "lookup", None)
        error_values = getattr(lookup, "error_values", None)
        if error_values is None:
            return True
        if len(error_values) < 2:
            return True
        lo = float(error_values[0]) - config.calibration_error_margin
        hi = float(error_values[-1]) + config.calibration_error_margin
        return lo <= error <= hi


    def _calibration_error_at_focus(self) -> float:
        try:
            return float(getattr(self._calibration, "error_at_focus", 0.0))
        except Exception:
            return 0.0

    def _apply_limits(self, target_z_um: float, config: AutofocusConfig) -> float:
        if self._z_lock_center_um is not None and config.max_abs_excursion_um is not None:
            excursion = float(config.max_abs_excursion_um)
            target_z_um = max(self._z_lock_center_um - excursion, min(self._z_lock_center_um + excursion, target_z_um))
        if config.stage_min_um is not None:
            target_z_um = max(config.stage_min_um, target_z_um)
        if config.stage_max_um is not None:
            target_z_um = min(config.stage_max_um, target_z_um)
        return target_z_um

    def _slew_limit(self, proposed_z_um: float, dt_s: float, current_z: float, config: AutofocusConfig) -> float:
        if config.max_slew_rate_um_per_s is None:
            return proposed_z_um
        anchor = self._last_commanded_z_um if self._last_commanded_z_um is not None else current_z
        max_delta = config.max_slew_rate_um_per_s * max(0.0, dt_s)
        return max(anchor - max_delta, min(anchor + max_delta, proposed_z_um))

    def _roi_confidence_ok(self, image, total_intensity: float, config: AutofocusConfig) -> bool:
        if config.min_roi_intensity is not None and total_intensity < config.min_roi_intensity:
            return False

        try:
            import numpy as np

            arr = np.asarray(image, dtype=float)
            if arr.ndim != 2:
                return False
            y0 = max(0, int(config.roi.y))
            x0 = max(0, int(config.roi.x))
            y1 = min(arr.shape[0], y0 + int(config.roi.height))
            x1 = min(arr.shape[1], x0 + int(config.roi.width))
            if y1 <= y0 or x1 <= x0:
                return False
            patch = arr[y0:y1, x0:x1]
            if patch.size == 0:
                return False
            pixels = [float(v) for v in patch.ravel()]
        except Exception:
            patch = [row[config.roi.x : config.roi.x + config.roi.width] for row in image[config.roi.y : config.roi.y + config.roi.height]]
            if len(patch) == 0 or len(patch[0]) == 0:
                return False
            pixels = [float(v) for r in patch for v in r]

        if not pixels:
            return False
        mean = sum(pixels) / len(pixels)
        var = sum((v - mean) ** 2 for v in pixels) / len(pixels)
        if config.min_roi_variance is not None and var < config.min_roi_variance:
            return False
        if config.max_saturated_fraction is not None:
            sat = sum(1 for v in pixels if v >= config.saturation_level)
            if sat / len(pixels) > config.max_saturated_fraction:
                return False
        return True

    def run_step(self, dt_s: float | None = None, frame: CameraFrame | None = None) -> AutofocusSample:
        loop_start = time.monotonic()
        if frame is None:
            frame = self._camera.get_frame()
        current_z = self._stage.get_z_um()
        if self._z_lock_center_um is None:
            self._z_lock_center_um = float(current_z)

        config = self.get_config_snapshot()

        if dt_s is None:
            dt_s = 1.0 / config.loop_hz
        dt_s = max(0.0, min(float(dt_s), config.max_dt_s))

        # Guard: skip duplicate frames (same timestamp as previous).
        if self._last_frame_ts is not None and frame.timestamp_s == self._last_frame_ts:
            self._state = AutofocusState.DEGRADED
            return AutofocusSample(frame.timestamp_s, 0.0, 0.0, current_z, current_z, 0.0, False, False, self._state, (time.monotonic() - loop_start) * 1e3)
        self._last_frame_ts = frame.timestamp_s

        total_intensity = roi_total_intensity(frame.image, config.roi)

        confidence_ok = self._roi_confidence_ok(frame.image, total_intensity, config)
        if config.edge_margin_px > 0 and centroid_near_edge(frame.image, config.roi, config.edge_margin_px):
            confidence_ok = False

        if not confidence_ok:
            self._degraded_count += 1
            self._state = AutofocusState.RECOVERY if self._degraded_count > 8 else AutofocusState.DEGRADED
            return AutofocusSample(frame.timestamp_s, 0.0, 0.0, current_z, current_z, total_intensity, False, False, self._state, (time.monotonic() - loop_start) * 1e3)
        self._degraded_count = 0
        self._config_lock = threading.RLock()

        error = astigmatic_error_signal(frame.image, config.roi)
        if not self._is_error_in_calibration_domain(error, config):
            self._degraded_count += 1
            self._state = AutofocusState.RECOVERY if self._degraded_count > 8 else AutofocusState.DEGRADED
            return AutofocusSample(
                frame.timestamp_s,
                error,
                0.0,
                current_z,
                current_z,
                total_intensity,
                False,
                False,
                self._state,
                (time.monotonic() - loop_start) * 1e3,
            )

        if config.lock_setpoint:
            focus_error = self._calibration_error_at_focus()
            # ROI-dependent background/aberration can shift the raw second-moment
            # error baseline. Lock the per-ROI bias at engagement so calibration
            # remains reusable across targets while preserving zero correction at
            # the current Z when lock starts.
            if self._setpoint_error is None:
                self._setpoint_error = error - focus_error
            error -= self._setpoint_error
        elif config.recenter_alpha > 0:
            if self._setpoint_error is None:
                self._setpoint_error = error
            self._setpoint_error = config.recenter_alpha * self._setpoint_error + (1.0 - config.recenter_alpha) * error
            error -= self._setpoint_error
        else:
            self._setpoint_error = None

        error_um = self._calibration.error_to_z_offset_um(error)
        if not math.isfinite(float(error_um)):
            self._state = AutofocusState.FAULT
            raise RuntimeError("Non-finite autofocus error encountered; check ROI/calibration")

        alpha = config.error_alpha
        if 0.0 < alpha < 1.0 and self._filtered_error_um is not None:
            error_um = alpha * self._filtered_error_um + (1.0 - alpha) * error_um
        self._filtered_error_um = error_um

        derivative = 0.0
        if self._last_error_um is not None and dt_s > 0:
            derivative = (error_um - self._last_error_um) / dt_s
        self._last_error_um = error_um
        d_alpha = config.derivative_alpha
        self._filtered_derivative = d_alpha * self._filtered_derivative + (1.0 - d_alpha) * derivative
        d_term = self._filtered_derivative
        if self._last_command_time_s is not None and config.stage_latency_s > 0:
            age = max(0.0, time.monotonic() - self._last_command_time_s)
            d_term *= math.exp(-age / config.stage_latency_s)

        candidate_integral = self._integral_um + error_um * dt_s
        candidate_integral = max(-config.integral_limit_um, min(config.integral_limit_um, candidate_integral))

        correction = -(
            config.kp * error_um
            + config.ki * candidate_integral
            + config.kd * d_term
        )
        correction = max(-config.max_step_um, min(config.max_step_um, correction))

        if abs(correction) <= config.command_deadband_um:
            self._state = AutofocusState.LOCKED
            return AutofocusSample(frame.timestamp_s, error, error_um, current_z, current_z, total_intensity, False, True, self._state, (time.monotonic() - loop_start) * 1e3)

        raw_target = current_z + correction
        slew_target = self._slew_limit(raw_target, dt_s, current_z, config)
        commanded_z = self._apply_limits(slew_target, config)

        # Anti-windup: commit integral only when not saturated by output limits.
        saturated = commanded_z != raw_target
        if not saturated:
            self._integral_um = candidate_integral

        self._stage.move_z_um(commanded_z)
        self._last_commanded_z_um = commanded_z
        self._last_command_time_s = time.monotonic()

        self._state = AutofocusState.LOCKING if abs(error_um) > config.command_deadband_um else AutofocusState.LOCKED
        return AutofocusSample(
            timestamp_s=frame.timestamp_s,
            error=error,
            error_um=error_um,
            stage_z_um=current_z,
            commanded_z_um=commanded_z,
            roi_total_intensity=total_intensity,
            control_applied=True,
            confidence_ok=True,
            state=self._state,
            loop_latency_ms=(time.monotonic() - loop_start) * 1e3,
        )

    def run(self, duration_s: float) -> list[AutofocusSample]:
        samples: list[AutofocusSample] = []
        loop_dt = 1.0 / self.loop_hz
        end = time.monotonic() + duration_s
        last_step_start: float | None = None
        while time.monotonic() < end:
            step_start = time.monotonic()
            dt_s = loop_dt if last_step_start is None else max(0.0, step_start - last_step_start)
            samples.append(self.run_step(dt_s=dt_s))
            last_step_start = step_start
            elapsed = time.monotonic() - step_start
            if elapsed < loop_dt:
                time.sleep(loop_dt - elapsed)
        return samples


class AutofocusWorker:
    """Background real-time autofocus worker."""

    def __init__(
        self,
        controller: AstigmaticAutofocusController,
        on_sample: Callable[[AutofocusSample], None] | None = None,
    ) -> None:
        self._controller = controller
        self._on_sample = on_sample
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_error: Exception | None = None

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_evt.clear()
            self._last_error = None
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self, *, wait: bool = True) -> None:
        self._stop_evt.set()
        if not wait:
            return
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        dt = 1.0 / self._controller.loop_hz
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            try:
                sample = self._controller.run_step(dt_s=dt)
                if self._on_sample is not None:
                    self._on_sample(sample)
            except Exception as exc:  # pragma: no cover - exercised by tests indirectly
                self._last_error = exc
                self._stop_evt.set()
                return
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
