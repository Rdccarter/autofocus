from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .calibration import FocusCalibration, ZhuangFocusCalibration
from .focus_metric import Roi, astigmatic_error_signal, centroid_near_edge, roi_total_intensity
from .interfaces import CameraInterface, StageInterface

# Either calibration type works — both provide error_to_z_offset_um().
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
    command_deadband_um: float = 0.02
    # Slew limit in um/s for commanded target changes.
    max_slew_rate_um_per_s: float | None = None
    # Setpoint locking + optional guarded recentering.
    lock_setpoint: bool = True
    recenter_alpha: float = 0.0


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
        self._setpoint_error: float = 0.0
        self._state = AutofocusState.CALIBRATED_READY
        self._degraded_count = 0

    @property
    def loop_hz(self) -> float:
        return self._config.loop_hz

    @property
    def calibration(self) -> CalibrationLike:
        return self._calibration

    @calibration.setter
    def calibration(self, value: CalibrationLike) -> None:
        self._calibration = value
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

    def _apply_limits(self, target_z_um: float) -> float:
        if self._z_lock_center_um is not None and self._config.max_abs_excursion_um is not None:
            excursion = float(self._config.max_abs_excursion_um)
            target_z_um = max(self._z_lock_center_um - excursion, min(self._z_lock_center_um + excursion, target_z_um))
        if self._config.stage_min_um is not None:
            target_z_um = max(self._config.stage_min_um, target_z_um)
        if self._config.stage_max_um is not None:
            target_z_um = min(self._config.stage_max_um, target_z_um)
        return target_z_um

    def _slew_limit(self, proposed_z_um: float, dt_s: float, current_z: float) -> float:
        if self._config.max_slew_rate_um_per_s is None:
            return proposed_z_um
        anchor = self._last_commanded_z_um if self._last_commanded_z_um is not None else current_z
        max_delta = self._config.max_slew_rate_um_per_s * max(0.0, dt_s)
        return max(anchor - max_delta, min(anchor + max_delta, proposed_z_um))

    def _roi_confidence_ok(self, image, total_intensity: float) -> bool:
        if self._config.min_roi_intensity is not None and total_intensity < self._config.min_roi_intensity:
            return False
        patch = [row[self._config.roi.x : self._config.roi.x + self._config.roi.width] for row in image[self._config.roi.y : self._config.roi.y + self._config.roi.height]]
        if not patch or not patch[0]:
            return False
        pixels = [float(v) for r in patch for v in r]
        if not pixels:
            return False
        mean = sum(pixels) / len(pixels)
        var = sum((v - mean) ** 2 for v in pixels) / len(pixels)
        if self._config.min_roi_variance is not None and var < self._config.min_roi_variance:
            return False
        if self._config.max_saturated_fraction is not None:
            sat = sum(1 for v in pixels if v >= self._config.saturation_level)
            if sat / len(pixels) > self._config.max_saturated_fraction:
                return False
        return True

    def run_step(self, dt_s: float | None = None) -> AutofocusSample:
        loop_start = time.monotonic()
        frame = self._camera.get_frame()
        current_z = self._stage.get_z_um()
        if self._z_lock_center_um is None:
            self._z_lock_center_um = float(current_z)

        if dt_s is None:
            dt_s = 1.0 / self._config.loop_hz
        dt_s = max(0.0, min(float(dt_s), self._config.max_dt_s))

        # Guard: skip duplicate frames (same timestamp as previous).
        if self._last_frame_ts is not None and frame.timestamp_s == self._last_frame_ts:
            self._state = AutofocusState.DEGRADED
            return AutofocusSample(frame.timestamp_s, 0.0, 0.0, current_z, current_z, 0.0, False, False, self._state, (time.monotonic() - loop_start) * 1e3)
        self._last_frame_ts = frame.timestamp_s

        total_intensity = roi_total_intensity(frame.image, self._config.roi)

        confidence_ok = self._roi_confidence_ok(frame.image, total_intensity)
        if self._config.edge_margin_px > 0 and centroid_near_edge(frame.image, self._config.roi, self._config.edge_margin_px):
            confidence_ok = False

        if not confidence_ok:
            self._degraded_count += 1
            self._state = AutofocusState.RECOVERY if self._degraded_count > 8 else AutofocusState.DEGRADED
            return AutofocusSample(frame.timestamp_s, 0.0, 0.0, current_z, current_z, total_intensity, False, False, self._state, (time.monotonic() - loop_start) * 1e3)
        self._degraded_count = 0

        error = astigmatic_error_signal(frame.image, self._config.roi)
        if self._config.lock_setpoint:
            error -= self._setpoint_error
        elif self._config.recenter_alpha > 0:
            self._setpoint_error = self._config.recenter_alpha * self._setpoint_error + (1.0 - self._config.recenter_alpha) * error
            error -= self._setpoint_error

        error_um = self._calibration.error_to_z_offset_um(error)
        if not math.isfinite(float(error_um)):
            self._state = AutofocusState.FAULT
            raise RuntimeError("Non-finite autofocus error encountered; check ROI/calibration")

        alpha = self._config.error_alpha
        if 0.0 < alpha < 1.0 and self._filtered_error_um is not None:
            error_um = alpha * self._filtered_error_um + (1.0 - alpha) * error_um
        self._filtered_error_um = error_um

        derivative = 0.0
        if self._last_error_um is not None and dt_s > 0:
            derivative = (error_um - self._last_error_um) / dt_s
        self._last_error_um = error_um
        d_alpha = self._config.derivative_alpha
        self._filtered_derivative = d_alpha * self._filtered_derivative + (1.0 - d_alpha) * derivative
        d_term = self._filtered_derivative
        if self._last_command_time_s is not None and self._config.stage_latency_s > 0:
            age = max(0.0, time.monotonic() - self._last_command_time_s)
            d_term *= math.exp(-age / self._config.stage_latency_s)

        candidate_integral = self._integral_um + error_um * dt_s
        candidate_integral = max(-self._config.integral_limit_um, min(self._config.integral_limit_um, candidate_integral))

        correction = -(
            self._config.kp * error_um
            + self._config.ki * candidate_integral
            + self._config.kd * d_term
        )
        correction = max(-self._config.max_step_um, min(self._config.max_step_um, correction))

        if abs(correction) <= self._config.command_deadband_um:
            self._state = AutofocusState.LOCKED
            return AutofocusSample(frame.timestamp_s, error, error_um, current_z, current_z, total_intensity, False, True, self._state, (time.monotonic() - loop_start) * 1e3)

        raw_target = current_z + correction
        slew_target = self._slew_limit(raw_target, dt_s, current_z)
        commanded_z = self._apply_limits(slew_target)

        # Anti-windup: commit integral only when not saturated by output limits.
        saturated = commanded_z != raw_target
        if not saturated:
            self._integral_um = candidate_integral

        self._stage.move_z_um(commanded_z)
        self._last_commanded_z_um = commanded_z
        self._last_command_time_s = time.monotonic()

        self._state = AutofocusState.LOCKING if abs(error_um) > self._config.command_deadband_um else AutofocusState.LOCKED
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
        loop_dt = 1.0 / self._config.loop_hz
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
