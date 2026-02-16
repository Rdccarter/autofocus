from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .calibration import FocusCalibration, ZhuangFocusCalibration
from .focus_metric import Roi, astigmatic_error_signal, centroid_near_edge, roi_total_intensity
from .interfaces import CameraInterface, StageInterface

# Either calibration type works — both provide error_to_z_offset_um().
CalibrationLike = FocusCalibration | ZhuangFocusCalibration


@dataclass(slots=True)
class AutofocusConfig:
    roi: Roi
    loop_hz: float = 30.0
    # Cap effective control-step dt to avoid large corrective jumps after
    # temporary stalls (GUI pauses, GC, device hiccups).
    max_dt_s: float = 0.2
    # PI control gains, in units of um of stage command per um equivalent error.
    kp: float = 0.6
    ki: float = 0.15
    max_step_um: float = 0.25
    integral_limit_um: float = 2.0
    stage_min_um: float | None = None
    stage_max_um: float | None = None
    # Safety clamp around initial lock position to avoid runaway absolute jumps.
    max_abs_excursion_um: float | None = 5.0
    # Freeze control updates when ROI total intensity drops below threshold.
    min_roi_intensity: float | None = None
    # Exponential moving average smoothing factor for the error signal.
    # 0.0 = no filtering (raw error used), 1.0 = ignore new measurements.
    error_alpha: float = 0.0
    # Reject frames when PSF centroid is within this many pixels of the ROI
    # boundary, to avoid biased second moments from a truncated PSF.
    edge_margin_px: float = 0.0
    # Do not issue stage moves smaller than this threshold (um) to reduce
    # high-frequency dithering/oscillation near focus.
    command_deadband_um: float = 0.02


@dataclass(slots=True)
class AutofocusSample:
    timestamp_s: float
    error: float
    error_um: float
    stage_z_um: float
    commanded_z_um: float
    roi_total_intensity: float
    control_applied: bool


class AstigmaticAutofocusController:
    """Closed-loop focus controller for a single astigmatic PSF target.

    Control rationale mirrors common astigmatic focus feedback loops:
    1) Measure anisotropy-based error from ROI around a single bright locus.
    2) Convert optical error into physical Z-equivalent units via calibration.
    3) Apply PI feedback with bounded step sizes for real-time stability.
    """

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

    @property
    def loop_hz(self) -> float:
        return self._config.loop_hz

    @property
    def calibration(self) -> CalibrationLike:
        return self._calibration

    @calibration.setter
    def calibration(self, value: CalibrationLike) -> None:
        self._calibration = value

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
        if self._config.edge_margin_px < 0:
            raise ValueError("edge_margin_px must be >= 0")
        if self._config.max_abs_excursion_um is not None and self._config.max_abs_excursion_um < 0:
            raise ValueError("max_abs_excursion_um must be >= 0 when provided")
        if self._config.command_deadband_um < 0:
            raise ValueError("command_deadband_um must be >= 0")

    def _apply_limits(self, target_z_um: float) -> float:
        if self._z_lock_center_um is not None and self._config.max_abs_excursion_um is not None:
            excursion = float(self._config.max_abs_excursion_um)
            target_z_um = max(self._z_lock_center_um - excursion, min(self._z_lock_center_um + excursion, target_z_um))
        if self._config.stage_min_um is not None:
            target_z_um = max(self._config.stage_min_um, target_z_um)
        if self._config.stage_max_um is not None:
            target_z_um = min(self._config.stage_max_um, target_z_um)
        return target_z_um

    def run_step(self, dt_s: float | None = None) -> AutofocusSample:
        frame = self._camera.get_frame()
        current_z = self._stage.get_z_um()
        if self._z_lock_center_um is None:
            self._z_lock_center_um = float(current_z)

        # Guard: skip duplicate frames (same timestamp as previous).
        # This prevents acting on stale data when the camera buffer stalls,
        # which is common with Micro-Manager circular buffer acquisition.
        if self._last_frame_ts is not None and frame.timestamp_s == self._last_frame_ts:
            return AutofocusSample(
                timestamp_s=frame.timestamp_s,
                error=0.0,
                error_um=0.0,
                stage_z_um=current_z,
                commanded_z_um=current_z,
                roi_total_intensity=0.0,
                control_applied=False,
            )
        self._last_frame_ts = frame.timestamp_s

        total_intensity = roi_total_intensity(frame.image, self._config.roi)

        # Guard: freeze if ROI intensity is too low (bead lost).
        if self._config.min_roi_intensity is not None and total_intensity < self._config.min_roi_intensity:
            return AutofocusSample(
                timestamp_s=frame.timestamp_s,
                error=0.0,
                error_um=0.0,
                stage_z_um=current_z,
                commanded_z_um=current_z,
                roi_total_intensity=total_intensity,
                control_applied=False,
            )

        # Guard: freeze if PSF centroid is near the ROI boundary (truncated PSF).
        if self._config.edge_margin_px > 0 and centroid_near_edge(
            frame.image, self._config.roi, self._config.edge_margin_px
        ):
            return AutofocusSample(
                timestamp_s=frame.timestamp_s,
                error=0.0,
                error_um=0.0,
                stage_z_um=current_z,
                commanded_z_um=current_z,
                roi_total_intensity=total_intensity,
                control_applied=False,
            )

        error = astigmatic_error_signal(frame.image, self._config.roi)
        error_um = self._calibration.error_to_z_offset_um(error)
        if not math.isfinite(float(error_um)):
            raise RuntimeError("Non-finite autofocus error encountered; check ROI/calibration")

        # Optional exponential moving average on the error signal.
        alpha = self._config.error_alpha
        if 0.0 < alpha < 1.0 and self._filtered_error_um is not None:
            error_um = alpha * self._filtered_error_um + (1.0 - alpha) * error_um
        self._filtered_error_um = error_um

        if dt_s is None:
            dt_s = 1.0 / self._config.loop_hz
        dt_s = max(0.0, min(float(dt_s), self._config.max_dt_s))

        self._integral_um += error_um * dt_s
        self._integral_um = max(
            -self._config.integral_limit_um,
            min(self._config.integral_limit_um, self._integral_um),
        )

        correction = -(self._config.kp * error_um + self._config.ki * self._integral_um)
        correction = max(-self._config.max_step_um, min(self._config.max_step_um, correction))

        if abs(correction) <= self._config.command_deadband_um:
            # Anti-chatter: ignore tiny corrections that mostly reflect metric
            # noise near lock and undo this step's integral accumulation.
            self._integral_um -= error_um * dt_s
            self._integral_um = max(
                -self._config.integral_limit_um,
                min(self._config.integral_limit_um, self._integral_um),
            )
            return AutofocusSample(
                timestamp_s=frame.timestamp_s,
                error=error,
                error_um=error_um,
                stage_z_um=current_z,
                commanded_z_um=current_z,
                roi_total_intensity=total_intensity,
                control_applied=False,
            )

        raw_target = current_z + correction
        commanded_z = self._apply_limits(raw_target)

        # Anti-windup: if the stage command was clamped by limits, undo the
        # integrator accumulation for this step so it does not wind up while
        # the stage is saturated.
        if commanded_z != raw_target:
            self._integral_um -= error_um * dt_s
            self._integral_um = max(
                -self._config.integral_limit_um,
                min(self._config.integral_limit_um, self._integral_um),
            )

        self._stage.move_z_um(commanded_z)

        return AutofocusSample(
            timestamp_s=frame.timestamp_s,
            error=error,
            error_um=error_um,
            stage_z_um=current_z,
            commanded_z_um=commanded_z,
            roi_total_intensity=total_intensity,
            control_applied=True,
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