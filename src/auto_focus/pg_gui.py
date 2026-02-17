from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

Slot = getattr(QtCore, "Slot", QtCore.pyqtSlot)

from .autofocus import AstigmaticAutofocusController, AutofocusConfig, AutofocusState, CalibrationLike
from .calibration import (
    CalibrationMetadata,
    CalibrationSample,
    ZhuangCalibrationSample,
    calibration_quality_issues,
    fit_linear_calibration_with_report,
    fit_zhuang_calibration,
    save_calibration_metadata_json,
    save_zhuang_calibration_samples_csv,
)
from .focus_metric import Roi, astigmatic_error_signal, extract_roi, fit_gaussian_psf, roi_total_intensity
from .interfaces import CameraFrame, CameraInterface, StageInterface
from .ui_signals import AutofocusSignals


class LatestFrameQueue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._seq = 0

    def put(self, frame: CameraFrame) -> None:
        normalized = _normalize_frame(frame)
        with self._lock:
            self._latest = normalized
            self._seq += 1

    def get_latest(self) -> tuple[CameraFrame | None, int]:
        with self._lock:
            return self._latest, self._seq


@dataclass(slots=True)
class RunStats:
    loop_latency_ms: deque[float]
    dropped_frames: int = 0
    total_frames: int = 0
    faults: list[str] | None = None




@dataclass(slots=True)
class FrameTransformState:
    rotation_deg: int = 0
    flip_h: bool = False
    flip_v: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def set(self, *, rotation_deg: int | None = None, flip_h: bool | None = None, flip_v: bool | None = None) -> None:
        with self._lock:
            if rotation_deg is not None:
                self.rotation_deg = int(rotation_deg) % 360
            if flip_h is not None:
                self.flip_h = bool(flip_h)
            if flip_v is not None:
                self.flip_v = bool(flip_v)

    def get(self) -> tuple[int, bool, bool]:
        with self._lock:
            return self.rotation_deg, self.flip_h, self.flip_v


def _apply_frame_transform(frame: CameraFrame, transform: FrameTransformState) -> CameraFrame:
    rotation_deg, flip_h, flip_v = transform.get()
    if rotation_deg % 360 == 0 and (not flip_h) and (not flip_v):
        return frame

    image = frame.image
    try:
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim == 3 and 1 in arr.shape:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Camera frame must be 2D (received shape={arr.shape!r})")

        k = (rotation_deg // 90) % 4
        if k:
            arr = np.rot90(arr, k=k)
        if flip_h:
            arr = np.fliplr(arr)
        if flip_v:
            arr = np.flipud(arr)
        return CameraFrame(image=arr.copy(), timestamp_s=frame.timestamp_s)
    except ImportError:
        if hasattr(image, "tolist") and callable(image.tolist):
            image = image.tolist()
        if isinstance(image, tuple):
            image = list(image)
        if not isinstance(image, list) or not image or not isinstance(image[0], (list, tuple)):
            raise ValueError("Camera frame must be 2D")
        rows = [list(r) for r in image]

        k = (rotation_deg // 90) % 4
        for _ in range(k):
            rows = [list(col) for col in zip(*rows[::-1])]
        if flip_h:
            rows = [row[::-1] for row in rows]
        if flip_v:
            rows = rows[::-1]
        return CameraFrame(image=rows, timestamp_s=frame.timestamp_s)


class CameraWorker(threading.Thread):
    def __init__(self, camera: CameraInterface, frame_queue: LatestFrameQueue, signals: AutofocusSignals, stop_evt: threading.Event, transform: FrameTransformState):
        super().__init__(daemon=True)
        self._camera = camera
        self._queue = frame_queue
        self._signals = signals
        self._stop_evt = stop_evt
        self._pause_evt = threading.Event()
        self._transform = transform

    def pause(self) -> None:
        self._pause_evt.set()

    def resume(self) -> None:
        self._pause_evt.clear()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            if self._pause_evt.is_set():
                time.sleep(0.01)
                continue
            try:
                frame = self._camera.get_frame()
                oriented = _apply_frame_transform(_normalize_frame(frame), self._transform)
                self._queue.put(oriented)
                latest, _ = self._queue.get_latest()
                if latest is not None:
                    self._signals.frame_ready.emit(latest)
            except Exception as exc:  # pragma: no cover
                self._signals.fault.emit(f"Camera failure: {exc}")
                time.sleep(0.05)


def _normalize_frame(frame: CameraFrame) -> CameraFrame:
    """Ensure camera frames are 2D and detached before cross-thread use."""
    try:
        import numpy as np

        arr = np.asarray(frame.image)
        if arr.ndim == 3 and 1 in arr.shape:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Camera frame must be 2D (received shape={arr.shape!r})")
        # Copy to detach from camera-owned buffers that may be reused.
        return CameraFrame(image=arr.copy(), timestamp_s=frame.timestamp_s)
    except ImportError:
        image = frame.image
        if hasattr(image, "tolist") and callable(image.tolist):
            image = image.tolist()
        if isinstance(image, tuple):
            image = list(image)
        if not isinstance(image, list) or not image or not isinstance(image[0], (list, tuple)):
            raise ValueError("Camera frame must be 2D")
        return CameraFrame(image=[list(row) for row in image], timestamp_s=frame.timestamp_s)


class AutofocusWorkerObject(QtCore.QObject):
    def __init__(
        self,
        controller: AstigmaticAutofocusController,
        frame_queue: LatestFrameQueue,
        signals: AutofocusSignals,
        stats: RunStats,
        stop_evt: threading.Event,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._frame_queue = frame_queue
        self._signals = signals
        self._stats = stats
        self._stop_evt = stop_evt
        self._running = False
        self._last_seq = -1
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._step)

    @Slot(tuple)
    def update_roi(self, roi_bounds: tuple[int, int, int, int]) -> None:
        self._controller.update_roi(Roi(*roi_bounds))

    @Slot(dict)
    def update_config(self, values: dict) -> None:
        self._controller.update_config(**values)

    @Slot()
    def start_loop(self) -> None:
        self._running = True
        self._timer.start(max(1, int(1000.0 / self._controller.loop_hz)))

    @Slot()
    def stop_loop(self) -> None:
        self._running = False
        self._timer.stop()

    def _step(self) -> None:
        if not self._running or self._stop_evt.is_set():
            return
        frame, seq = self._frame_queue.get_latest()
        if frame is None:
            return
        self._stats.total_frames += 1
        if self._last_seq >= 0 and seq - self._last_seq > 1:
            self._stats.dropped_frames += (seq - self._last_seq - 1)
        self._last_seq = seq
        try:
            sample = self._controller.run_step(frame=frame)
            self._stats.loop_latency_ms.append(sample.loop_latency_ms)
            self._signals.autofocus_update.emit(sample)
            self._signals.state_changed.emit(sample.state.value)
        except Exception as exc:  # pragma: no cover
            msg = f"Autofocus failure: {exc}"
            if self._stats.faults is not None:
                self._stats.faults.append(msg)
            self._signals.fault.emit(msg)


class AutofocusMainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        camera: CameraInterface,
        stage: StageInterface,
        calibration: CalibrationLike,
        default_config: AutofocusConfig,
        calibration_output_path: str | None = None,
        calibration_half_range_um: float = 0.75,
        calibration_steps: int = 21,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Autofocus Instrument Panel")
        self._camera = camera
        self._stage = stage
        self._calibration = calibration
        self._config = default_config
        self._calibration_output_path = calibration_output_path or "calibration_sweep.csv"
        self._calibration_half_range_um = max(0.05, float(calibration_half_range_um))
        self._calibration_steps = max(5, int(calibration_steps))

        self._signals = AutofocusSignals()
        self._stop_evt = threading.Event()
        self._frame_queue = LatestFrameQueue()
        self._stats = RunStats(loop_latency_ms=deque(maxlen=1000), faults=[])
        self._history_t = deque(maxlen=400)
        self._history_z = deque(maxlen=400)
        self._history_err = deque(maxlen=400)
        self._history_corr = deque(maxlen=400)
        self._last_cmd = None
        self._image_levels: tuple[float, float] | None = None
        self._frame_transform = FrameTransformState()

        self._controller = AstigmaticAutofocusController(
            camera=self._camera,
            stage=self._stage,
            config=self._config,
            calibration=self._calibration,
        )

        self._af_thread = QtCore.QThread(self)
        self._af_worker = AutofocusWorkerObject(self._controller, self._frame_queue, self._signals, self._stats, self._stop_evt)
        self._af_worker.moveToThread(self._af_thread)
        self._signals.roi_changed.connect(self._af_worker.update_roi, QtCore.Qt.QueuedConnection)
        self._signals.config_changed.connect(self._af_worker.update_config, QtCore.Qt.QueuedConnection)
        self._af_thread.start()

        self._camera_worker = CameraWorker(self._camera, self._frame_queue, self._signals, self._stop_evt, self._frame_transform)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        self._graphics = pg.GraphicsLayoutWidget()
        self._view = self._graphics.addViewBox(lockAspect=True)
        self._img = pg.ImageItem()
        self._view.addItem(self._img)
        self._roi = pg.RectROI([self._config.roi.x, self._config.roi.y], [self._config.roi.width, self._config.roi.height], pen=pg.mkPen('c', width=2))
        self._view.addItem(self._roi)
        self._state_badge = QtWidgets.QLabel("CALIBRATED_READY")
        self._state_badge.setStyleSheet("background:#444;color:white;padding:4px;font-weight:bold;")

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.addWidget(self._state_badge)
        left_panel.addWidget(self._graphics)
        top.addLayout(left_panel, 3)

        controls = QtWidgets.QVBoxLayout()
        self._status = QtWidgets.QLabel("Ready")
        self._z_lbl = QtWidgets.QLabel("Z: --")
        self._corr_lbl = QtWidgets.QLabel("Last correction: --")
        self._lat_lbl = QtWidgets.QLabel("Loop latency: --")
        self._conf_lbl = QtWidgets.QLabel("Confidence: --")
        self._drop_lbl = QtWidgets.QLabel("Frames dropped: 0")
        for w in [self._status, self._z_lbl, self._corr_lbl, self._lat_lbl, self._conf_lbl, self._drop_lbl]:
            controls.addWidget(w)

        self._start_btn = QtWidgets.QPushButton("Start Autofocus")
        self._stop_btn = QtWidgets.QPushButton("Stop Autofocus")
        self._cal_btn = QtWidgets.QPushButton("Calibrate")
        self._lock_setpoint = QtWidgets.QCheckBox("Lock setpoint")
        self._lock_setpoint.setChecked(True)
        for w in [self._start_btn, self._stop_btn, self._cal_btn, self._lock_setpoint]:
            controls.addWidget(w)

        self._kp = QtWidgets.QDoubleSpinBox(); self._kp.setValue(self._config.kp); self._kp.setPrefix("Kp ")
        self._ki = QtWidgets.QDoubleSpinBox(); self._ki.setValue(self._config.ki); self._ki.setPrefix("Ki ")
        self._kd = QtWidgets.QDoubleSpinBox(); self._kd.setValue(self._config.kd); self._kd.setPrefix("Kd ")
        self._max_speed = QtWidgets.QDoubleSpinBox(); self._max_speed.setValue(self._config.max_slew_rate_um_per_s or 0.0); self._max_speed.setPrefix("Max speed ")
        self._max_step = QtWidgets.QDoubleSpinBox(); self._max_step.setValue(self._config.max_step_um); self._max_step.setPrefix("Max step ")
        self._deadband = QtWidgets.QDoubleSpinBox(); self._deadband.setValue(self._config.command_deadband_um); self._deadband.setPrefix("Deadband ")
        for w in [self._kp, self._ki, self._kd, self._max_speed, self._max_step, self._deadband]:
            controls.addWidget(w)

        self._rotation = QtWidgets.QComboBox()
        self._rotation.addItems(["Rotate 0Â°", "Rotate 90Â°", "Rotate 180Â°", "Rotate 270Â°"])
        self._flip_h = QtWidgets.QCheckBox("Flip horizontal")
        self._flip_v = QtWidgets.QCheckBox("Flip vertical")
        controls.addWidget(self._rotation)
        controls.addWidget(self._flip_h)
        controls.addWidget(self._flip_v)

        top.addLayout(controls, 1)

        self._plot = pg.PlotWidget(title="Diagnostics (10s)")
        self._plot.addLegend()
        self._z_curve = self._plot.plot(pen='y', name='Z')
        self._e_curve = self._plot.plot(pen='c', name='Error')
        self._c_curve = self._plot.plot(pen='m', name='Correction')
        layout.addWidget(self._plot, 1)

    def _connect_signals(self) -> None:
        self._signals.frame_ready.connect(self._on_frame)
        self._signals.autofocus_update.connect(self._on_update)
        self._signals.state_changed.connect(self._on_state)
        self._signals.fault.connect(self._on_fault)
        self._signals.status.connect(self._on_status)

        self._roi.sigRegionChangeFinished.connect(self._emit_roi_change)
        self._start_btn.clicked.connect(lambda: QtCore.QMetaObject.invokeMethod(self._af_worker, "start_loop", QtCore.Qt.QueuedConnection))
        self._stop_btn.clicked.connect(lambda: QtCore.QMetaObject.invokeMethod(self._af_worker, "stop_loop", QtCore.Qt.QueuedConnection))
        self._cal_btn.clicked.connect(self._run_calibration)
        self._lock_setpoint.toggled.connect(self._on_lock_setpoint)
        self._kp.valueChanged.connect(lambda v: self._queue_config_update(kp=float(v)))
        self._ki.valueChanged.connect(lambda v: self._queue_config_update(ki=float(v)))
        self._kd.valueChanged.connect(lambda v: self._queue_config_update(kd=float(v)))
        self._max_step.valueChanged.connect(lambda v: self._queue_config_update(max_step_um=float(v)))
        self._deadband.valueChanged.connect(lambda v: self._queue_config_update(command_deadband_um=float(v)))
        self._max_speed.valueChanged.connect(
            lambda v: self._queue_config_update(max_slew_rate_um_per_s=(None if v <= 0 else float(v)))
        )
        self._rotation.currentIndexChanged.connect(self._on_transform_changed)
        self._flip_h.toggled.connect(self._on_transform_changed)
        self._flip_v.toggled.connect(self._on_transform_changed)

    def _on_transform_changed(self, *_args) -> None:
        rotation = int(self._rotation.currentIndex()) * 90
        self._frame_transform.set(
            rotation_deg=rotation,
            flip_h=self._flip_h.isChecked(),
            flip_v=self._flip_v.isChecked(),
        )
        self._image_levels = None

    def _queue_config_update(self, **values) -> None:
        self._signals.config_changed.emit(values)

    def _emit_roi_change(self) -> None:
        pos = self._roi.pos()
        size = self._roi.size()
        roi = (max(0, int(pos.x())), max(0, int(pos.y())), max(1, int(size.x())), max(1, int(size.y())))
        self._signals.roi_changed.emit(roi)

    @Slot(object)
    def _on_frame(self, frame: CameraFrame) -> None:
        try:
            import numpy as np
            arr = np.asarray(frame.image)
            if arr.ndim == 3 and 1 in arr.shape:
                arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Camera frame must be 2D (received shape={arr.shape!r})")

            if arr.dtype.kind in ("f", "c"):
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    lo = float(np.min(finite))
                    hi = float(np.max(finite))
                    if lo == hi:
                        hi = lo + 1.0
                    self._image_levels = (lo, hi)
                elif self._image_levels is None:
                    self._image_levels = (0.0, 1.0)
                self._img.setImage(arr, autoLevels=False, levels=self._image_levels)
            else:
                self._img.setImage(arr, autoLevels=False)
        except Exception as exc:
            self._signals.fault.emit(f"Display failure: {exc}")

    @Slot(object)
    def _on_update(self, sample) -> None:
        now = time.monotonic()
        self._history_t.append(now)
        self._history_z.append(sample.commanded_z_um)
        self._history_err.append(sample.error_um)
        corr = 0.0 if self._last_cmd is None else sample.commanded_z_um - self._last_cmd
        self._history_corr.append(corr)
        self._last_cmd = sample.commanded_z_um

        self._z_lbl.setText(f"Z: {sample.commanded_z_um:+.3f} Âµm")
        self._corr_lbl.setText(f"Last correction: {corr*1000.0:+.1f} nm")
        self._lat_lbl.setText(f"Loop latency: {sample.loop_latency_ms:.1f} ms")
        self._conf_lbl.setText(f"Confidence: {'good' if sample.confidence_ok else 'low'}")
        self._drop_lbl.setText(f"Frames dropped: {self._stats.dropped_frames}")
        self._status.setText(f"I={sample.roi_total_intensity:.0f} err={sample.error:+.4f}")
        if sample.confidence_ok:
            self._roi.setPen(pg.mkPen('c', width=2))
        else:
            self._roi.setPen(pg.mkPen('r', width=2))
        self._update_plot()

    def _update_plot(self) -> None:
        if not self._history_t:
            return
        t0 = self._history_t[-1]
        xs = [t - t0 for t in self._history_t]
        self._z_curve.setData(xs, list(self._history_z))
        self._e_curve.setData(xs, list(self._history_err))
        self._c_curve.setData(xs, list(self._history_corr))

    @Slot(str)
    def _on_state(self, state: str) -> None:
        colors = {
            AutofocusState.LOCKED.value: '#1b5e20',
            AutofocusState.DEGRADED.value: '#f9a825',
            AutofocusState.FAULT.value: '#b71c1c',
        }
        self._state_badge.setText(state)
        self._state_badge.setStyleSheet(f"background:{colors.get(state, '#444')};color:white;padding:4px;font-weight:bold;")

    @Slot(str)
    def _on_fault(self, message: str) -> None:
        self._status.setText(message)
        self._on_state(AutofocusState.FAULT.value)

    @Slot(str)
    def _on_status(self, message: str) -> None:
        self._status.setText(message)

    def _on_lock_setpoint(self, enabled: bool) -> None:
        self._queue_config_update(lock_setpoint=bool(enabled))

    def _run_calibration(self) -> None:
        # Read ROI directly from the widget on the GUI thread so the
        # calibration always uses exactly what is drawn on screen, even if
        # a queued roi_changed signal hasn't been processed by the AF worker yet.
        pos = self._roi.pos()
        size = self._roi.size()
        gui_roi = Roi(
            x=max(0, int(pos.x())),
            y=max(0, int(pos.y())),
            width=max(1, int(size.x())),
            height=max(1, int(size.y())),
        )
        # Push the same ROI to the controller so post-calibration autofocus
        # uses the identical region.
        self._signals.roi_changed.emit((gui_roi.x, gui_roi.y, gui_roi.width, gui_roi.height))

        def _task() -> None:
            try:
                QtCore.QMetaObject.invokeMethod(self._af_worker, "stop_loop", QtCore.Qt.QueuedConnection)
                roi = gui_roi
                center = float(self._stage.get_z_um())

                # Determine stage travel limits for sweep clamping.
                stage_range: tuple[float, float] | None = None
                get_range = getattr(self._stage, "get_range_um", None)
                if callable(get_range):
                    try:
                        stage_range = get_range()
                    except Exception:
                        pass
                # Fall back to config-level clamps if hardware query unavailable
                config_snap = self._controller.get_config_snapshot()
                if stage_range is None:
                    lo = config_snap.stage_min_um
                    hi = config_snap.stage_max_um
                    if lo is not None or hi is not None:
                        stage_range = (lo if lo is not None else -1e9, hi if hi is not None else 1e9)

                def _wait_for_settled_frame(last_seq: int, settle_s: float, move_time_s: float, timeout_s: float = 2.0):
                    """Wait for both settle time and a fresh frame in one loop.

                    Blocks until at least `settle_s` has elapsed since
                    `move_time_s` AND a new frame (seq > last_seq) is
                    available, whichever comes last.  This avoids the
                    stop-and-go cadence from separate sleep + frame-wait.
                    """
                    settle_deadline = move_time_s + settle_s
                    timeout_deadline = time.monotonic() + max(0.05, timeout_s)
                    while time.monotonic() < timeout_deadline:
                        now = time.monotonic()
                        settled = now >= settle_deadline
                        frame, seq = self._frame_queue.get_latest()
                        if settled and frame is not None and seq > last_seq:
                            return frame, seq
                        time.sleep(0.005)
                    raise RuntimeError("Timed out waiting for settled camera frame during calibration")

                def _moment_sigma(patch) -> tuple[float, float]:
                    """Moment-based sigma fallback for GUI calibration."""
                    try:
                        import numpy as _np
                        arr = _np.asarray(patch, dtype=float)
                        if arr.ndim != 2 or arr.size == 0:
                            return (float("nan"), float("nan"))
                        total = float(arr.sum())
                        if total <= 0:
                            return (float("nan"), float("nan"))
                        y_idx, x_idx = _np.indices(arr.shape, dtype=float)
                        cx = float((x_idx * arr).sum() / total)
                        cy = float((y_idx * arr).sum() / total)
                        var_x = float((((x_idx - cx) ** 2) * arr).sum() / total)
                        var_y = float((((y_idx - cy) ** 2) * arr).sum() / total)
                        import math as _m
                        sx = _m.sqrt(var_x) if var_x > 0 else float("nan")
                        sy = _m.sqrt(var_y) if var_y > 0 else float("nan")
                        return (sx, sy)
                    except Exception:
                        return (float("nan"), float("nan"))

                def _collect_and_check(*, half_range_um: float, n_steps: int, settle_s: float):
                    if n_steps < 2:
                        raise ValueError("n_steps must be >= 2")
                    z_min_um = center - half_range_um
                    z_max_um = center + half_range_um

                    # Clamp sweep to stage travel limits to avoid out-of-range errors.
                    if stage_range is not None:
                        old_min, old_max = z_min_um, z_max_um
                        z_min_um = max(z_min_um, stage_range[0])
                        z_max_um = min(z_max_um, stage_range[1])
                        if z_min_um != old_min or z_max_um != old_max:
                            self._signals.status.emit(
                                f"Sweep clamped to stage range [{z_min_um:.2f}, {z_max_um:.2f}] µm"
                            )
                    if z_max_um <= z_min_um:
                        raise ValueError(
                            f"Sweep range is empty after clamping to stage limits "
                            f"(center={center:.2f}, half_range={half_range_um:.2f}, "
                            f"stage_range={stage_range}). Move stage away from travel limit."
                        )

                    step = (z_max_um - z_min_um) / float(n_steps - 1)
                    forward_targets = [z_min_um + i * step for i in range(n_steps)]
                    targets = forward_targets + list(reversed(forward_targets))

                    samples_local: list[CalibrationSample] = []
                    zhuang_samples_local: list[ZhuangCalibrationSample] = []
                    failed_moves: list[tuple[float, Exception]] = []
                    _, last_seq = self._frame_queue.get_latest()

                    for i, target_z in enumerate(targets):
                        try:
                            self._stage.move_z_um(target_z)
                        except Exception as exc:
                            failed_moves.append((target_z, exc))
                            self._signals.status.emit(
                                f"Calibrating {i+1}/{len(targets)} — skipped z={target_z:.3f} (move failed)"
                            )
                            continue
                        move_finished_s = time.monotonic()
                        measured_z = target_z
                        try:
                            measured_z = float(self._stage.get_z_um())
                        except Exception:
                            pass
                        frame, last_seq = _wait_for_settled_frame(last_seq, settle_s, move_finished_s)
                        err = astigmatic_error_signal(frame.image, roi)
                        weight = roi_total_intensity(frame.image, roi)
                        samples_local.append(CalibrationSample(z_um=measured_z, error=err, weight=max(0.0, weight)))

                        patch = extract_roi(frame.image, roi)
                        gauss = fit_gaussian_psf(patch)
                        if gauss is not None and gauss.r_squared > 0.3:
                            sx = gauss.sigma_x
                            sy = gauss.sigma_y
                            ell = gauss.ellipticity
                            fit_r2 = gauss.r_squared
                        else:
                            # Moment-based fallback: still usable by Zhuang fitter
                            sx, sy = _moment_sigma(patch)
                            ell = sx / sy if sy > 0 else float("nan")
                            fit_r2 = float("nan")  # mark as moment fallback
                        zhuang_samples_local.append(
                            ZhuangCalibrationSample(
                                z_um=measured_z,
                                error=err,
                                weight=max(0.0, weight),
                                sigma_x=sx,
                                sigma_y=sy,
                                ellipticity=ell,
                                fit_r2=fit_r2,
                            )
                        )
                        self._signals.status.emit(f"Calibrating {i+1}/{len(targets)}")

                    if len(samples_local) < 2:
                        if failed_moves:
                            _, first_exc = failed_moves[0]
                            raise RuntimeError(
                                f"Calibration sweep failed: {len(samples_local)} succeeded, "
                                f"{len(failed_moves)} moves failed. First: {first_exc}"
                            )
                        raise RuntimeError("Calibration sweep could not collect enough valid points.")

                    try:
                        z_report = fit_zhuang_calibration(zhuang_samples_local)
                        return zhuang_samples_local, z_report.calibration, []
                    except Exception as exc:
                        self._signals.status.emit(f"Zhuang fit failed ({exc}); falling back to linear fit")
                        report_local = fit_linear_calibration_with_report(samples_local, robust=True)
                        issues_local = calibration_quality_issues(samples_local, report_local)
                        return zhuang_samples_local, report_local.calibration, issues_local

                samples, calibration_fit, issues = _collect_and_check(
                    half_range_um=self._calibration_half_range_um,
                    n_steps=self._calibration_steps,
                    settle_s=0.12,
                )

                mismatch_issue = "up/down sweep mismatch is high"
                if issues and any(mismatch_issue in issue for issue in issues):
                    self._signals.status.emit(
                        "Calibration retry: detected up/down mismatch, rerunning with slower settle and finer sweep"
                    )
                    retry_half_range = max(0.05, self._calibration_half_range_um * 0.8)
                    retry_steps = max(self._calibration_steps + 10, int(self._calibration_steps * 1.5))
                    samples_retry, calibration_retry, issues_retry = _collect_and_check(
                        half_range_um=retry_half_range,
                        n_steps=retry_steps,
                        settle_s=0.25,
                    )
                    if len(issues_retry) < len(issues):
                        samples, calibration_fit, issues = samples_retry, calibration_retry, issues_retry

                if issues:
                    self._signals.fault.emit("Calibration failed: " + "; ".join(issues))
                    return
                self._calibration = calibration_fit
                self._controller.calibration = self._calibration
                out = Path(self._calibration_output_path)
                save_zhuang_calibration_samples_csv(out, samples)
                meta = CalibrationMetadata(
                    roi_size=(roi.width, roi.height),
                    stage_type=type(self._stage).__name__,
                    created_at_unix_s=time.time(),
                )
                save_calibration_metadata_json(out.with_suffix('.meta.json'), meta)
                self._signals.status.emit(f"Calibration saved: {out}")
            except Exception as exc:
                self._signals.fault.emit(f"Calibration failure: {exc}")

        threading.Thread(target=_task, daemon=True).start()

    def closeEvent(self, event):  # noqa: N802
        self._stop_evt.set()
        QtCore.QMetaObject.invokeMethod(self._af_worker, "stop_loop", QtCore.Qt.QueuedConnection)
        self._af_thread.quit()
        self._af_thread.wait(1000)
        report = {
            "median_loop_latency_ms": (sorted(self._stats.loop_latency_ms)[len(self._stats.loop_latency_ms)//2] if self._stats.loop_latency_ms else None),
            "dropped_frames": self._stats.dropped_frames,
            "total_frames": self._stats.total_frames,
            "faults": self._stats.faults,
            "config": asdict(self._controller.get_config_snapshot()),
        }
        Path("autofocus_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        super().closeEvent(event)

    def start(self) -> None:
        self._camera_worker.start()


def launch_pg_autofocus_gui(
    camera: CameraInterface,
    stage: StageInterface,
    *,
    calibration: CalibrationLike,
    default_config: AutofocusConfig,
    calibration_output_path: str | None = None,
    calibration_half_range_um: float = 0.75,
    calibration_steps: int = 21,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = AutofocusMainWindow(
        camera=camera,
        stage=stage,
        calibration=calibration,
        default_config=default_config,
        calibration_output_path=calibration_output_path,
        calibration_half_range_um=calibration_half_range_um,
        calibration_steps=calibration_steps,
    )
    win.resize(1280, 860)
    win.start()
    win.show()
    app.exec()
