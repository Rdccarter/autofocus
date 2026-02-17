from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

Slot = getattr(QtCore, "Slot", QtCore.pyqtSlot)

from .autofocus import AstigmaticAutofocusController, AutofocusConfig, AutofocusState
from .calibration import (
    CalibrationMetadata,
    FocusCalibration,
    auto_calibrate,
    calibration_quality_issues,
    fit_linear_calibration_with_report,
    save_calibration_metadata_json,
    save_calibration_samples_csv,
)
from .focus_metric import Roi
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


class CameraWorker(threading.Thread):
    def __init__(self, camera: CameraInterface, frame_queue: LatestFrameQueue, signals: AutofocusSignals, stop_evt: threading.Event):
        super().__init__(daemon=True)
        self._camera = camera
        self._queue = frame_queue
        self._signals = signals
        self._stop_evt = stop_evt
        self._pause_evt = threading.Event()

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
                self._queue.put(frame)
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
        calibration: FocusCalibration,
        default_config: AutofocusConfig,
        calibration_output_path: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Autofocus Instrument Panel")
        self._camera = camera
        self._stage = stage
        self._calibration = calibration
        self._config = default_config
        self._calibration_output_path = calibration_output_path or "calibration_sweep.csv"

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

        self._camera_worker = CameraWorker(self._camera, self._frame_queue, self._signals, self._stop_evt)

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

        self._z_lbl.setText(f"Z: {sample.commanded_z_um:+.3f} µm")
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
        def _task() -> None:
            try:
                QtCore.QMetaObject.invokeMethod(self._af_worker, "stop_loop", QtCore.Qt.QueuedConnection)
                self._camera_worker.pause()
                roi = self._controller.get_config_snapshot().roi
                center = float(self._stage.get_z_um())
                samples = auto_calibrate(
                    self._camera,
                    self._stage,
                    roi,
                    z_min_um=center - 0.75,
                    z_max_um=center + 0.75,
                    n_steps=21,
                )
                report = fit_linear_calibration_with_report(samples, robust=True)
                issues = calibration_quality_issues(samples, report)
                if issues:
                    self._signals.fault.emit("Calibration failed: " + "; ".join(issues))
                    return
                self._calibration = FocusCalibration(error_at_focus=0.0, error_to_um=report.calibration.error_to_um)
                self._controller.calibration = self._calibration
                out = Path(self._calibration_output_path)
                save_calibration_samples_csv(out, samples)
                meta = CalibrationMetadata(
                    roi_size=(roi.width, roi.height),
                    stage_type=type(self._stage).__name__,
                    created_at_unix_s=time.time(),
                )
                save_calibration_metadata_json(out.with_suffix('.meta.json'), meta)
                self._signals.status.emit(f"Calibration saved: {out}")
            except Exception as exc:
                self._signals.fault.emit(f"Calibration failure: {exc}")
            finally:
                self._camera_worker.resume()

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
    calibration: FocusCalibration,
    default_config: AutofocusConfig,
    calibration_output_path: str | None = None,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = AutofocusMainWindow(
        camera=camera,
        stage=stage,
        calibration=calibration,
        default_config=default_config,
        calibration_output_path=calibration_output_path,
    )
    win.resize(1280, 860)
    win.start()
    win.show()
    app.exec()
