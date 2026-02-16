from __future__ import annotations

import math
import os
import threading
from pathlib import Path

from .autofocus import (
    AstigmaticAutofocusController,
    AutofocusConfig,
    AutofocusSample,
    AutofocusWorker,
)
from .calibration import (
    FocusCalibration,
    ZhuangFocusCalibration,
    auto_calibrate,
    auto_calibrate_zhuang,
    calibration_quality_issues,
    fit_linear_calibration_with_report,
    fit_zhuang_calibration,
    save_calibration_samples_csv,
    save_zhuang_calibration_samples_csv,
)
from .focus_metric import Roi, astigmatic_error_signal
from .interfaces import CameraInterface, StageInterface


def _prepare_napari_environment() -> None:
    """Set safe defaults to avoid third-party napari plugin crashes.

    Some environments have incompatible external napari plugins (e.g. pydantic
    API mismatches) that can crash viewer startup and interaction. Disabling
    external plugin auto-discovery keeps core viewer behavior stable.
    """
    os.environ.setdefault("NAPARI_DISABLE_PLUGINS", "1")
    os.environ.setdefault("NAPARI_DISABLE_PLUGIN_ENTRY_POINTS", "1")
    os.environ.setdefault("NAPARI_DISABLE_PLUGIN_ENTRYPOINTS", "1")


def _calibration_plan_from_nm(center_z_um: float, half_range_nm: float, step_nm: float) -> tuple[float, float, int, float]:
    """Build calibration sweep bounds/steps from nanometer UI values."""

    half_range_nm = max(1.0, float(half_range_nm))
    step_nm = max(1.0, float(step_nm))
    span_nm = 2.0 * half_range_nm

    z_min = center_z_um - (half_range_nm / 1000.0)
    z_max = center_z_um + (half_range_nm / 1000.0)

    n_steps = int(math.floor(span_nm / step_nm)) + 1
    n_steps = max(2, n_steps)
    effective_step_nm = span_nm / float(n_steps - 1)
    return z_min, z_max, n_steps, effective_step_nm




def _build_runtime_calibration(base_calibration):
    """Return runtime calibration used by control loop.

    Runtime autofocus uses calibration as a relative move model (slope + sign).
    Target error is always zero for astigmatic focus lock, so we intentionally
    force error_at_focus to 0.0 to keep behavior reproducible across restarts
    and ROI retargets.

    For ZhuangFocusCalibration, the lookup table already maps error→z offset
    relative to focus, so we use it directly.
    """
    if isinstance(base_calibration, ZhuangFocusCalibration):
        # Zhuang calibration already has error_at_focus=0.0 and uses lookup table
        return base_calibration

    return FocusCalibration(
        error_at_focus=0.0,
        error_to_um=base_calibration.error_to_um,
    )


def launch_autofocus_viewer(
    camera: CameraInterface,
    stage: StageInterface,
    *,
    calibration: FocusCalibration,
    default_config: AutofocusConfig,
    interval_ms: int = 20,
    calibration_output_path: str | None = None,
    calibration_half_range_um: float = 0.75,
    calibration_steps: int = 21,
) -> None:
    """Live napari viewer with interactive ROI selection and background autofocus.

    Controls:
    - Draw ROI rectangle on the "ROI" layer to start/re-target autofocus.
    - Click the "Run Calibration Sweep" button (or press `c`) to sweep Z and save CSV.
    - Press `Escape` to stop and close.
    """

    _prepare_napari_environment()

    try:
        import napari
        import numpy as np
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QDoubleSpinBox, QLabel, QPushButton, QWidget, QVBoxLayout
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "napari is required for the live viewer. Install with: pip install napari"
        ) from exc

    current_calibration = calibration

    initial = np.asarray(camera.get_frame().image)
    viewer = napari.Viewer(title="Autofocus Ã¢â‚¬â€ draw ROI to start")
    image_layer = viewer.add_image(initial, name="camera", blending="opaque")

    roi_layer = viewer.add_shapes(
        name="ROI",
        edge_color="cyan",
        edge_width=2,
        face_color="transparent",
    )
    roi_layer.mode = "add_rectangle"

    status_text = viewer.text_overlay
    status_text.visible = True
    status_text.position = "top_left"
    status_text.font_size = 12
    status_text.text = "Draw a rectangle on the PSF to begin autofocus"

    initial_step_nm = (2.0 * float(calibration_half_range_um) * 1000.0) / max(1, int(calibration_steps) - 1)

    state: dict = {
        "worker": None,
        "last_sample": None,
        "last_roi": None,
        "calibration_message": "",
        "calibration_busy": False,
        "calibration_progress": "",
        "pending_roi": None,
        "calibration_cancel_evt": threading.Event(),
        "autofocus_enabled": True,
    }

    def _roi_from_rectangle(rect_coords) -> Roi:
        arr = np.asarray(rect_coords)
        y_min = int(np.floor(arr[:, 0].min()))
        y_max = int(np.ceil(arr[:, 0].max()))
        x_min = int(np.floor(arr[:, 1].min()))
        x_max = int(np.ceil(arr[:, 1].max()))
        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)
        return Roi(x=max(0, x_min), y=max(0, y_min), width=w, height=h)

    def _stop_worker(*, wait: bool = True) -> None:
        worker = state.get("worker")
        if worker is not None:
            worker.stop(wait=wait)
            state["worker"] = None

    def _on_sample(sample: AutofocusSample) -> None:
        state["last_sample"] = sample

    def _start_autofocus(roi: Roi) -> None:
        if not state.get("autofocus_enabled", True):
            state["last_roi"] = roi
            return
        _stop_worker(wait=False)
        config = AutofocusConfig(
            roi=roi,
            loop_hz=default_config.loop_hz,
            kp=default_config.kp,
            ki=default_config.ki,
            max_step_um=default_config.max_step_um,
            integral_limit_um=default_config.integral_limit_um,
            stage_min_um=default_config.stage_min_um,
            stage_max_um=default_config.stage_max_um,
            max_abs_excursion_um=default_config.max_abs_excursion_um,
            min_roi_intensity=default_config.min_roi_intensity,
            error_alpha=default_config.error_alpha,
            edge_margin_px=default_config.edge_margin_px,
        )
        runtime_calibration = _build_runtime_calibration(current_calibration)
        controller = AstigmaticAutofocusController(
            camera=camera,
            stage=stage,
            config=config,
            calibration=runtime_calibration,
        )
        worker = AutofocusWorker(controller=controller, on_sample=_on_sample)
        state["worker"] = worker
        state["last_roi"] = roi
        worker.start()

    def _apply_pending_roi() -> None:
        roi = state.get("pending_roi")
        if roi is None:
            return
        _start_autofocus(roi)

    def _on_roi_change(_event=None) -> None:
        try:
            shapes = roi_layer.data
            if not shapes:
                return
            rect = shapes[-1]
            roi = _roi_from_rectangle(rect)
            state["pending_roi"] = roi
            roi_apply_timer.start(200)
        except Exception as exc:
            state["calibration_message"] = f"ROI update failed: {exc}"

    def _run_calibration_sweep(roi: Roi) -> None:
        nonlocal current_calibration

        def _on_calibration_step(step_idx: int, total_steps: int, target_z: float, measured_z: float | None, ok: bool) -> None:
            if ok and measured_z is not None:
                state["calibration_progress"] = (
                    f"Calibration {step_idx}/{total_steps}: "
                    f"target={target_z:+0.3f} um, measured={measured_z:+0.3f} um"
                )
            else:
                state["calibration_progress"] = (
                    f"Calibration {step_idx}/{total_steps}: "
                    f"target={target_z:+0.3f} um (move failed, continuing)"
                )

        try:
            state["calibration_busy"] = True
            state["calibration_progress"] = ""
            _stop_worker()
            center_z = float(stage.get_z_um())
            z_min, z_max, dynamic_steps, _ = _calibration_plan_from_nm(
                center_z, range_spin_nm.value(), step_spin_nm.value()
            )

            # Collect samples with full Gaussian PSF fitting for Zhuang model
            zhuang_samples = auto_calibrate_zhuang(
                camera,
                stage,
                roi,
                z_min_um=z_min,
                z_max_um=z_max,
                n_steps=dynamic_steps,
                should_stop=state["calibration_cancel_evt"].is_set,
                on_step=_on_calibration_step,
            )
            stage.move_z_um(center_z)

            if calibration_output_path:
                out_path = Path(calibration_output_path)
            else:
                out_path = Path.cwd() / "calibration_sweep.csv"

            # Save extended CSV with Gaussian fit data
            save_zhuang_calibration_samples_csv(out_path, zhuang_samples)

            # Convert to basic samples for linear fit / quality checks
            from .calibration import CalibrationSample
            basic_samples = [
                CalibrationSample(z_um=s.z_um, error=s.error, weight=s.weight)
                for s in zhuang_samples
            ]

            # Try Zhuang model first, fall back to linear
            zhuang_ok = False
            try:
                state["calibration_progress"] = "Fitting Zhuang defocus model..."
                zhuang_report = fit_zhuang_calibration(zhuang_samples)
                if (zhuang_report.r2_x > 0.7 and zhuang_report.r2_y > 0.7
                        and zhuang_report.usable_range_um is not None):
                    current_calibration = zhuang_report.calibration
                    rng = zhuang_report.usable_range_um
                    state["calibration_message"] = (
                        f"Zhuang calibration complete: "
                        f"{zhuang_report.n_good_fits}/{zhuang_report.n_samples} good fits | "
                        f"R²(σx)={zhuang_report.r2_x:.3f}, R²(σy)={zhuang_report.r2_y:.3f} | "
                        f"range=[{rng[0]:+.3f}, {rng[1]:+.3f}] µm | "
                        f"linear_slope≈{zhuang_report.linear_slope:+.4f} um/error | "
                        f"saved to {out_path}"
                    )
                    zhuang_ok = True
            except Exception as zhuang_exc:
                state["calibration_progress"] = (
                    f"Zhuang fit failed ({zhuang_exc}), falling back to linear..."
                )

            if not zhuang_ok:
                # Fall back to linear calibration
                report = fit_linear_calibration_with_report(basic_samples, robust=True)
                issues = calibration_quality_issues(basic_samples, report)
                if issues:
                    raise RuntimeError(
                        " ; ".join(["Calibration quality check failed"] + issues)
                    )
                current_calibration = FocusCalibration(
                    error_at_focus=0.0, error_to_um=report.calibration.error_to_um
                )
                state["calibration_message"] = (
                    f"Linear calibration (Zhuang unavailable): "
                    f"{len(basic_samples)} samples saved to {out_path} | "
                    f"slope={report.calibration.error_to_um:+0.4f} um/error, "
                    f"R²={report.r2:0.4f}"
                )

        except Exception as exc:  # pragma: no cover
            state["calibration_message"] = f"Calibration failed: {exc}"
        finally:
            state["calibration_busy"] = False
            state["calibration_progress"] = ""
            # Use the latest ROI (user may have redrawn during the sweep).
            restart_roi = state.get("last_roi") or roi
            _start_autofocus(restart_roi)

    def _trigger_calibration() -> None:
        if state.get("calibration_busy"):
            state["calibration_cancel_evt"].set()
            state["calibration_message"] = "Stopping calibration sweep..."
            return
        roi = state.get("last_roi")
        if roi is None:
            state["calibration_message"] = "Calibration needs an ROI: draw a rectangle first"
            return
        state["calibration_cancel_evt"].clear()
        center_z = float(stage.get_z_um())
        z_min, z_max, dynamic_steps, effective_step_nm = _calibration_plan_from_nm(
            center_z, range_spin_nm.value(), step_spin_nm.value()
        )
        state["calibration_message"] = (
            f"Calibration sweep: {z_min:+0.3f} to {z_max:+0.3f} um "
            f"in {dynamic_steps} steps per pass ({2 * dynamic_steps} total, up+down; ~{effective_step_nm:0.1f} nm step). "
            "Click 'Stop Calibration Sweep' to cancel."
        )
        t = threading.Thread(target=_run_calibration_sweep, args=(roi,), daemon=True)
        t.start()


    def _toggle_autofocus() -> None:
        if state.get("autofocus_enabled", True):
            state["autofocus_enabled"] = False
            _stop_worker(wait=False)
            state["calibration_message"] = "Autofocus paused"
            return

        state["autofocus_enabled"] = True
        roi = state.get("last_roi")
        if roi is None and roi_layer.data:
            roi = _roi_from_rectangle(roi_layer.data[-1])
        if roi is None:
            state["calibration_message"] = "Autofocus enabled: draw an ROI rectangle to start"
            return
        state["calibration_message"] = "Autofocus running"
        _start_autofocus(roi)

    autofocus_button = QPushButton("Stop Autofocus")
    autofocus_button.setToolTip("Start/stop autofocus control without closing the viewer")
    autofocus_button.clicked.connect(_toggle_autofocus)

    range_label = QLabel("Calibration half-range (nm)")
    range_spin_nm = QDoubleSpinBox()
    range_spin_nm.setDecimals(1)
    range_spin_nm.setRange(1.0, 100000.0)
    range_spin_nm.setSingleStep(25.0)
    range_spin_nm.setValue(float(calibration_half_range_um) * 1000.0)

    step_label = QLabel("Calibration step size (nm)")
    step_spin_nm = QDoubleSpinBox()
    step_spin_nm.setDecimals(1)
    step_spin_nm.setRange(1.0, 100000.0)
    step_spin_nm.setSingleStep(10.0)
    step_spin_nm.setValue(max(1.0, initial_step_nm))

    calibrate_button = QPushButton("Run Calibration Sweep")
    calibrate_button.setToolTip(
        "Pause autofocus, sweep Z around the current position, and export calibration CSV"
    )
    calibrate_button.clicked.connect(_trigger_calibration)

    control_widget = QWidget()
    control_layout = QVBoxLayout(control_widget)
    control_layout.setContentsMargins(8, 8, 8, 8)
    control_layout.addWidget(autofocus_button)
    control_layout.addWidget(range_label)
    control_layout.addWidget(range_spin_nm)
    control_layout.addWidget(step_label)
    control_layout.addWidget(step_spin_nm)
    control_layout.addWidget(calibrate_button)
    viewer.window.add_dock_widget(control_widget, area="right", name="Autofocus Controls")

    @viewer.bind_key("c")
    def _calibrate(_viewer_ref):  # noqa: ARG001
        _trigger_calibration()

    roi_apply_timer = QTimer()
    roi_apply_timer.setSingleShot(True)
    roi_apply_timer.timeout.connect(_apply_pending_roi)

    roi_layer.events.data.connect(_on_roi_change)

    timer = QTimer()

    def _refresh() -> None:
        autofocus_button.setText("Stop Autofocus" if state.get("autofocus_enabled", True) else "Start Autofocus")
        calibrate_button.setText("Stop Calibration Sweep" if state.get("calibration_busy") else "Run Calibration Sweep")

        try:
            frame = camera.get_frame()
            image_layer.data = np.asarray(frame.image)
        except Exception as exc:
            status_text.text = f"Live frame error: {exc}"
            return

        current_z = None
        try:
            current_z = float(stage.get_z_um())
        except Exception:
            pass

        sample = state.get("last_sample")
        if sample is not None:
            ctrl = "ON " if sample.control_applied else "OFF"
            status_text.text = (
                f"AF {ctrl} | "
                f"err={sample.error:+.4f}  err_um={sample.error_um:+.3f}  "
                f"z(now)={(sample.stage_z_um if current_z is None else current_z):+.3f} â†’ cmd={sample.commanded_z_um:+.3f} um  "
                f"I={sample.roi_total_intensity:.0f}"
            )
            worker = state.get("worker")
            if worker is not None and worker.last_error is not None:
                status_text.text += f"  Ã¢Å¡Â  {worker.last_error}"
        elif not roi_layer.data:
            status_text.text = "Draw a rectangle on the PSF to begin autofocus"
            if current_z is not None:
                status_text.text += f" | z={current_z:+.3f} um"
        else:
            status_text.text = "Starting autofocus..."
            if current_z is not None:
                status_text.text += f" | z={current_z:+.3f} um"

        if state.get("calibration_message"):
            status_text.text += f"\n{state['calibration_message']}"
        if state.get("calibration_progress"):
            status_text.text += f"\n{state['calibration_progress']}"

    timer.timeout.connect(_refresh)
    timer.start(max(1, int(interval_ms)))

    @viewer.bind_key("Escape")
    def _quit(viewer_ref):  # noqa: ARG001
        _stop_worker()
        viewer_ref.close()

    viewer.window._orca_focus_timer = timer  # type: ignore[attr-defined]
    viewer.window._orca_focus_roi_timer = roi_apply_timer  # type: ignore[attr-defined]
    napari.run()
    _stop_worker()


def launch_napari_viewer(camera: CameraInterface, interval_ms: int = 20) -> None:
    """Display a live interactive camera stream using napari."""

    _prepare_napari_environment()

    try:
        import napari
        from qtpy.QtCore import QTimer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "napari is required for the live viewer. Install with: pip install napari"
        ) from exc

    initial = camera.get_frame().image
    viewer = napari.Viewer(title="Live ORCA stream")
    layer = viewer.add_image(initial, name="camera", blending="opaque")

    timer = QTimer()

    def update() -> None:
        frame = camera.get_frame()
        layer.data = frame.image

    timer.timeout.connect(update)
    timer.start(max(1, int(interval_ms)))

    viewer.window._orca_focus_timer = timer  # type: ignore[attr-defined]
    napari.run()


def launch_matplotlib_viewer(camera: CameraInterface) -> None:
    """Legacy matplotlib viewer fallback."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for interactive display. "
            "Install with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots()
    initial = camera.get_frame().image
    im = ax.imshow(initial, cmap="gray")
    ax.set_title("Live ORCA stream")

    def update(_: int):
        frame = camera.get_frame()
        im.set_data(frame.image)
        return (im,)

    fig._orca_focus_anim = FuncAnimation(fig, update, interval=40, blit=True)  # type: ignore[attr-defined]
    plt.show()


def launch_live_viewer(camera: CameraInterface) -> None:
    """Launch napari viewer, falling back to matplotlib if needed."""

    try:
        launch_napari_viewer(camera)
    except RuntimeError:
        launch_matplotlib_viewer(camera)
