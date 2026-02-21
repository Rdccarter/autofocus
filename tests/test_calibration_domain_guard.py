from __future__ import annotations


from dataclasses import dataclass

from auto_focus.autofocus import AstigmaticAutofocusController, AutofocusConfig, AutofocusState, AutofocusWorker
from auto_focus.calibration import FocusCalibration
from auto_focus.focus_metric import Roi
from auto_focus.hardware import HamamatsuOrcaCamera
from auto_focus.interfaces import CameraFrame
from auto_focus.calibration import ZhuangFocusCalibration
from auto_focus.zhuang import ZhuangParams, ZhuangEllipticityParams, ZhuangLookupTable


class DummyCamera:
    def __init__(self) -> None:
        self.frame = CameraFrame(image=[[1.0]], timestamp_s=1.0)

    def get_frame(self) -> CameraFrame:
        return self.frame


class DummyStage:
    def __init__(self) -> None:
        self.z = 0.0
        self.moves: list[float] = []

    def get_z_um(self) -> float:
        return self.z

    def move_z_um(self, target_z_um: float) -> None:
        self.moves.append(target_z_um)
        self.z = target_z_um


@dataclass(slots=True)
class DummyLookup:
    error_values: list[float]


@dataclass(slots=True)
class DummyZhuangCalibration:
    lookup: DummyLookup

    def error_to_z_offset_um(self, error: float) -> float:
        return error


def _config() -> AutofocusConfig:
    return AutofocusConfig(roi=Roi(x=0, y=0, width=1, height=1), kp=1.0, ki=0.0, kd=0.0, max_step_um=1.0)


def test_out_of_range_linear_calibration_skips_control(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.8)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.2, error_max=0.2)
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state == AutofocusState.DEGRADED
    assert stage.moves == []


def test_out_of_range_lookup_calibration_skips_control(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.9)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    # exercise lookup fallback path used by Zhuang-style calibrations
    cal = DummyZhuangCalibration(lookup=DummyLookup(error_values=[-0.3, 0.3]))
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state == AutofocusState.DEGRADED
    assert stage.moves == []


def test_lock_setpoint_initializes_roi_bias_from_calibration_focus(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.12)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    # At lock engagement, controller should treat current ROI bias as setpoint
    # offset so it does not command an immediate move.
    cal = FocusCalibration(error_at_focus=0.05, error_to_um=1.0, error_min=-0.2, error_max=0.2)
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state == AutofocusState.LOCKED
    assert stage.moves == []


def test_lock_setpoint_applies_only_delta_from_initial_roi_bias(monkeypatch):
    signal = {"value": 0.12}

    def _err(_img, _roi):
        return signal["value"]

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", _err)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = FocusCalibration(error_at_focus=0.05, error_to_um=1.0, error_min=-0.3, error_max=0.3)
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    first = controller.run_step()
    assert first.control_applied is False

    signal["value"] = 0.16  # +0.04 vs engagement baseline
    controller._camera.frame = CameraFrame(image=[[1.0]], timestamp_s=2.0)
    second = controller.run_step()

    assert second.control_applied is True
    assert stage.moves, "expected correction move"
    assert stage.moves[-1] < 0.0



def test_out_of_range_sets_diagnostic(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.95)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.2, error_max=0.2)
    controller = AstigmaticAutofocusController(DummyCamera(), DummyStage(), _config(), cal)

    sample = controller.run_step()

    assert sample.diagnostic == "error outside calibration domain"


def test_transition_lock_disabled_recenter_enabled(monkeypatch):
    signal = {"value": 0.10}

    def _err(_img, _roi):
        return signal["value"]

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", _err)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = FocusCalibration(error_at_focus=0.05, error_to_um=1.0, error_min=-0.3, error_max=0.3)
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    controller.run_step()  # initializes lock setpoint
    controller.update_config(lock_setpoint=False, recenter_alpha=0.5)
    controller.reset_lock_state()
    controller._camera.frame = CameraFrame(image=[[1.0]], timestamp_s=2.0)
    signal["value"] = 0.18

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state == AutofocusState.LOCKED


def test_roi_change_resets_setpoint_memory(monkeypatch):
    signal = {"value": 0.12}

    def _err(_img, _roi):
        return signal["value"]

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", _err)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = FocusCalibration(error_at_focus=0.05, error_to_um=1.0, error_min=-0.3, error_max=0.3)
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    controller.run_step()
    signal["value"] = 0.20
    controller.update_roi(Roi(x=0, y=0, width=1, height=1))
    controller._camera.frame = CameraFrame(image=[[1.0]], timestamp_s=2.0)

    sample = controller.run_step()

    # Fresh ROI lock should not apply an immediate move.
    assert sample.control_applied is False
    assert sample.state == AutofocusState.LOCKED


def test_lookup_domain_guard_handles_unsorted_error_values(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.6)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = DummyZhuangCalibration(lookup=DummyLookup(error_values=[0.3, -0.3]))
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state == AutofocusState.DEGRADED


def test_worker_uses_elapsed_dt(monkeypatch):
    class StubController:
        loop_hz = 10.0

        def __init__(self):
            self.dt_values: list[float] = []

        def run_step(self, dt_s=None):
            self.dt_values.append(float(dt_s))
            if len(self.dt_values) >= 2:
                raise RuntimeError("stop")
            return None

    clock = {"t": 100.0}

    def fake_monotonic():
        val = clock["t"]
        clock["t"] += 0.15
        return val

    monkeypatch.setattr("auto_focus.autofocus.time.monotonic", fake_monotonic)
    monkeypatch.setattr("auto_focus.autofocus.time.sleep", lambda _s: None)

    controller = StubController()
    worker = AutofocusWorker(controller)
    worker._run_loop()

    assert len(controller.dt_values) == 2
    assert controller.dt_values[0] == 0.1
    assert controller.dt_values[1] == 0.45


def test_hamamatsu_orca_requires_frame_source():
    try:
        HamamatsuOrcaCamera(frame_source=None)
    except ValueError as exc:
        assert "frame_source" in str(exc)
    else:
        raise AssertionError("expected ValueError when frame_source is None")


def test_zhuang_from_axis_params_uses_sigma_ratio_for_e0():
    px = ZhuangParams(sigma0=4.0, A=0.0, B=0.0, c=0.0, d=1.0)
    py = ZhuangParams(sigma0=1.0, A=0.0, B=0.0, c=0.0, d=1.0)

    q = ZhuangEllipticityParams.from_axis_params(px, py)

    assert q.e0 == 4.0


def test_zhuang_calibration_has_range_checker():
    lookup = ZhuangLookupTable(error_values=[-0.2, 0.2], z_values_um=[-1.0, 1.0], z_range=(-1.0, 1.0))
    cal = ZhuangFocusCalibration(
        params=ZhuangEllipticityParams(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        params_x=ZhuangParams(1.0, 0.0, 0.0, 0.0, 1.0),
        params_y=ZhuangParams(1.0, 0.0, 0.0, 0.0, 1.0),
        lookup=lookup,
    )

    assert cal.is_error_in_range(0.0)
    assert not cal.is_error_in_range(0.5)
