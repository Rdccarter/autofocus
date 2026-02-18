# Realtime Focusfeedback for Tracking Loci in Live Cells

A real-time astigmatic autofocus toolkit for microscopy.

This project provides:
- a closed-loop Z autofocus controller,
- calibration tooling (linear and Zhuang-based),
- a live PyQt/pyqtgraph GUI for tuning and diagnostics,
- camera/stage backend adapters (simulation + hardware backends).

---

## What problem this package solves

In astigmatic autofocus, bead shape changes with defocus. The software measures an error signal from a small ROI around a target and converts that into Z corrections.

This package is designed to support:
- **single calibration reuse** across multiple ROIs,
- **safe loop behavior** when image quality or calibration-domain assumptions are violated,
- **interactive operation** for alignment and calibration.

---

## Features

### Core autofocus loop
- PID-like loop with configurable `kp`, `ki`, `kd`, loop rate, deadband, max step, and slew limiting.
- Optional safety clamps:
  - absolute stage range (`stage_min_um`, `stage_max_um`),
  - max excursion from lock center.
- Confidence gating (intensity, variance, saturation, edge proximity).
- Calibration-domain guard to skip control when measured error is outside trusted calibration range.

### Calibration
- Linear and robust linear fitting.
- Physics-informed Zhuang model support when Gaussian PSF fit data is available.
- CSV load/save for sweep reuse.
- Quality checks with human-readable issues.

### Live GUI (`show-live`)
- Live image + draggable ROI.
- Start/Stop autofocus.
- In-GUI calibration sweep and save.
- Control tuning widgets (gains, deadband, speed).
- Rotation/flip display transforms.
- **Software ROI XY tracking** (no XY stage required):
  - Track ROI in XY (enable/disable)
  - Track gain
  - Track deadband (px)
  - Track max step (px/frame)
- **Display controls**:
  - Histogram/LUT panel
  - Autoscale button
  - Min/max level sliders
  - Gamma control via LUT

---

## Installation

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2) Install package

```bash
pip install -e .
```

### 3) Install extras as needed

```bash
# GUI + numpy stack
pip install -e .[ui]

# test/dev tools
pip install -e .[dev]

# optional camera backend
pip install -e .[camera]
```

> Note: many workflows require `numpy` and GUI dependencies from the `ui` extra.

---

## Quick start

### Simulated camera/stage with live GUI

```bash
python -m auto_focus.cli --camera simulate --show-live --allow-missing-calibration
```

This launches the GUI using a temporary default calibration. Use **Calibrate** in the GUI and then restart with a saved calibration CSV.

### Simulated run without GUI

```bash
python -m auto_focus.cli --camera simulate --duration 5 --loop-hz 30
```

### Reuse an existing calibration file

```bash
python -m auto_focus.cli \
  --camera simulate \
  --calibration-csv calibration_sweep.csv \
  --calibration-model zhuang
```

---

## CLI reference (most used options)

- Runtime/loop:
  - `--duration`
  - `--loop-hz`
  - `--max-dt-s`
- Live GUI:
  - `--show-live`
- Backends:
  - `--camera {simulate,orca,andor,micromanager}`
  - `--stage {simulate,mcl,micromanager}` (optional, auto-chosen if omitted)
- Control:
  - `--kp`, `--ki`
  - `--max-step`
  - `--command-deadband-um`
  - `--stage-min-um`, `--stage-max-um`
  - `--af-max-excursion-um`
- Calibration:
  - `--calibration-csv`
  - `--allow-missing-calibration`
  - `--calibration-model {zhuang,linear,poly2,piecewise}`
  - `--calibration-half-range-um`
  - `--calibration-steps`
  - `--calibration-expected-slope {auto,positive,negative}`

---

## Recommended operational workflow

1. Start in live mode with simulation/hardware.
2. Draw ROI tightly around one stable fiducial.
3. Run calibration sweep in GUI.
4. Confirm quality (no warnings/faults).
5. Start autofocus.
6. If target drifts laterally, enable **Track ROI in XY** and tune:
   - begin with gain ~0.3–0.5,
   - deadband ~1–2 px,
   - max step ~5–10 px/frame.
7. Reuse the same calibration CSV on future runs.

---

## Understanding “Lock setpoint”

- **Checked**: lock and maintain per-ROI setpoint behavior so loop corrections are relative to the engaged setpoint.
- **Unchecked**: do not enforce that lock behavior (useful for specific tuning/recenter cases).

When toggled, lock state is reset so stateful loop memory is re-acquired cleanly.

---

## Troubleshooting

### `Calibration CSV not found`
Provide a valid path with `--calibration-csv`, or run live mode with `--allow-missing-calibration` to create one.

### `Calibration CSV failed quality checks`
Re-run calibration with:
- better ROI SNR,
- sweep centered around focus,
- reduced sweep range if curve is strongly non-linear.

### Autofocus goes degraded / stops correcting
Common causes:
- low ROI confidence (dim/saturated/truncated target),
- measured error outside calibration domain,
- duplicate/invalid frames.

Use the GUI status text and diagnostics plot to identify which condition occurred.

### Pytest import fails with missing `numpy`
Install UI dependencies (`pip install -e .[ui]`) or `numpy` directly in your environment.

---

## Project layout

- `src/auto_focus/autofocus.py` — control loop and state machine
- `src/auto_focus/calibration.py` — calibration models and fitting
- `src/auto_focus/pg_gui.py` — pyqtgraph GUI
- `src/auto_focus/cli.py` — command-line entrypoint
- `src/auto_focus/hardware.py` — hardware wrappers
- `tests/` — regression tests

---

## License

MIT (see project metadata in `pyproject.toml`).
