from __future__ import annotations

from pyqtgraph.Qt import QtCore

Signal = getattr(QtCore, "Signal", QtCore.pyqtSignal)


class AutofocusSignals(QtCore.QObject):
    frame_ready = Signal(object)
    autofocus_update = Signal(object)
    state_changed = Signal(str)
    fault = Signal(str)
    roi_changed = Signal(tuple)
