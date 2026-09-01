"""
The cross-section control strip and the viewer state behind it.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .viewer import PyVistaViewer

SLIDER_STEPS = 1000


class SliceController(QtCore.QObject):
    """
    Owns the Slice row and keeps it in step with the dataset the viewer holds. `build_toolbar` returns the row for the caller to place, and `refresh` matches the row to whatever was loaded last.

    The slider counts SLIDER_STEPS steps across the axis, and this maps that count onto the coordinates of the dataset. A result with fewer than three axes is already one plane, so the row goes insensitive rather than cut a plane out of nothing.
    """

    def __init__(self, viewer: PyVistaViewer, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer

        self._checkbox = QtWidgets.QCheckBox("Slice")
        self._checkbox.setChecked(False)
        self._checkbox.toggled.connect(self._handle_toggled)

        self._axis_combo = QtWidgets.QComboBox()
        self._axis_combo.addItems(["X", "Y", "Z"])
        self._axis_combo.setCurrentIndex(2)
        self._axis_combo.setEnabled(False)
        self._axis_combo.currentIndexChanged.connect(self._handle_axis_changed)

        self._position_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._position_slider.setRange(0, SLIDER_STEPS)
        self._position_slider.setValue(SLIDER_STEPS // 2)
        self._position_slider.setEnabled(False)
        self._position_slider.valueChanged.connect(self._handle_position_changed)

        self._position_label = QtWidgets.QLabel("-")
        self._position_label.setMinimumWidth(90)

    def build_toolbar(self) -> QtWidgets.QHBoxLayout:
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._checkbox)
        layout.addSpacing(16)
        layout.addWidget(QtWidgets.QLabel("Axis"))
        layout.addWidget(self._axis_combo)
        layout.addSpacing(16)
        layout.addWidget(QtWidgets.QLabel("Position"))
        layout.addWidget(self._position_slider, 1)
        layout.addWidget(self._position_label)
        return layout

    def refresh(self) -> None:
        """
        Match the controls to the dataset that is now loaded. A 1D or 2D result is already one plane, so the controls go insensitive and the viewer shows it flat by itself.
        """

        can_slice = self._viewer.can_slice()
        self._checkbox.setEnabled(can_slice)
        self._axis_combo.setEnabled(can_slice and self._checkbox.isChecked())
        self._position_slider.setEnabled(can_slice and self._checkbox.isChecked())
        if not can_slice:
            self._checkbox.setToolTip("This result has fewer than three axes, so it is already a cross-section.")
            self._position_label.setText("-")
            return
        self._checkbox.setToolTip("Cut the result on one axis and face the camera at that plane.")
        self._apply_position()

    def _apply_position(self) -> None:
        extent = self._viewer.read_slice_extent(self._axis_combo.currentIndex())
        if extent is None:
            self._position_label.setText("-")
            return
        low, high, unit = extent
        position = low + (high - low) * self._position_slider.value() / SLIDER_STEPS
        self._position_label.setText(f"{position:.3g} {unit}")
        self._viewer.set_slice_position(position)

    def _handle_toggled(self, checked: bool) -> None:
        self._axis_combo.setEnabled(checked)
        self._position_slider.setEnabled(checked)
        if checked:
            self._apply_position()
        self._viewer.set_show_slice(checked)

    def _handle_axis_changed(self, index: int) -> None:
        self._position_slider.blockSignals(True)
        self._position_slider.setValue(SLIDER_STEPS // 2)
        self._position_slider.blockSignals(False)
        self._viewer.set_slice_axis(index)
        self._apply_position()

    def _handle_position_changed(self, _: int) -> None:
        self._apply_position()
