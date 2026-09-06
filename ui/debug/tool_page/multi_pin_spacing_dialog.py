"""Editor for per-gap multi-pin spacing specifications."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ui.i18n import tr


def normalized_spacing_specs(raw_specs: object, gap_count: int) -> list[dict[str, float]]:
    source = list(raw_specs) if isinstance(raw_specs, (list, tuple)) else []
    result: list[dict[str, float]] = []
    for index in range(max(0, int(gap_count))):
        raw = source[index] if index < len(source) and isinstance(source[index], dict) else {}
        try:
            nominal = max(0.0, float(raw.get("nominal", 0.0) or 0.0))
            lower = max(0.0, float(raw.get("lower_tolerance", 0.0) or 0.0))
            upper = max(0.0, float(raw.get("upper_tolerance", 0.0) or 0.0))
        except (TypeError, ValueError):
            nominal, lower, upper = 0.0, 0.0, 0.0
        result.append(
            {
                "nominal": nominal,
                "lower_tolerance": lower,
                "upper_tolerance": upper,
            }
        )
    return result


class MultiPinSpacingDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent,
        *,
        expected_pin_count: int,
        unit: str,
        spacing_specs: object,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("debug.measurement.spacing_dialog_title"))
        self.resize(680, 500)
        self._unit = str(unit or "px")
        self._gap_count = max(0, int(expected_pin_count) - 1)
        self._spins: list[tuple[QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox]] = []

        root = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            tr(
                "debug.measurement.spacing_dialog_hint",
                pins=int(expected_pin_count),
                gaps=self._gap_count,
                unit=self._unit,
            )
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        defaults = QtWidgets.QHBoxLayout()
        self.default_nominal = self._make_spin()
        self.default_lower = self._make_spin()
        self.default_upper = self._make_spin()
        defaults.addWidget(QtWidgets.QLabel(tr("debug.measurement.spacing_nominal")))
        defaults.addWidget(self.default_nominal)
        defaults.addWidget(QtWidgets.QLabel(tr("debug.measurement.spacing_lower_tolerance")))
        defaults.addWidget(self.default_lower)
        defaults.addWidget(QtWidgets.QLabel(tr("debug.measurement.spacing_upper_tolerance")))
        defaults.addWidget(self.default_upper)
        apply_button = QtWidgets.QPushButton(tr("debug.measurement.spacing_apply_all"))
        apply_button.clicked.connect(self._apply_defaults)
        defaults.addWidget(apply_button)
        root.addLayout(defaults)

        self.table = QtWidgets.QTableWidget(self._gap_count, 5)
        self.table.setHorizontalHeaderLabels(
            [
                tr("debug.measurement.spacing_position"),
                tr("debug.measurement.spacing_nominal"),
                tr("debug.measurement.spacing_lower_tolerance"),
                tr("debug.measurement.spacing_upper_tolerance"),
                tr("debug.measurement.spacing_range"),
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for column in (1, 2, 3, 4):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Stretch)
        specs = normalized_spacing_specs(spacing_specs, self._gap_count)
        for row, spec in enumerate(specs):
            position = QtWidgets.QTableWidgetItem(f"P{row + 1}–P{row + 2}")
            position.setFlags(position.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, position)
            nominal = self._make_spin(spec["nominal"])
            lower = self._make_spin(spec["lower_tolerance"])
            upper = self._make_spin(spec["upper_tolerance"])
            self._spins.append((nominal, lower, upper))
            self.table.setCellWidget(row, 1, nominal)
            self.table.setCellWidget(row, 2, lower)
            self.table.setCellWidget(row, 3, upper)
            nominal.valueChanged.connect(lambda _value, r=row: self._update_range(r))
            lower.valueChanged.connect(lambda _value, r=row: self._update_range(r))
            upper.valueChanged.connect(lambda _value, r=row: self._update_range(r))
            self._update_range(row)
        if specs:
            self.default_nominal.setValue(specs[0]["nominal"])
            self.default_lower.setValue(specs[0]["lower_tolerance"])
            self.default_upper.setValue(specs[0]["upper_tolerance"])
        root.addWidget(self.table, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _make_spin(self, value: float = 0.0) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(0.0, 1000000.0)
        spin.setSingleStep(0.01)
        spin.setKeyboardTracking(False)
        spin.setSuffix(f" {self._unit}")
        spin.setValue(float(value))
        return spin

    def _apply_defaults(self) -> None:
        nominal = float(self.default_nominal.value())
        lower = float(self.default_lower.value())
        upper = float(self.default_upper.value())
        for row, (nominal_spin, lower_spin, upper_spin) in enumerate(self._spins):
            nominal_spin.setValue(nominal)
            lower_spin.setValue(lower)
            upper_spin.setValue(upper)
            self._update_range(row)

    def _update_range(self, row: int) -> None:
        if row < 0 or row >= len(self._spins):
            return
        nominal, lower, upper = self._spins[row]
        minimum = float(nominal.value()) - float(lower.value())
        maximum = float(nominal.value()) + float(upper.value())
        item = QtWidgets.QTableWidgetItem(f"{minimum:.4f}～{maximum:.4f} {self._unit}")
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 4, item)

    def spacing_specs(self) -> list[dict[str, float]]:
        return [
            {
                "nominal": float(nominal.value()),
                "lower_tolerance": float(lower.value()),
                "upper_tolerance": float(upper.value()),
            }
            for nominal, lower, upper in self._spins
        ]


__all__ = ["MultiPinSpacingDialog", "normalized_spacing_specs"]
