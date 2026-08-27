"""Debug DI/DO workflow helpers for ToolPage."""

from __future__ import annotations

from PySide6 import QtWidgets

from ui.i18n import tr

_DEBUG_IO_NAME_LABELS = {
    "camera_trigger_sensor": ("debug.io_name.camera_trigger_sensor", "DI0_CAMERA_TRIGGER"),
    "reject_position_sensor": ("debug.io_name.reject_position_sensor", "DI1_REJECT_POSITION"),
    "start_button": ("debug.io_name.start_button", "DI2_START"),
    "stop_button": ("debug.io_name.stop_button", "DI3_STOP"),
    "reserved_in_4": ("debug.io_name.reserved_in_4", "DI4_RESERVED"),
    "safety_ok": ("debug.io_name.safety_ok", "DI5_SAFETY_OK"),
    "end_test_sensor": ("debug.io_name.end_test_sensor", "DI6_END_TEST"),
    "good_outlet_sensor": ("debug.io_name.good_outlet_sensor", "DI7_GOOD_OUTLET"),
    "waste_outlet_sensor": ("debug.io_name.waste_outlet_sensor", "DI8_WASTE_OUTLET"),
    "door_closed": ("debug.io_name.door_closed", "DI9_DOOR_CLOSED"),
    "door_upper_closed": ("debug.io_name.door_upper_closed", "DI10_UPPER_DOOR_CLOSED"),
    "tower_red": ("debug.io_name.tower_red", "DO_TOWER_RED"),
    "tower_green": ("debug.io_name.tower_green", "DO_TOWER_GREEN"),
    "tower_blue": ("debug.io_name.tower_blue", "DO_TOWER_BLUE"),
    "waste_removal": ("debug.io_name.waste_removal", "DO3_WASTE_REMOVAL"),
    "conveyor_run": ("debug.io_name.conveyor_run", "DO4_CONVEYOR_RUN"),
    "button_green": ("debug.io_name.button_green", "DO5_BUTTON_GREEN"),
    "button_blue": ("debug.io_name.button_blue", "DO7_BUTTON_BLUE"),
    "buzzer": ("debug.io_name.buzzer", "DO_BUZZER"),
    "button_red": ("debug.io_name.button_red", "DO9_BUTTON_RED"),
}



def _debug_io_display_name(name: str, channel: int | None = None) -> str:
    entry = _DEBUG_IO_NAME_LABELS.get(str(name))
    if entry is None:
        return str(name)
    key, code = entry
    label = tr(key)
    if label == key:
        label = str(name)
    return f"{label}\n{code}"


def _debug_io_channel_maps(io_controller):
    di_map = {}
    do_map = {}
    mapping = getattr(io_controller, "mapping", None)
    if mapping is None:
        return di_map, do_map
    for name in mapping.di_names():
        cfg = mapping.get_input(name)
        di_map[int(cfg.channel)] = (name, cfg)
    for name in mapping.do_names():
        cfg = mapping.get_output(name)
        do_map[int(cfg.channel)] = (name, cfg)
    return di_map, do_map


def _set_debug_di_indicator(indicator, active: bool) -> None:
    indicator.setStyleSheet(
        "background:#2fbf71;border:2px solid #86efac;border-radius:16px;"
        if active
        else "background:#7a7a7a;border:2px solid #9a9a9a;border-radius:16px;"
    )


def _reset_debug_io_panels(self) -> None:
    self._debug_output_buttons.clear()
    for channel, card in self._debug_di_cards.items():
        card.setVisible(False)
    for channel, indicator in self._debug_di_indicators.items():
        _set_debug_di_indicator(indicator, False)
        indicator.setToolTip(f"DI_{channel}\nDisconnected")
    for channel, hint in self._debug_di_hints.items():
        hint.setText(tr("debug.unmapped"))
        hint.setToolTip(f"DI_{channel}\nUnmapped")
    for channel, card in self._debug_do_cards.items():
        card.setVisible(False)
    for channel, button in self._debug_do_channel_buttons.items():
        button.blockSignals(True)
        button.setChecked(False)
        button.setEnabled(False)
        button.blockSignals(False)
        button.setToolTip(f"DO_{channel}\nDisconnected")
    for channel, hint in self._debug_do_hints.items():
        hint.setText(tr("debug.unmapped"))
        hint.setToolTip(f"DO_{channel}\nUnmapped")
    self.lbl_debug_io_mapping_summary.setText(tr("debug.mapping_not_loaded"))


def _update_debug_io_panels(self, di_word: int, do_word: int) -> None:
    di_map, do_map = _debug_io_channel_maps(self._debug_io_controller)
    self._debug_output_buttons.clear()

    for channel, indicator in self._debug_di_indicators.items():
        mapped = di_map.get(channel)
        self._debug_di_cards[channel].setVisible(mapped is not None)
        if mapped is None:
            continue

        name, cfg = mapped
        raw_state = bool(int(di_word) & (1 << channel))
        display_state = raw_state if cfg.active_high else not raw_state
        tooltip = (
            f"DI_{channel}\n"
            f"name: {name}\n"
            f"active: {'high' if cfg.active_high else 'low'}\n"
            f"level: {'HIGH' if raw_state else 'LOW'}\n"
            f"logic: {'ON' if display_state else 'OFF'}"
        )
        _set_debug_di_indicator(indicator, display_state)
        indicator.setToolTip(tooltip)
        hint = self._debug_di_hints[channel]
        hint.setText(_debug_io_display_name(name, channel))
        hint.setToolTip(tooltip)

    for channel, button in self._debug_do_channel_buttons.items():
        mapped = do_map.get(channel)
        self._debug_do_cards[channel].setVisible(mapped is not None)
        if mapped is None:
            button.blockSignals(True)
            button.setEnabled(False)
            button.setChecked(False)
            button.blockSignals(False)
            continue

        name, cfg = mapped
        raw_state = bool(int(do_word) & (1 << channel))
        display_state = raw_state if cfg.active_high else not raw_state
        tooltip = (
            f"DO_{channel}\n"
            f"name: {name}\n"
            f"active: {'high' if cfg.active_high else 'low'}\n"
            f"level: {'HIGH' if raw_state else 'LOW'}\n"
            f"logic: {'ON' if display_state else 'OFF'}"
        )
        self._debug_output_buttons[name] = button
        button.blockSignals(True)
        button.setEnabled(True)
        button.setChecked(display_state)
        button.blockSignals(False)
        button.setToolTip(tooltip)
        hint = self._debug_do_hints[channel]
        hint.setText(_debug_io_display_name(name, channel))
        hint.setToolTip(tooltip)

    input_states = []
    for channel, (name, cfg) in sorted(di_map.items()):
        raw_state = bool(int(di_word) & (1 << channel))
        state = raw_state if cfg.active_high else not raw_state
        input_states.append(f"{name}={'ON' if state else 'OFF'}")

    output_states = []
    for channel, (name, cfg) in sorted(do_map.items()):
        raw_state = bool(int(do_word) & (1 << channel))
        state = raw_state if cfg.active_high else not raw_state
        output_states.append(f"{name}={'ON' if state else 'OFF'}")

    self.lbl_debug_di_snapshot.setText(
        f"DI 0x{int(di_word):04X}" + (f"  {'  '.join(input_states)}" if input_states else "")
    )
    self.lbl_debug_do_snapshot.setText(
        f"DO 0x{int(do_word):04X}" + (f"  {'  '.join(output_states)}" if output_states else "")
    )
    enabled_channels = ", ".join(f"DO_{channel}" for channel in sorted(do_map)) or "-"
    self.lbl_debug_io_mapping_summary.setText(
        f"Mapping: DI {len(di_map)} / DO {len(do_map)}; enabled output channels {enabled_channels}"
    )


def _open_debug_io(self) -> None:
    runtime_ctrl = self.runtime_controller()
    if runtime_ctrl is None:
        QtWidgets.QMessageBox.warning(self, "DI/DO Debug", "Runtime IO service is unavailable in the current environment")
        return

    if getattr(self, "_runtime_io_ready", False) and getattr(self, "_runtime_io_controller", None) is not None:
        self._apply_runtime_io_debug_state()
        self.lbl_status.setText("Status: DI/DO debug attached to runtime IO")
        return

    if not runtime_ctrl.initialize_startup_io(force=True):
        detail = getattr(self, "_runtime_io_status_detail", "") or "unknown error"
        QtWidgets.QMessageBox.critical(self, "DI/DO Debug", f"Failed to open IO debug: {detail}")
        return

    self._apply_runtime_io_debug_state()
    self.lbl_status.setText("Status: DI/DO debug reloaded from runtime IO")


def _close_debug_io(self, *, silent: bool = False) -> None:
    self._debug_io_timer.stop()
    if self._debug_io_controller is not None and not getattr(self, "_debug_io_uses_runtime_controller", False):
        try:
            self._debug_io_controller.clear_outputs()
        except Exception:
            pass
        try:
            self._debug_io_controller.close()
        except Exception:
            pass
    if getattr(self, "_debug_io_uses_runtime_controller", False):
        self._debug_io_uses_runtime_controller = False
    self._debug_io_controller = None
    self.lbl_debug_di_snapshot.setText(tr("debug.di_disconnected"))
    self.lbl_debug_do_snapshot.setText(tr("debug.do_disconnected"))
    _reset_debug_io_panels(self)
    if getattr(self, "_runtime_io_ready", False) and getattr(self, "_runtime_io_controller", None) is not None:
        self._apply_runtime_io_debug_state()
    if not silent:
        self.lbl_status.setText("Status: DI/DO debug closed")


def _refresh_debug_io_snapshot(self) -> None:
    if self._debug_io_controller is None or not self._debug_io_controller.is_open:
        return
    try:
        di_word = self._debug_io_controller.board.read_di_word()
        do_word = self._debug_io_controller.board.read_do_word()
    except Exception as exc:
        self.lbl_debug_di_snapshot.setText(f"DI read failed ({exc})")
        self.lbl_debug_do_snapshot.setText(f"DO read failed ({exc})")
        self.lbl_debug_io_mapping_summary.setText("Mapping: read failed")
        return
    _update_debug_io_panels(self, di_word, do_word)


def _set_debug_output_channel(self, channel: int, on: bool) -> None:
    button = self._debug_do_channel_buttons.get(int(channel))
    if self._debug_io_controller is None or not self._debug_io_controller.is_open:
        QtWidgets.QMessageBox.information(self, "DI/DO Debug", "Open IO debug first")
        if button is not None:
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        return

    _di_map, do_map = _debug_io_channel_maps(self._debug_io_controller)
    mapped = do_map.get(int(channel))
    if mapped is None:
        if button is not None:
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        QtWidgets.QMessageBox.information(self, "DI/DO Debug", f"DO_{channel} is not mapped and cannot be written")
        return

    name, _cfg = mapped
    try:
        self._debug_io_controller.set_output(name, on)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(self, "DI/DO Debug", f"{name} output failed: {exc}")
    self._refresh_debug_io_snapshot()


def _set_debug_output(self, name: str, on: bool) -> None:
    if self._debug_io_controller is None or not self._debug_io_controller.is_open:
        QtWidgets.QMessageBox.information(self, "DI/DO Debug", "Open IO debug first")
        return
    _di_map, do_map = _debug_io_channel_maps(self._debug_io_controller)
    for channel, (mapped_name, _cfg) in do_map.items():
        if mapped_name == name:
            self._set_debug_output_channel(channel, on)
            return
    QtWidgets.QMessageBox.information(self, "DI/DO Debug", f"{name} has no mapped output channel")
