from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io_mapping import IoMapping, business_to_level, level_to_business
from .nkio_board import NkioBoard


class IoController:
    """Business-facing IO controller built on top of NkioBoard and IoMapping."""

    def __init__(
        self,
        board: NkioBoard,
        mapping: IoMapping,
    ) -> None:
        self.board = board
        self.mapping = mapping

    @classmethod
    def from_config_file(
        cls,
        board_config_file: str | Path,
        mapping_file: str | Path,
        *,
        dll_path: str | Path | None = None,
    ) -> "IoController":
        board = NkioBoard(config_file=board_config_file, dll_path=dll_path)
        mapping = IoMapping.from_json_file(mapping_file)
        return cls(board=board, mapping=mapping)

    @classmethod
    def default_debug_controller(
        cls,
        board_config_file: str | Path,
        *,
        dll_path: str | Path | None = None,
    ) -> "IoController":
        board = NkioBoard(config_file=board_config_file, dll_path=dll_path)
        mapping = IoMapping.default_debug_mapping()
        return cls(board=board, mapping=mapping)

    def open(self) -> None:
        self.board.open()

    def close(self) -> None:
        self.board.close()

    @property
    def is_open(self) -> bool:
        return self.board.is_open

    def read_input(self, name: str) -> bool:
        cfg = self.mapping.get_input(name)
        level = self.board.read_di_channel(cfg.channel)
        return level_to_business(level, cfg.active_high)

    def read_output(self, name: str) -> bool:
        cfg = self.mapping.get_output(name)
        level = self.board.read_do_channel(cfg.channel)
        return level_to_business(level, cfg.active_high)

    def set_output(self, name: str, on: bool) -> None:
        cfg = self.mapping.get_output(name)
        level = business_to_level(on, cfg.active_high)
        self.board.write_do_channel(cfg.channel, level)

    def set_outputs(self, updates: dict[str, bool]) -> None:
        raw_updates: dict[int, bool] = {}
        for name, on in updates.items():
            cfg = self.mapping.get_output(name)
            raw_updates[cfg.channel] = business_to_level(on, cfg.active_high)
        self.board.set_do_channels(raw_updates)

    def clear_outputs(self, names: Iterable[str] | None = None) -> None:
        target_names = list(names) if names is not None else self.mapping.do_names()
        self.set_outputs({name: False for name in target_names})

    def snapshot_inputs(self) -> dict[str, bool]:
        return {name: self.read_input(name) for name in self.mapping.di_names()}

    def snapshot_outputs(self) -> dict[str, bool]:
        return {name: self.read_output(name) for name in self.mapping.do_names()}

    def set_tower_light(
        self,
        *,
        red: bool | None = None,
        green: bool | None = None,
        blue: bool | None = None,
    ) -> None:
        updates: dict[str, bool] = {}
        if red is not None:
            updates["tower_red"] = bool(red)
        if green is not None:
            updates["tower_green"] = bool(green)
        if blue is not None:
            updates["tower_blue"] = bool(blue)
        if updates:
            self.set_outputs(updates)

    def tower_all_off(self) -> None:
        self.set_tower_light(red=False, green=False, blue=False)

    def set_camera_light(self, camera_index: int, on: bool) -> None:
        if int(camera_index) == 1:
            self.set_output("light_cam1", on)
            return
        if int(camera_index) == 2:
            self.set_output("light_cam2", on)
            return
        raise ValueError(f"unsupported camera index: {camera_index}")

    def read_foot_switch(self) -> bool:
        return self.read_input("foot_switch")
