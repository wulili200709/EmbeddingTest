from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IoChannelConfig:
    name: str
    channel: int
    active_high: bool = True


def business_to_level(on: bool, active_high: bool) -> bool:
    return bool(on) if active_high else not bool(on)


def level_to_business(level: bool, active_high: bool) -> bool:
    return bool(level) if active_high else not bool(level)


class IoMapping:
    """Business-name to physical DI/DO channel mapping."""

    def __init__(
        self,
        *,
        di: dict[str, IoChannelConfig] | None = None,
        do: dict[str, IoChannelConfig] | None = None,
    ) -> None:
        self._di = dict(di or {})
        self._do = dict(do or {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IoMapping":
        di_data = data.get("di", {})
        do_data = data.get("do", {})
        if not isinstance(di_data, dict) or not isinstance(do_data, dict):
            raise ValueError("io mapping requires dict fields: 'di' and 'do'")
        di = {name: cls._parse_channel_config(name, item, default_active_high=True) for name, item in di_data.items()}
        do = {name: cls._parse_channel_config(name, item, default_active_high=True) for name, item in do_data.items()}
        return cls(di=di, do=do)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "IoMapping":
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"invalid io mapping file: {file_path}")
        return cls.from_dict(data)

    @classmethod
    def default_debug_mapping(cls) -> "IoMapping":
        return cls.from_dict(
            {
                "di": {
                    "foot_switch": {"channel": 0, "active_high": True},
                },
                "do": {
                    "tower_red": {"channel": 0, "active_high": True},
                    "tower_green": {"channel": 1, "active_high": True},
                    "tower_blue": {"channel": 2, "active_high": True},
                    # Light polarity is still site-specific; active_high here is just a default.
                    "light_cam1": {"channel": 3, "active_high": False},
                    "light_cam2": {"channel": 4, "active_high": False},
                },
            }
        )

    @staticmethod
    def _parse_channel_config(name: str, item: Any, default_active_high: bool) -> IoChannelConfig:
        if isinstance(item, int):
            return IoChannelConfig(name=name, channel=int(item), active_high=default_active_high)
        if not isinstance(item, dict):
            raise ValueError(f"invalid channel config for '{name}': {item!r}")
        if "channel" not in item:
            raise ValueError(f"missing 'channel' for '{name}'")
        active_high = bool(item.get("active_high", default_active_high))
        return IoChannelConfig(name=name, channel=int(item["channel"]), active_high=active_high)

    def di_names(self) -> list[str]:
        return list(self._di.keys())

    def do_names(self) -> list[str]:
        return list(self._do.keys())

    def get_input(self, name: str) -> IoChannelConfig:
        try:
            return self._di[name]
        except KeyError as exc:
            raise KeyError(f"unknown DI mapping: {name}") from exc

    def get_output(self, name: str) -> IoChannelConfig:
        try:
            return self._do[name]
        except KeyError as exc:
            raise KeyError(f"unknown DO mapping: {name}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "di": {
                name: {"channel": cfg.channel, "active_high": cfg.active_high}
                for name, cfg in self._di.items()
            },
            "do": {
                name: {"channel": cfg.channel, "active_high": cfg.active_high}
                for name, cfg in self._do.items()
            },
        }
