"""Default IO mapping helpers for the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class IoChannelConfig:
    channel: int
    active_high: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "channel": int(self.channel),
            "active_high": bool(self.active_high),
        }


@dataclass(frozen=True)
class IoConfig:
    di: Dict[str, IoChannelConfig]
    do: Dict[str, IoChannelConfig]

    def to_dict(self) -> Dict[str, object]:
        return {
            "di": {name: cfg.to_dict() for name, cfg in self.di.items()},
            "do": {name: cfg.to_dict() for name, cfg in self.do.items()},
        }


def default_io_config() -> IoConfig:
    return IoConfig(
        di={
            "foot_switch": IoChannelConfig(channel=0, active_high=True),
            "reject_signal": IoChannelConfig(channel=1, active_high=True),
            "reserved_in_1": IoChannelConfig(channel=2, active_high=True),
            "reserved_in_2": IoChannelConfig(channel=3, active_high=True),
        },
        do={
            "tower_red": IoChannelConfig(channel=0, active_high=False),
            "tower_green": IoChannelConfig(channel=1, active_high=False),
            "tower_blue": IoChannelConfig(channel=2, active_high=False),
            "light_cam1": IoChannelConfig(channel=3, active_high=False),
            "light_cam2": IoChannelConfig(channel=4, active_high=False),
            "buzzer": IoChannelConfig(channel=5, active_high=False),
            "reserved_out_2": IoChannelConfig(channel=6, active_high=False),
        },
    )


__all__ = ["IoChannelConfig", "IoConfig", "default_io_config"]
