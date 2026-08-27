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
            "camera_trigger_sensor": IoChannelConfig(channel=0, active_high=False),
            "reject_position_sensor": IoChannelConfig(channel=1, active_high=False),
            "start_button": IoChannelConfig(channel=2, active_high=True),
            "stop_button": IoChannelConfig(channel=3, active_high=False),
            "reserved_in_4": IoChannelConfig(channel=4, active_high=True),
            "safety_ok": IoChannelConfig(channel=5, active_high=True),
            "end_test_sensor": IoChannelConfig(channel=6, active_high=True),
            "good_outlet_sensor": IoChannelConfig(channel=7, active_high=True),
            "waste_outlet_sensor": IoChannelConfig(channel=8, active_high=True),
            "door_closed": IoChannelConfig(channel=9, active_high=True),
            "door_upper_closed": IoChannelConfig(channel=10, active_high=True),
        },
        do={
            "tower_red": IoChannelConfig(channel=0, active_high=False),
            "tower_green": IoChannelConfig(channel=1, active_high=False),
            "tower_blue": IoChannelConfig(channel=2, active_high=False),
            "waste_removal": IoChannelConfig(channel=3, active_high=False),
            "conveyor_run": IoChannelConfig(channel=4, active_high=False),
            "button_green": IoChannelConfig(channel=5, active_high=False),
            "button_blue": IoChannelConfig(channel=7, active_high=False),
            "buzzer": IoChannelConfig(channel=8, active_high=False),
            "button_red": IoChannelConfig(channel=9, active_high=False),
        },
    )


__all__ = ["IoChannelConfig", "IoConfig", "default_io_config"]
