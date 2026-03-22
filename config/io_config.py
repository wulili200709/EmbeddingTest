"""
io_config.py

第一版 IO 默认配置。

当前已确认：
  - 三色灯低电平有效
  - 光源极性待确认，因此保留为可配置项
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class IoChannelConfig:
    channel: str
    active_high: bool


@dataclass(frozen=True)
class IoConfig:
    board_model: str
    foot_switch_di: str
    tower_red: IoChannelConfig
    tower_green: IoChannelConfig
    tower_blue: IoChannelConfig
    light_cam1: IoChannelConfig
    light_cam2: IoChannelConfig

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_io_config() -> IoConfig:
    return IoConfig(
        board_model="NP-6133-16I16O",
        foot_switch_di="DI0",
        tower_red=IoChannelConfig(channel="DO0", active_high=False),
        tower_green=IoChannelConfig(channel="DO1", active_high=False),
        tower_blue=IoChannelConfig(channel="DO2", active_high=False),
        # 光源极性暂未确认：这里先给出占位默认值，后续以现场配置为准。
        light_cam1=IoChannelConfig(channel="DO3", active_high=False),
        light_cam2=IoChannelConfig(channel="DO4", active_high=False),
    )


__all__ = ["IoChannelConfig", "IoConfig", "default_io_config"]
