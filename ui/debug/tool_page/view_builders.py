from __future__ import annotations

from importlib import import_module

from .action_panel_view import build_action_panel
from .sample_panel_view import build_sample_panel
from .tool_config_view import build_tool_config_panel


def build_camera_debug_page(*args, **kwargs):
    module = import_module(f"{__package__}.camera_debug_view")
    return module.build_camera_debug_page(*args, **kwargs)


def build_io_debug_page(*args, **kwargs):
    module = import_module(f"{__package__}.io_debug_view")
    return module.build_io_debug_page(*args, **kwargs)


__all__ = [
    "build_action_panel",
    "build_camera_debug_page",
    "build_io_debug_page",
    "build_sample_panel",
    "build_tool_config_panel",
]
