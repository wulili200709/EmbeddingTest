from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from application.runtime.conveyor import _load_conveyor_config


class _Signal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, message: str) -> None:
        self.messages.append(message)


class _Runtime:
    def __init__(self) -> None:
        self.logAppended = _Signal()
        self._conveyor_config_path = None


class ConveyorRuntimeConfigTests(unittest.TestCase):
    def test_invalid_configuration_stops_startup_instead_of_using_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config" / "defaults"
            config_dir.mkdir(parents=True)
            (config_dir / "conveyor_control.json").write_text(
                '{"upper_door_sensor_enabled": "false"}',
                encoding="utf-8",
            )
            runtime = _Runtime()

            with patch(
                "application.runtime.conveyor.packaged_embedding_test_root",
                return_value=root,
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid conveyor configuration"):
                    _load_conveyor_config(runtime)

            self.assertTrue(runtime.logAppended.messages)
            self.assertNotIn("using defaults", runtime.logAppended.messages[-1])


if __name__ == "__main__":
    unittest.main()
