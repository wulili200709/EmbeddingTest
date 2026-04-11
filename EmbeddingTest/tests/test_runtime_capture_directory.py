from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.runtime.execution import _save_frame


class _DummySession:
    def __init__(self, product_dir: str = "", current_product: str = "Default") -> None:
        self.product_dir = product_dir
        self.current_product = current_product


class _DummyRuntime:
    def __init__(self, capture_dir: Path, product_dir: str = "", current_product: str = "Default") -> None:
        self._session = _DummySession(product_dir, current_product=current_product)
        self._runtime_capture_dir = capture_dir
        self._frame_lock = threading.RLock()
        self._last_capture_paths: dict[str, str] = {}
        self._last_preview_frames: dict[str, object] = {}
        self._log_messages: list[str] = []
        self.logAppended = _DummyEmitter(self._log_messages)


class _DummyEmitter:
    def __init__(self, messages: list[str]) -> None:
        self._messages = messages

    def emit(self, message: str) -> None:
        self._messages.append(str(message))


class RuntimeCaptureDirectoryTest(unittest.TestCase):
    def test_save_frame_exports_into_dated_product_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            capture_dir = Path(tmpdir) / "custom_runtime_capture"
            runtime = _DummyRuntime(capture_dir, current_product="250A_OG")

            saved_path = Path(_save_frame(runtime, "cam1", np.zeros((16, 20, 3), dtype=np.uint8)))

            self.assertTrue(saved_path.exists())
            relative_parts = saved_path.relative_to(capture_dir).parts
            self.assertEqual(len(relative_parts), 3)
            self.assertRegex(relative_parts[0], r"^\d{4}-\d{2}-\d{2}$")
            self.assertEqual(relative_parts[1], "250A_OG")
            self.assertEqual(saved_path.suffix.lower(), ".png")
            self.assertTrue(saved_path.with_suffix(".json").exists())

    def test_save_frame_deletes_oldest_date_folder_when_free_space_is_below_threshold(self) -> None:
        disk_usage = namedtuple("disk_usage", "total used free")
        with tempfile.TemporaryDirectory() as tmpdir:
            capture_dir = Path(tmpdir) / "custom_runtime_capture"
            oldest_day = capture_dir / "2026-03-28" / "Legacy"
            newer_day = capture_dir / "2026-03-29" / "Legacy"
            oldest_day.mkdir(parents=True, exist_ok=True)
            newer_day.mkdir(parents=True, exist_ok=True)
            (oldest_day / "old.png").write_bytes(b"png")
            (newer_day / "new.png").write_bytes(b"png")
            runtime = _DummyRuntime(capture_dir, current_product="250A_OG")

            with patch(
                "application.runtime.execution.shutil.disk_usage",
                side_effect=[
                    disk_usage(total=10, used=9, free=512 * 1024 * 1024),
                    disk_usage(total=10, used=7, free=2 * 1024 * 1024 * 1024),
                ],
            ):
                saved_path = Path(_save_frame(runtime, "cam1", np.zeros((16, 20, 3), dtype=np.uint8)))

            self.assertTrue(saved_path.exists())
            self.assertFalse((capture_dir / "2026-03-28").exists())
            self.assertTrue((capture_dir / "2026-03-29").exists())
            self.assertTrue(any("deleted old capture folder" in message for message in runtime._log_messages))


if __name__ == "__main__":
    unittest.main()
