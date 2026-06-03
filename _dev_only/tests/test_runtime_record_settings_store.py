from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from ui.shell.dialogs import RuntimeRecordSettingsStore


class _TempRuntimeRecordSettingsStore(RuntimeRecordSettingsStore):
    def __init__(self, path: Path) -> None:
        self._path = path

    def path(self) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self._path


class RuntimeRecordSettingsStoreTest(unittest.TestCase):
    def test_save_and_load_preserve_records_and_images_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _TempRuntimeRecordSettingsStore(Path(tmpdir) / "runtime_record_settings.json")
            store.save(
                {
                    "runtime_records_dir": r"C:\records",
                    "runtime_images_dir": r"C:\images",
                }
            )

            loaded = store.load()

            self.assertEqual(loaded["runtime_records_dir"], r"C:\records")
            self.assertEqual(loaded["runtime_images_dir"], r"C:\images")

    def test_load_legacy_settings_defaults_runtime_images_directory_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "runtime_record_settings.json"
            settings_path.write_text(
                json.dumps({"runtime_records_dir": r"C:\records"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            store = _TempRuntimeRecordSettingsStore(settings_path)

            loaded = store.load()

            self.assertEqual(loaded["runtime_records_dir"], r"C:\records")
            self.assertEqual(loaded["runtime_images_dir"], "")


if __name__ == "__main__":
    unittest.main()
