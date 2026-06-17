from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.product_session import ProductSession, SessionData
from infrastructure.json_backup import (
    BACKUP_DIR_NAME,
    PRODUCT_JSON_TYPES,
    load_json_with_recovery,
    protect_product_json_files,
    write_json_with_backup,
)


_LOCAL_TMP_ROOT = PROJECT_DIR / ".tmp_test_runs"
_LOCAL_TMP_ROOT.mkdir(exist_ok=True)


def _make_temp_root() -> Path:
    root = _LOCAL_TMP_ROOT / f"json_backup_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class JsonBackupRecoveryTest(unittest.TestCase):
    @staticmethod
    def _valid_session_payload(**overrides) -> dict:
        payload = {
            "runtime_cam1_serial": "DA9521010",
            "runtime_cam2_serial": "",
            "runtime_capture_policy": "ng_only",
            "foot_trigger_delay_ms": 470,
            "ng_stop_delay_ms": 2000,
        }
        payload.update(overrides)
        return payload

    def test_corrupt_primary_restores_latest_valid_backup(self) -> None:
        root = _make_temp_root()
        try:
            path = root / "session.json"
            expected = self._valid_session_payload()
            write_json_with_backup(path, expected, expected_type=dict)

            path.write_bytes(b"\xff\xfe\x00broken json")

            loaded = load_json_with_recovery(path, expected_type=dict)

            self.assertEqual(loaded, expected)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), expected)
            self.assertTrue((root / BACKUP_DIR_NAME / "recovery.log").is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_recovery_skips_corrupt_newest_backup(self) -> None:
        root = _make_temp_root()
        try:
            path = root / "product_params.json"
            write_json_with_backup(path, {"topk": 1}, expected_type=dict)
            write_json_with_backup(path, {"topk": 2}, expected_type=dict)

            backups = sorted(
                (root / BACKUP_DIR_NAME).glob("product_params.json.*.bak"),
                reverse=True,
            )
            self.assertGreaterEqual(len(backups), 2)
            backups[0].write_text("{broken", encoding="utf-8")
            path.write_text("{also broken", encoding="utf-8")

            loaded = load_json_with_recovery(path, expected_type=dict)

            self.assertEqual(loaded, {"topk": 1})
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_rolling_backup_limit_is_enforced(self) -> None:
        root = _make_temp_root()
        try:
            path = root / "inspection_items.json"
            for index in range(8):
                write_json_with_backup(
                    path,
                    [{"item_id": str(index)}],
                    expected_type=list,
                    max_backups=3,
                )

            backups = list(
                (root / BACKUP_DIR_NAME).glob("inspection_items.json.*.bak")
            )
            self.assertEqual(len(backups), 3)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_startup_protects_all_four_product_json_files(self) -> None:
        root = _make_temp_root()
        try:
            product_dir = root / "ProductA"
            product_dir.mkdir()
            payloads = {
                "session.json": self._valid_session_payload(
                    runtime_cam1_serial="CAM-A",
                ),
                "product_params.json": {"topk": 3},
                "inspection_items.json": [],
                "sample_annotations.json": {"images": {}},
            }
            for file_name, payload in payloads.items():
                (product_dir / file_name).write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )

            protect_product_json_files(root)

            backup_dir = product_dir / BACKUP_DIR_NAME
            for file_name in PRODUCT_JSON_TYPES:
                self.assertEqual(
                    len(list(backup_dir.glob(f"{file_name}.*.bak"))),
                    1,
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_product_session_recovers_camera_binding(self) -> None:
        root = _make_temp_root()
        try:
            session = ProductSession(str(root))
            session.load()
            session.switch_product("Default")
            session.save_session(
                SessionData(
                    runtime_cam1_serial="CAM-PRIMARY",
                    runtime_cam2_serial="CAM-SECONDARY",
                    foot_trigger_delay_ms=470,
                    ng_stop_delay_ms=2000,
                )
            )
            Path(session.session_json).write_bytes(b"\x80\x81corrupt")

            loaded = session.load_session()

            self.assertEqual(loaded.runtime_cam1_serial, "CAM-PRIMARY")
            self.assertEqual(loaded.runtime_cam2_serial, "CAM-SECONDARY")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_empty_camera_serials_restore_older_complete_session(self) -> None:
        root = _make_temp_root()
        try:
            path = root / "session.json"
            expected = self._valid_session_payload()
            write_json_with_backup(path, expected, expected_type=dict)
            path.write_text(
                json.dumps(
                    self._valid_session_payload(
                        runtime_cam1_serial="",
                        runtime_cam2_serial="",
                    )
                ),
                encoding="utf-8",
            )

            loaded = load_json_with_recovery(path, expected_type=dict)

            self.assertEqual(loaded, expected)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_zero_delays_are_not_backed_up_and_restore_older_session(self) -> None:
        root = _make_temp_root()
        try:
            path = root / "session.json"
            expected = self._valid_session_payload()
            write_json_with_backup(path, expected, expected_type=dict)
            write_json_with_backup(
                path,
                self._valid_session_payload(
                    foot_trigger_delay_ms=0,
                    ng_stop_delay_ms=0,
                ),
                expected_type=dict,
            )

            backups = list((root / BACKUP_DIR_NAME).glob("session.json.*.bak"))
            self.assertEqual(len(backups), 1)
            loaded = load_json_with_recovery(path, expected_type=dict)

            self.assertEqual(loaded, expected)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_startup_removes_existing_session_backup_with_zero_delay(self) -> None:
        root = _make_temp_root()
        try:
            product_dir = root / "ProductA"
            backup_dir = product_dir / BACKUP_DIR_NAME
            backup_dir.mkdir(parents=True)
            valid_payload = self._valid_session_payload()
            (product_dir / "session.json").write_text(
                json.dumps(valid_payload),
                encoding="utf-8",
            )
            invalid_backup = backup_dir / "session.json.20260101-000000-000000-old.bak"
            invalid_backup.write_text(
                json.dumps(
                    self._valid_session_payload(
                        foot_trigger_delay_ms=0,
                    )
                ),
                encoding="utf-8",
            )

            protect_product_json_files(root)

            self.assertFalse(invalid_backup.exists())
            self.assertEqual(
                len(list(backup_dir.glob("session.json.*.bak"))),
                1,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
