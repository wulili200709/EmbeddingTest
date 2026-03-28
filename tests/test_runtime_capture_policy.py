from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from application.product_session import ProductSession, SessionData
from application.runtime_controller import (
    RUNTIME_CAPTURE_POLICY_ALL,
    RUNTIME_CAPTURE_POLICY_NG_ONLY,
    delete_capture_artifacts,
    normalize_capture_retention_policy,
    retained_capture_paths_for_policy,
)


class RuntimeCapturePolicyTest(unittest.TestCase):
    def test_session_persists_runtime_capture_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session = ProductSession(tmpdir)
            session.load()
            session.switch_product("Default")

            session.save_session(
                SessionData(
                    runtime_capture_policy=RUNTIME_CAPTURE_POLICY_ALL,
                )
            )

            loaded = session.load_session()
            self.assertEqual(loaded.runtime_capture_policy, RUNTIME_CAPTURE_POLICY_ALL)

    def test_session_stores_relative_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session = ProductSession(tmpdir)
            session.load()
            session.switch_product("Default")
            product_dir = Path(session.product_dir)
            image_path = product_dir / "debug_capture" / "cam1.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"png")

            session.save_session(
                SessionData(
                    ok_files=[str(image_path)],
                    ref_image=str(image_path),
                )
            )

            raw = json.loads(Path(session.session_json).read_text(encoding="utf-8"))
            self.assertEqual(raw["ok_files"], ["debug_capture/cam1.png"])
            self.assertEqual(raw["ref_image"], "debug_capture/cam1.png")

            loaded = session.load_session()
            self.assertEqual(loaded.ok_files, [str(image_path)])
            self.assertEqual(loaded.ref_image, str(image_path))

    def test_ng_only_policy_keeps_paths_only_for_ng(self) -> None:
        capture_paths = {
            "cam1": r"C:\tmp\cam1.png",
            "cam2": r"C:\tmp\cam2.png",
        }

        self.assertEqual(
            retained_capture_paths_for_policy(
                RUNTIME_CAPTURE_POLICY_NG_ONLY,
                "OK",
                capture_paths,
            ),
            {},
        )
        self.assertEqual(
            retained_capture_paths_for_policy(
                RUNTIME_CAPTURE_POLICY_NG_ONLY,
                "NG",
                capture_paths,
            ),
            capture_paths,
        )
        self.assertEqual(
            retained_capture_paths_for_policy(
                RUNTIME_CAPTURE_POLICY_ALL,
                "OK",
                capture_paths,
            ),
            capture_paths,
        )

    def test_delete_capture_artifacts_removes_image_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "runtime_cam1.png"
            json_path = image_path.with_suffix(".json")
            image_path.write_bytes(b"png")
            json_path.write_text("{}", encoding="utf-8")

            delete_capture_artifacts({"cam1": str(image_path)})

            self.assertFalse(image_path.exists())
            self.assertFalse(json_path.exists())

    def test_invalid_policy_defaults_to_ng_only(self) -> None:
        self.assertEqual(
            normalize_capture_retention_policy("unexpected"),
            RUNTIME_CAPTURE_POLICY_NG_ONLY,
        )


if __name__ == "__main__":
    unittest.main()
