from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from application.auto_roi_service import build_auto_roi_spec


class AutoRoiServiceTests(unittest.TestCase):
    def test_shared_spec_builder_validates_missing_ncc_model(self) -> None:
        result = build_auto_roi_spec(
            method="ncc",
            ref_image="",
            product_dir="product",
            camera_role="cam1",
            labels=["roi"],
            ncc_model_path="missing.ncc",
        )

        self.assertIsNone(result.spec)
        self.assertEqual(result.issue.message_key, "template.no_model")

    def test_shared_spec_builder_creates_worker_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.ncc"
            model_path.touch()
            result = build_auto_roi_spec(
                method="ncc",
                ref_image="",
                product_dir=temp_dir,
                camera_role="cam2",
                labels=["part", "part"],
                ncc_model_path=str(model_path),
            )

            self.assertIsNotNone(result.spec)
            payload = result.spec.payload(["one.bmp"], only_missing=True, pre_resolved=False)
            self.assertEqual(payload["camera_role"], "cam2")
            self.assertEqual(payload["labels"], ["part", "part"])
            self.assertTrue(payload["only_missing"])


if __name__ == "__main__":
    unittest.main()
