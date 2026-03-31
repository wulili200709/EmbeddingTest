from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from domain.inspection_models import (
    CameraRuntimeResult,
    InspectionItemResult,
    RuntimeInspectionResult,
)
from services.record_writer import CsvRecordWriter, TestRecordService


class RuntimeRecordWriterTest(unittest.TestCase):
    def test_write_product_result_includes_only_expected_item_level_columns(self) -> None:
        runtime_result = RuntimeInspectionResult(
            task_id="runtime_001",
            product_name="demo_product",
            recipe_name="demo_recipe.json",
            final_result="NG",
            duration_ms=123,
            camera_results={
                "cam1": CameraRuntimeResult(
                    camera_id="cam1",
                    result="NG",
                    detail="cam1 detail",
                    image_path="capture/cam1.png",
                ),
            },
            item_results=[
                InspectionItemResult(
                    item_id="roi1",
                    display_name="ROI 1",
                    camera_id="cam1",
                    roi_label="roi1",
                    result="NG",
                    detail="diff=0.12",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            service = TestRecordService(CsvRecordWriter(tmpdir))
            file_path = service.write_product_result(
                product_name=runtime_result.product_name,
                recipe_name=runtime_result.recipe_name,
                final_result=runtime_result.final_result,
                camera1_result="NG",
                duration_ms=runtime_result.duration_ms,
                extra_fields=runtime_result.to_record_extra_fields(),
            )

            with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(
                list(row.keys()),
                [
                    "record_time",
                    "product_name",
                    "final_result",
                    "camera1_result",
                    "camera2_result",
                    "error_message",
                    "item_01_enabled",
                    "item_01_name",
                    "item_01_result",
                    "item_01_roi_label",
                ],
            )
            self.assertEqual(row["camera1_result"], "NG")
            self.assertEqual(row["item_01_enabled"], "True")
            self.assertEqual(row["item_01_name"], "ROI 1")
            self.assertEqual(row["item_01_result"], "NG")
            self.assertEqual(row["item_01_roi_label"], "roi1")

    def test_append_record_rewrites_legacy_header_to_current_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            file_path = base_dir / f"{datetime.now().strftime('%Y-%m-%d')}.csv"
            with file_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "record_time",
                        "product_name",
                        "final_result",
                        "camera1_result",
                        "camera2_result",
                        "error_message",
                        "task_id",
                        "item_01_id",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "record_time": f"{datetime.now().strftime('%Y-%m-%d')} 10:00:00",
                        "product_name": "legacy_product",
                        "final_result": "OK",
                        "camera1_result": "OK",
                        "camera2_result": "",
                        "error_message": "",
                        "task_id": "old_task",
                        "item_01_id": "old_roi",
                    }
                )

            writer = CsvRecordWriter(base_dir)
            record = TestRecordService(writer).write_product_result(
                product_name="demo_product",
                final_result="NG",
                camera1_result="NG",
                extra_fields={
                    "item_01_enabled": True,
                    "item_01_name": "ROI 1",
                    "item_01_result": "NG",
                    "item_01_roi_label": "roi1",
                },
            )

            self.assertEqual(record, file_path)
            with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 2)
            self.assertNotIn("task_id", rows[0])
            self.assertNotIn("item_01_id", rows[0])
            self.assertEqual(rows[0]["product_name"], "legacy_product")
            self.assertEqual(rows[1]["item_01_name"], "ROI 1")
            self.assertEqual(rows[1]["item_01_result"], "NG")


if __name__ == "__main__":
    unittest.main()
