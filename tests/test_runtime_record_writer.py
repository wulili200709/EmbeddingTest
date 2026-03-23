from __future__ import annotations

import csv
import sys
import tempfile
import unittest
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
    def test_write_product_result_includes_item_level_columns(self) -> None:
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
            self.assertEqual(row["task_id"], "runtime_001")
            self.assertEqual(row["item_count"], "1")
            self.assertEqual(row["cam1_result"], "NG")
            self.assertEqual(row["item_01_id"], "roi1")
            self.assertEqual(row["item_01_result"], "NG")
            self.assertEqual(row["item_01_detail"], "diff=0.12")


if __name__ == "__main__":
    unittest.main()
