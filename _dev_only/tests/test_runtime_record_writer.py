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
    InspectionItemResult,
    RuntimeInspectionResult,
)
from services.record_writer import CsvRecordWriter, TestRecordService


class RuntimeRecordWriterTest(unittest.TestCase):
    def test_write_product_result_uses_camera_qualified_binary_roi_columns(self) -> None:
        runtime_result = RuntimeInspectionResult(
            task_id="runtime_001",
            product_name="demo_product",
            recipe_name="demo_recipe.json",
            final_result="NG",
            duration_ms=123,
            item_results=[
                InspectionItemResult(
                    item_id="roi1",
                    display_name="ROI 1",
                    camera_id="cam2",
                    roi_label="roi1",
                    result="NG",
                ),
                InspectionItemResult(
                    item_id="roi1",
                    display_name="ROI 1",
                    camera_id="cam1",
                    roi_label="roi1",
                    result=" ok ",
                ),
                InspectionItemResult(
                    item_id="roi2",
                    display_name="ROI 2",
                    camera_id="cam1",
                    roi_label="roi2",
                    result="PASS",
                ),
                InspectionItemResult(
                    item_id="roi3",
                    display_name="ROI 3",
                    camera_id="cam1",
                    roi_label="roi3",
                    result="MEASURED",
                ),
                InspectionItemResult(
                    item_id="roi4",
                    display_name="ROI 4",
                    camera_id="cam1",
                    roi_label="roi4",
                    result="PENDING",
                ),
                InspectionItemResult(
                    item_id="roi2",
                    display_name="ROI 2",
                    camera_id="cam2",
                    roi_label="roi2",
                    enabled=False,
                    result="DISABLED",
                ),
                InspectionItemResult(
                    item_id="roi5_pending",
                    display_name="ROI 5 pending",
                    camera_id="cam1",
                    roi_label="roi5",
                    result="PENDING",
                ),
                InspectionItemResult(
                    item_id="roi5_pass",
                    display_name="ROI 5 pass",
                    camera_id="cam1",
                    roi_label="roi5",
                    result="PASS",
                ),
                InspectionItemResult(
                    item_id="roi3_ok",
                    display_name="ROI 3 OK",
                    camera_id="cam2",
                    roi_label="roi3",
                    result="OK",
                ),
                InspectionItemResult(
                    item_id="roi3_ng",
                    display_name="ROI 3 NG",
                    camera_id="cam2",
                    roi_label="roi3",
                    result="NG",
                ),
                InspectionItemResult(
                    item_id="roi4",
                    display_name="ROI 4",
                    camera_id="cam2",
                    roi_label="roi4",
                    result="INACTIVE",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            service = TestRecordService(CsvRecordWriter(tmpdir))
            file_path = service.write_product_result(
                product_name=runtime_result.product_name,
                recipe_name=runtime_result.recipe_name,
                final_result=runtime_result.final_result,
                camera1_result="OK",
                camera2_result="NG",
                duration_ms=runtime_result.duration_ms,
                error_message="must not be serialized",
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
                    "cam1.ROI 1",
                    "cam1.ROI 2",
                    "cam1.ROI 3",
                    "cam1.ROI 4",
                    "cam1.ROI 5 pending",
                    "cam1.ROI 5 pass",
                    "cam2.ROI 1",
                    "cam2.ROI 2",
                    "cam2.ROI 3 OK",
                    "cam2.ROI 3 NG",
                    "cam2.ROI 4",
                ],
            )
            self.assertEqual(row["cam1.ROI 1"], "OK")
            self.assertEqual(row["cam2.ROI 1"], "NG")
            self.assertEqual(row["cam1.ROI 2"], "OK")
            self.assertEqual(row["cam1.ROI 3"], "")
            self.assertEqual(row["cam1.ROI 4"], "")
            self.assertEqual(row["cam1.ROI 5 pending"], "")
            self.assertEqual(row["cam1.ROI 5 pass"], "OK")
            self.assertEqual(row["cam2.ROI 2"], "")
            self.assertEqual(row["cam2.ROI 3 OK"], "OK")
            self.assertEqual(row["cam2.ROI 3 NG"], "NG")
            self.assertEqual(row["cam2.ROI 4"], "")
            self.assertLessEqual(
                {row[key] for key in row if key not in {"record_time", "product_name"}},
                {"OK", "NG", ""},
            )

    def test_append_record_only_grows_dynamic_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = TestRecordService(CsvRecordWriter(tmpdir))
            file_path = service.write_product_result(
                product_name="demo_product",
                final_result="OK",
                extra_fields={"cam1.roi1": "OK"},
            )
            service.write_product_result(
                product_name="demo_product",
                final_result="NG",
                extra_fields={"cam2.roi1": "NG"},
            )
            service.write_product_result(
                product_name="demo_product",
                final_result="OK",
                extra_fields={"cam1.roi2": "OK"},
            )

            with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(
                list(rows[0].keys()),
                [
                    "record_time",
                    "product_name",
                    "cam1.roi1",
                    "cam2.roi1",
                    "cam1.roi2",
                ],
            )
            self.assertEqual(rows[0]["cam1.roi1"], "OK")
            self.assertEqual(rows[0]["cam2.roi1"], "")
            self.assertEqual(rows[1]["cam1.roi1"], "")
            self.assertEqual(rows[1]["cam2.roi1"], "NG")
            self.assertEqual(rows[2]["cam1.roi2"], "OK")

    def test_append_record_preserves_legacy_columns_while_adding_roi(self) -> None:
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
                        "task_id",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "record_time": f"{datetime.now().strftime('%Y-%m-%d')} 10:00:00",
                        "product_name": "legacy_product",
                        "final_result": "OK",
                        "camera1_result": "OK",
                        "task_id": "old_task",
                    }
                )

            writer = CsvRecordWriter(base_dir)
            record = TestRecordService(writer).write_product_result(
                product_name="demo_product",
                final_result="NG",
                camera1_result="NG",
                extra_fields={"cam1.roi1": "NG"},
            )

            self.assertEqual(record, file_path)
            with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 2)
            self.assertEqual(
                list(rows[0].keys()),
                [
                    "record_time",
                    "product_name",
                    "cam1.roi1",
                    "final_result",
                    "camera1_result",
                    "task_id",
                ],
            )
            self.assertEqual(rows[0]["product_name"], "legacy_product")
            self.assertEqual(rows[0]["final_result"], "OK")
            self.assertEqual(rows[0]["task_id"], "old_task")
            self.assertEqual(rows[0]["cam1.roi1"], "")
            self.assertEqual(rows[1]["final_result"], "")
            self.assertEqual(rows[1]["cam1.roi1"], "NG")

    def test_empty_roi_label_uses_display_name_and_nonbinary_status_is_empty(self) -> None:
        runtime_result = RuntimeInspectionResult(
            task_id="runtime_002",
            product_name="demo_product",
            final_result="OK",
            item_results=[
                InspectionItemResult(
                    item_id="line_distance",
                    display_name="卡尺距离测量",
                    camera_id="cam1",
                    roi_label="",
                    algorithm_code="line_distance_ref_normal",
                    result="MEASURED",
                    value=5.724321,
                    unit="mm",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            service = TestRecordService(CsvRecordWriter(tmpdir))
            file_path = service.write_product_result(
                product_name=runtime_result.product_name,
                final_result=runtime_result.final_result,
                camera1_result="OK",
                extra_fields=runtime_result.to_record_extra_fields(),
            )

            with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["cam1.卡尺距离测量"], "")
            self.assertNotIn("cam1.line_distance", row)
            self.assertNotIn("item_01_distance", row)


if __name__ == "__main__":
    unittest.main()
