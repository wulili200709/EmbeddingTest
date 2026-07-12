from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PySide6 import QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


from ui.runtime.runtime_mode_pyside6 import RuntimeModePage


def _write_runtime_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record_time", "product_name"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class RuntimeModeDailyCountsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_switching_products_restores_same_day_counts_for_each_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = Path(tmpdir) / "2026-04-08.csv"
            _write_runtime_csv(
                record_path,
                [
                    {
                        "record_time": "2026-04-08 08:00:00",
                        "product_name": "RED",
                        "final_result": "OK",
                        "camera1_result": "OK",
                        "camera2_result": "",
                        "error_message": "",
                    },
                    {
                        "record_time": "2026-04-08 08:10:00",
                        "product_name": "RED",
                        "final_result": "OK",
                        "camera1_result": "OK",
                        "camera2_result": "",
                        "error_message": "",
                    },
                    {
                        "record_time": "2026-04-08 09:00:00",
                        "product_name": "OG",
                        "final_result": "OK",
                        "camera1_result": "OK",
                        "camera2_result": "",
                        "error_message": "",
                    },
                    {
                        "record_time": "2026-04-08 09:10:00",
                        "product_name": "OG",
                        "final_result": "NG",
                        "camera1_result": "NG",
                        "camera2_result": "",
                        "error_message": "",
                    },
                ],
            )

            page = RuntimeModePage()
            page.set_record_path(str(record_path))

            page.set_current_product("RED")
            self.assertEqual(page.lbl_ok_count.text(), "OK: 2")
            self.assertEqual(page.lbl_ng_count.text(), "NG: 0")

            page.set_current_product("OG")
            self.assertEqual(page.lbl_ok_count.text(), "OK: 1")
            self.assertEqual(page.lbl_ng_count.text(), "NG: 1")

            page.set_current_product("RED")
            self.assertEqual(page.lbl_ok_count.text(), "OK: 2")
            self.assertEqual(page.lbl_ng_count.text(), "NG: 0")

    def test_daily_counts_support_legacy_and_roi_record_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = Path(tmpdir) / "2026-04-08.csv"
            _write_runtime_csv(
                record_path,
                [
                    {
                        "record_time": "2026-04-08 08:00:00",
                        "product_name": "DEMO",
                        "final_result": "OK",
                    },
                    {
                        "record_time": "2026-04-08 08:10:00",
                        "product_name": "DEMO",
                        "cam1.roi1": "OK",
                        "cam2.roi1": "",
                    },
                    {
                        "record_time": "2026-04-08 08:20:00",
                        "product_name": "DEMO",
                        "cam1.roi1": "OK",
                        "cam2.roi1": "NG",
                    },
                    {
                        "record_time": "2026-04-08 08:30:00",
                        "product_name": "DEMO",
                        "cam1.roi1": "",
                        "cam2.roi1": "",
                    },
                    {
                        "record_time": "2026-04-08 08:40:00",
                        "product_name": "OTHER",
                        "cam1.roi1": "NG",
                    },
                ],
            )

            page = RuntimeModePage()
            page.set_record_path(str(record_path))
            page.set_current_product("DEMO")

            self.assertEqual(page.lbl_ok_count.text(), "OK: 2")
            self.assertEqual(page.lbl_ng_count.text(), "NG: 1")

            page.set_current_product("OTHER")
            self.assertEqual(page.lbl_ok_count.text(), "OK: 0")
            self.assertEqual(page.lbl_ng_count.text(), "NG: 1")

            restored_page = RuntimeModePage()
            restored_page.set_current_product("DEMO")
            restored_page.set_record_path(str(record_path))
            self.assertEqual(restored_page.lbl_ok_count.text(), "OK: 2")
            self.assertEqual(restored_page.lbl_ng_count.text(), "NG: 1")

    def test_new_daily_record_file_resets_counts_before_next_result_is_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            day1_path = Path(tmpdir) / "2026-04-08.csv"
            day2_path = Path(tmpdir) / "2026-04-09.csv"
            _write_runtime_csv(
                day1_path,
                [
                    {
                        "record_time": "2026-04-08 08:00:00",
                        "product_name": "RED",
                        "final_result": "OK",
                        "camera1_result": "OK",
                        "camera2_result": "",
                        "error_message": "",
                    }
                ],
            )
            _write_runtime_csv(day2_path, [])

            page = RuntimeModePage()
            page.set_current_product("RED")
            page.set_record_path(str(day1_path))
            self.assertEqual(page.lbl_ok_count.text(), "OK: 1")
            self.assertEqual(page.lbl_ng_count.text(), "NG: 0")

            page.set_record_path(str(day2_path))
            self.assertEqual(page.lbl_ok_count.text(), "OK: 0")
            self.assertEqual(page.lbl_ng_count.text(), "NG: 0")

            page.set_final_result("NG", "first result in new day")
            self.assertEqual(page.lbl_ok_count.text(), "OK: 0")
            self.assertEqual(page.lbl_ng_count.text(), "NG: 1")


if __name__ == "__main__":
    unittest.main()
