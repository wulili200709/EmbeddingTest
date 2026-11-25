from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from domain.inspection_models import InspectionItemResult, RuntimeInspectionResult
from application.runtime.execution import _write_runtime_record_sync
from infrastructure.audit_store import AuditStore, PermissionService, RuntimeRecordStore
from ui.shell.audit_dialogs import RuntimeRecordsDialog, UserPermissionDialog


class RuntimeRecordStoreTest(unittest.TestCase):
    def _result(
        self,
        *,
        product_name: str,
        final_result: str,
        item_results: list[InspectionItemResult],
    ) -> RuntimeInspectionResult:
        return RuntimeInspectionResult(
            task_id=f"runtime_{product_name}",
            product_name=product_name,
            recipe_name=f"{product_name}.json",
            final_result=final_result,
            duration_ms=125,
            item_results=item_results,
        )

    def test_stores_variable_roi_counts_and_exports_current_product_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeRecordStore(Path(tmpdir) / "audit.db")
            store.write_runtime_result(
                self._result(
                    product_name="A",
                    final_result="NG",
                    item_results=[
                        InspectionItemResult(
                            item_id="roi1",
                            display_name="密封圈1",
                            camera_id="cam1",
                            roi_label="roi1",
                            result="OK",
                        ),
                        InspectionItemResult(
                            item_id="roi2",
                            display_name="缺口",
                            camera_id="cam1",
                            roi_label="roi2",
                            result="NG",
                        ),
                    ],
                )
            )
            store.write_runtime_result(
                self._result(
                    product_name="B",
                    final_result="OK",
                    item_results=[
                        InspectionItemResult(
                            item_id="roi1",
                            display_name="外观",
                            camera_id="cam1",
                            roi_label="roi1",
                            result="OK",
                        ),
                        InspectionItemResult(
                            item_id="roi1",
                            display_name="定位",
                            camera_id="cam2",
                            roi_label="roi1",
                            result="OK",
                        ),
                        InspectionItemResult(
                            item_id="roi2",
                            display_name="高度",
                            camera_id="cam2",
                            roi_label="roi2",
                            result="OK",
                            value=12.5,
                            unit="mm",
                        ),
                    ],
                )
            )

            all_runs = store.query_runs(limit=10)
            by_product = {row["product_name"]: row for row in all_runs}
            self.assertEqual(len(store.roi_results_for_run(by_product["A"]["id"])), 2)
            self.assertEqual(len(store.roi_results_for_run(by_product["B"]["id"])), 3)

            product_a_runs = store.query_runs(product_name="A")
            details_by_run = store.roi_results_for_run_ids([row["id"] for row in product_a_runs])
            export_path = Path(tmpdir) / "product_a.csv"
            store.export_runs_csv(product_a_runs, details_by_run, export_path)

            with export_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(
                list(rows[0].keys()),
                [
                    "record_time",
                    "product_name",
                    "final_result",
                    "cam1.密封圈1",
                    "cam1.缺口",
                ],
            )
            self.assertEqual(rows[0]["product_name"], "A")
            self.assertEqual(rows[0]["cam1.密封圈1"], "OK")
            self.assertEqual(rows[0]["cam1.缺口"], "NG")
            self.assertNotIn("recipe_name", rows[0])
            self.assertNotIn("duration_ms", rows[0])
            self.assertNotIn("error_message", rows[0])

            all_details_by_run = store.roi_results_for_run_ids([row["id"] for row in all_runs])
            all_export_path = Path(tmpdir) / "all_products.csv"
            store.export_runs_csv(all_runs, all_details_by_run, all_export_path)
            with all_export_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                all_rows = list(csv.DictReader(csv_file))

            all_rows_by_product = {row["product_name"]: row for row in all_rows}
            self.assertIn("cam1.密封圈1", all_rows_by_product["A"])
            self.assertIn("cam2.高度", all_rows_by_product["A"])
            self.assertEqual(all_rows_by_product["A"]["cam2.高度"], "")
            self.assertEqual(all_rows_by_product["B"]["cam1.缺口"], "")
            self.assertEqual(all_rows_by_product["B"]["cam2.高度"], "OK")

    def test_runtime_record_permissions_can_be_granted_per_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AuditStore(Path(tmpdir) / "audit.db")
            store.set_role_permissions("admin", {"runtime_records.view"})
            store.create_user("record_viewer", "test-password", "admin")
            service = PermissionService(store)

            self.assertIsNotNone(service.login("record_viewer", "test-password"))
            self.assertTrue(service.has("runtime_records.view"))
            self.assertFalse(service.has("runtime_records.export"))

    def test_runtime_persistence_writes_database_without_csv_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeRecordStore(Path(tmpdir) / "audit.db")
            emitted: list[str] = []
            runtime = SimpleNamespace(
                _runtime_results_store=store,
                logAppended=SimpleNamespace(emit=emitted.append),
            )
            result = self._result(
                product_name="persisted",
                final_result="OK",
                item_results=[
                    InspectionItemResult(
                        item_id="roi1",
                        display_name="外观",
                        camera_id="cam1",
                        roi_label="roi1",
                        result="OK",
                    )
                ],
            )

            _write_runtime_record_sync(runtime, None, result, lock_on_ng=False)

            rows = store.query_runs(product_name="persisted")
            self.assertEqual(len(rows), 1)
            self.assertEqual(store.roi_results_for_run(rows[0]["id"])[0]["display_name"], "外观")
            self.assertEqual(emitted, [])


class RuntimeRecordPermissionDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    @staticmethod
    def _permission_item(dialog: UserPermissionDialog, key: str) -> QtWidgets.QListWidgetItem:
        for index in range(dialog.list_permissions.count()):
            item = dialog.list_permissions.item(index)
            if str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "") == key:
                return item
        raise AssertionError(f"permission item not found: {key}")

    def test_only_super_admin_can_change_runtime_record_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AuditStore(Path(tmpdir) / "audit.db")
            restricted = UserPermissionDialog(
                None,
                store,
                can_manage_runtime_records=False,
            )
            role_index = restricted.cmb_role.findData("admin")
            restricted.cmb_role.setCurrentIndex(role_index)
            self.app.processEvents()
            restricted_item = self._permission_item(restricted, "runtime_records.view")
            self.assertFalse(bool(restricted_item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled))
            restricted.close()

            elevated = UserPermissionDialog(
                None,
                store,
                can_manage_runtime_records=True,
            )
            role_index = elevated.cmb_role.findData("admin")
            elevated.cmb_role.setCurrentIndex(role_index)
            self.app.processEvents()
            elevated_item = self._permission_item(elevated, "runtime_records.view")
            self.assertTrue(bool(elevated_item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled))
            elevated.close()

    def test_runtime_records_dialog_displays_run_and_roi_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeRecordStore(Path(tmpdir) / "audit.db")
            store.write_runtime_result(
                RuntimeInspectionResult(
                    task_id="runtime_dialog",
                    product_name="demo",
                    final_result="NG",
                    item_results=[
                        InspectionItemResult(
                            item_id="roi1",
                            display_name="密封圈",
                            camera_id="cam1",
                            roi_label="roi1",
                            result="NG",
                        )
                    ],
                )
            )
            dialog = RuntimeRecordsDialog(None, store, can_export=True)
            self.assertEqual(dialog.table_runs.rowCount(), 1)
            self.assertEqual(dialog.table_roi_results.rowCount(), 1)
            self.assertEqual(dialog.table_runs.item(0, 1).text(), "demo")
            self.assertEqual(dialog.table_roi_results.item(0, 1).text(), "密封圈")
            dialog.close()


if __name__ == "__main__":
    unittest.main()
