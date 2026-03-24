import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from domain.inspection_items import InspectionItem
from domain.inspection_models import InspectionItemResult
from ui.debug.tool_page import page as page_module
from ui.debug.tool_page.page import ToolPage


class _DummyCanvas:
    def __init__(self, path: str) -> None:
        self._path = path
        self.overlays = None

    def image_path(self) -> str | None:
        return self._path

    def set_overlays(self, overlays) -> None:
        self.overlays = overlays


class _DummyLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


class _DummyAlgo:
    def is_learning_tool(self, algorithm_code: str) -> bool:
        return str(algorithm_code) == "shared_backbone_register"

    def current_learning_backbone(self) -> str:
        return "efficientnet_b0"

    def resolve_tool_algorithm(self, algorithm_code: str) -> str:
        if str(algorithm_code) == "shared_backbone_register":
            return "efficientnet_b0"
        return str(algorithm_code)


class _Harness:
    _run_test = ToolPage._run_test
    _test_target_inspection_items = ToolPage._test_target_inspection_items
    _record_roi_result = ToolPage._record_roi_result

    def __init__(self, image_path: str) -> None:
        self.canvas = _DummyCanvas(image_path)
        self.algo = _DummyAlgo()
        self.session = SimpleNamespace(current_product="Demo")
        self.loc_method = "line2dup"
        self.inspection_items = [
            InspectionItem(
                item_id="roi1",
                display_name="ROI1",
                camera_id="cam1",
                roi_label="roi1",
                algorithm_code="shared_backbone_register",
            ),
            InspectionItem(
                item_id="roi2",
                display_name="ROI2",
                camera_id="cam1",
                roi_label="roi2",
                algorithm_code="meanintensity",
            ),
            InspectionItem(
                item_id="roi3",
                display_name="ROI3",
                camera_id="cam2",
                roi_label="roi3",
                algorithm_code="meanintensity",
            ),
        ]
        self._selected_item = self.inspection_items[0]
        self._roi_results_by_image = {}
        self.rows = []
        self.log_rows = []
        self.loaded_paths = []
        self.lbl_status = _DummyLabel()

    def _selected_inspection_item(self):
        return self._selected_item

    def _line2dup_output_labels(self):
        return ["roi1", "roi2", "roi3"]

    def _populate_results_table(self, rows):
        self.rows = list(rows)

    def _append_test_log(self, row):
        self.log_rows.append(dict(row))
        return "demo.csv"

    def _load_canvas_image(self, path: str) -> None:
        self.loaded_paths.append(path)

    def current_algorithm(self) -> str:
        return "efficientnet_b0"


class _FakeInspectionExecutor:
    last_predictor = None
    last_request = None

    def __init__(self, predictor) -> None:
        type(self).last_predictor = predictor

    def execute(self, request):
        type(self).last_request = request
        return SimpleNamespace(
            result="NG",
            match_ms=12.5,
            infer_ms=9.0,
            raw_row={
                "item_rows": [
                    {"pred": "OK", "diff": 0.1},
                    {"pred": "NG", "diff": 0.7},
                ]
            },
            item_results=[
                InspectionItemResult(
                    item_id="roi1",
                    display_name="ROI1",
                    camera_id="cam1",
                    roi_label="roi1",
                    algorithm_code="shared_backbone_register",
                    result="OK",
                ),
                InspectionItemResult(
                    item_id="roi2",
                    display_name="ROI2",
                    camera_id="cam1",
                    roi_label="roi2",
                    algorithm_code="meanintensity",
                    result="NG",
                ),
            ],
        )


class ToolPageRunTestAllItemsTest(unittest.TestCase):
    def test_run_test_reuses_executor_for_all_enabled_items_of_selected_camera(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            image_path = handle.name

        harness = _Harness(image_path)
        original_executor = page_module.InspectionExecutor
        try:
            page_module.InspectionExecutor = _FakeInspectionExecutor
            harness._run_test()
        finally:
            page_module.InspectionExecutor = original_executor
            os.unlink(image_path)

        request = _FakeInspectionExecutor.last_request
        self.assertIsNotNone(request)
        self.assertEqual(request.camera_id, "cam1")
        self.assertEqual([item.item_id for item in request.items], ["roi1", "roi2"])
        self.assertEqual(
            harness._roi_results_by_image[request.image_path],
            {"roi1": "ok", "roi2": "ng"},
        )
        self.assertEqual(len(harness.rows), 2)
        self.assertEqual(harness.rows[0]["file_name"], f"{Path(image_path).name} [ROI1]")
        self.assertEqual(harness.rows[1]["file_name"], f"{Path(image_path).name} [ROI2]")
        self.assertEqual(harness.loaded_paths, [image_path])
        self.assertIn("overall=NG", harness.lbl_status.text)
        self.assertIn("tools=2", harness.lbl_status.text)


if __name__ == "__main__":
    unittest.main()
