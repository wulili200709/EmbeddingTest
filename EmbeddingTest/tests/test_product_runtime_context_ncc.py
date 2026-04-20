from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


import application.runtime_context as runtime_context_module
from application.runtime_context import ProductRuntimeContext


class _FakeAlgo:
    def load_params(self, _path: str) -> None:
        pass


class _FakeSession:
    product_dir = "demo-product"
    product_params_path = "product_params.json"
    inspection_items_path = "inspection_items.json"

    def load_session(self):
        return SimpleNamespace(loc_method="ncc", ref_image="")


class ProductRuntimeContextNccTest(unittest.TestCase):
    def test_reload_syncs_inspection_items_from_ncc_model_specs(self) -> None:
        context = ProductRuntimeContext.__new__(ProductRuntimeContext)
        context.session = _FakeSession()
        context.algo = _FakeAlgo()
        saved_payloads: list[list[object]] = []

        with (
            mock.patch.object(runtime_context_module, "load_inspection_items", return_value=[]),
            mock.patch.object(runtime_context_module, "save_inspection_items", side_effect=lambda items, _path: saved_payloads.append(list(items))),
            mock.patch.object(runtime_context_module.ncc_locator, "model_is_ready", side_effect=lambda _product_dir, role: role == "cam1"),
            mock.patch.object(
                runtime_context_module.ncc_locator,
                "inspection_item_specs_for_product",
                return_value=[
                    {"roi_label": "roi1", "display_name": "Hole"},
                    {"roi_label": "roi2", "display_name": "Pusher"},
                ],
            ),
        ):
            context.reload()

        self.assertEqual(context.loc_method, "ncc")
        self.assertEqual(
            [(item.camera_id, item.roi_label, item.display_name) for item in context.inspection_items],
            [("cam1", "roi1", "Hole"), ("cam1", "roi2", "Pusher")],
        )
        self.assertEqual(len(saved_payloads), 1)


if __name__ == "__main__":
    unittest.main()
