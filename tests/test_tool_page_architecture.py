from __future__ import annotations

import unittest

from ui.debug.tool_page import ToolPage
from ui.debug.tool_page.operation_mixins import ToolPageOperationsMixin


class ToolPageArchitectureTests(unittest.TestCase):
    def test_tool_page_operations_are_declared_in_static_mro(self) -> None:
        self.assertIn(ToolPageOperationsMixin, ToolPage.__mro__)
        self.assertTrue(hasattr(ToolPage, "_autogen_roi_for_images"))
        self.assertTrue(hasattr(ToolPage, "_open_debug_io"))
        self.assertTrue(hasattr(ToolPage, "_predict_image"))

    def test_tool_page_ui_builder_is_split_into_stable_sections(self) -> None:
        self.assertTrue(hasattr(ToolPage, "_build_header_ui"))
        self.assertTrue(hasattr(ToolPage, "_build_workspace_ui"))
        self.assertTrue(hasattr(ToolPage, "_build_footer_ui"))
        self.assertTrue(hasattr(ToolPage, "_build_compatibility_ui"))


if __name__ == "__main__":
    unittest.main()
