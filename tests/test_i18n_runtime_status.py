from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from ui.i18n import language_code, set_language, tr, tr_status_text


class RuntimeStatusI18nTest(unittest.TestCase):
    def test_runtime_status_text_follows_language(self) -> None:
        previous = language_code()
        try:
            set_language("en_US", persist=False)
            self.assertEqual(tr_status_text("未检测"), "Not Tested")
            self.assertEqual(tr_status_text("相机未接入"), "Not connected")
            self.assertEqual(tr_status_text("已禁用"), "Disabled")

            set_language("zh_CN", persist=False)
            self.assertEqual(tr_status_text("未检测"), "未检测")
            self.assertEqual(tr_status_text("相机未接入"), "未连接")
            self.assertEqual(tr_status_text("已禁用"), "已禁用")
        finally:
            set_language(previous, persist=False)

    def test_clear_current_test_list_button_text_exists(self) -> None:
        previous = language_code()
        try:
            set_language("en_US", persist=False)
            self.assertEqual(tr("debug.clear_current_test_list"), "Clear Current Test List")

            set_language("zh_CN", persist=False)
            self.assertEqual(tr("debug.clear_current_test_list"), "清空当前测试列表")
        finally:
            set_language(previous, persist=False)


if __name__ == "__main__":
    unittest.main()
