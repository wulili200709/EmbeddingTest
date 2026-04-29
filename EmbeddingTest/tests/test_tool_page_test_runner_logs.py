from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ui.debug.tool_page.test_runner import _append_test_log


class _DummyMode:
    def currentText(self) -> str:
        return "proto"


class _DummyNumber:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class TestRunnerLogTest(unittest.TestCase):
    def test_append_test_log_normalizes_legacy_learning_algorithm_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = str(Path(tmpdir) / "20260428.csv")
            tool_page = SimpleNamespace(
                session=SimpleNamespace(current_product="Demo"),
                _daily_test_log_path=lambda: csv_path,
                current_algorithm=lambda: "b0",
                cmb_mode=_DummyMode(),
                spin_margin=_DummyNumber(0.02),
                spin_topk=_DummyNumber(3),
            )

            _append_test_log(
                tool_page,
                {
                    "algorithm": "mobilenet_v3_large",
                    "tool_name": "ROI1",
                    "camera_id": "cam1",
                    "roi_label": "roi1",
                    "file_name": "sample.png",
                    "pred": "OK",
                },
            )

            with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["algorithm"], "b2")


if __name__ == "__main__":
    unittest.main()
