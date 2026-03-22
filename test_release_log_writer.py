from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from services.record_writer import CsvReleaseLogWriter, ReleaseLogService


class ReleaseLogWriterTest(unittest.TestCase):
    def test_write_event_creates_daily_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ReleaseLogService(CsvReleaseLogWriter(tmpdir))
            file_path = service.write_event(
                product_name="demo_product",
                recipe_name="demo_recipe.json",
                event_type="release_request",
                result="granted",
                message="密码正确，已放行一次",
                runtime_state="ReleasedPendingConsume",
                extra_fields={"operator": "runtime_ui"},
            )

            self.assertTrue(file_path.exists())
            with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["product_name"], "demo_product")
            self.assertEqual(rows[0]["event_type"], "release_request")
            self.assertEqual(rows[0]["result"], "granted")
            self.assertEqual(rows[0]["runtime_state"], "ReleasedPendingConsume")
            self.assertEqual(rows[0]["operator"], "runtime_ui")


if __name__ == "__main__":
    unittest.main()
