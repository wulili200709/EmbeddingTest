from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from devices import IoMapping


class DefaultIoMappingTest(unittest.TestCase):
    def test_default_mapping_routes_button_red_to_do9(self) -> None:
        mapping_path = PROJECT_DIR / "config" / "defaults" / "io_mapping.json"

        mapping = IoMapping.from_json_file(mapping_path)

        cfg = mapping.get_output("button_red")
        self.assertEqual(cfg.channel, 9)
        self.assertFalse(cfg.active_high)

    def test_default_mapping_exposes_outputs_through_do15(self) -> None:
        mapping_path = PROJECT_DIR / "config" / "defaults" / "io_mapping.json"

        mapping = IoMapping.from_json_file(mapping_path)

        expected_reserved = {
            "reserved_out_3": 9,
            "reserved_out_4": 10,
            "reserved_out_5": 11,
            "reserved_out_6": 12,
            "reserved_out_7": 13,
            "reserved_out_8": 14,
            "reserved_out_9": 15,
        }

        for name, channel in expected_reserved.items():
            cfg = mapping.get_output(name)
            self.assertEqual(cfg.channel, channel, name)
            self.assertFalse(cfg.active_high, name)


if __name__ == "__main__":
    unittest.main()
