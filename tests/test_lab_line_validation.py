from __future__ import annotations

import copy
import json
import unittest

from tools.lab_line_validation import POINT_NAMES, evaluate_methods


def _linear_records() -> list[dict]:
    return [
        {"image": f"{product + 1}.png", "name": name,
         "legacy_px": 50.0 + 2.0 * product + position,
         "auto_px": (50.0 + 2.0 * product + position) * 1.25,
         "lab_mm": (50.0 + 2.0 * product + position) * 0.01,
         "status": "ok"}
        for product in range(6) for position, name in enumerate(POINT_NAMES)
    ]


class LabLineValidationTests(unittest.TestCase):
    def test_linear_cases_fit_both_methods_without_residuals(self) -> None:
        result = evaluate_methods(_linear_records())
        for method, expected_scale in (("legacy", 0.01), ("auto", 0.008)):
            scopes = result["methods"][method]
            self.assertAlmostEqual(scopes["same_batch"]["scale"], expected_scale)
            for scope in (scopes["same_batch"], scopes["leave_one_product_out"]):
                self.assertEqual(scope["n"], 84)
                self.assertTrue(scope["complete"])
                self.assertLess(scope["max_abs"], 1e-12)
            self.assertEqual(len(scopes["leave_one_product_out"]["folds"]), 6)
        json.dumps(result, allow_nan=False)

    def test_held_out_values_cannot_change_training_parameters(self) -> None:
        rows = _linear_records()
        before = evaluate_methods(rows)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["image"] == "3.png":
                row["legacy_px"] *= 7.0
                row["auto_px"] *= 13.0
                row["lab_mm"] += 9.0
        after = evaluate_methods(changed)
        for method in ("legacy", "auto"):
            old_fold = before["methods"][method]["leave_one_product_out"]["folds"][2]
            new_fold = after["methods"][method]["leave_one_product_out"]["folds"][2]
            self.assertEqual(old_fold["scale"], new_fold["scale"])
            self.assertEqual(old_fold["offsets"], new_fold["offsets"])
            self.assertNotIn("3.png", new_fold["training_images"])
            self.assertEqual(new_fold["training_product_count"], 5)
            self.assertEqual(set(new_fold["training_count_by_point"].values()), {5})
            self.assertGreater(new_fold["max_abs"], 1.0)

    def test_review_values_are_included_and_missing_points_stay_visible(self) -> None:
        rows = _linear_records()
        rows[0]["status"] = "review"
        for row in rows:
            if row["name"] == "L7":
                row["auto_px"] = None
                row["status"] = "failed"
        result = evaluate_methods(rows)
        self.assertEqual(result["review_count"], 1)
        self.assertEqual(result["failed_count"], 6)
        legacy = result["methods"]["legacy"]["leave_one_product_out"]
        self.assertEqual(legacy["n"], 84)
        self.assertEqual(legacy["evaluated_review_count"], 1)
        self.assertEqual(legacy["evaluated_failed_count"], 6)
        auto = result["methods"]["auto"]["leave_one_product_out"]
        self.assertEqual(auto["n"], 0)
        self.assertEqual(auto["failed_training_fold_count"], 6)
        self.assertIsNone(auto["mae"])
        self.assertEqual(len(auto["predictions"]), 84)
        self.assertEqual(auto["folds"][0]["missing_training_points"], ["L7"])
        json.dumps(result, allow_nan=False)

    def test_one_missing_measurement_is_not_silently_reported_as_complete(self) -> None:
        rows = _linear_records()
        rows[0]["auto_px"] = float("nan")
        rows[0]["status"] = "failed"
        scope = evaluate_methods(rows)["methods"]["auto"]["leave_one_product_out"]
        self.assertEqual(scope["n"], 83)
        self.assertEqual(scope["evaluation_status"], "partial")
        self.assertFalse(scope["complete"])
        self.assertEqual(scope["per_point"]["U1"]["n"], 5)
        self.assertIsNone(scope["predictions"][0]["predicted_mm"])
        self.assertEqual(scope["folds"][1]["training_count_by_point"]["U1"], 4)
        json.dumps(scope, allow_nan=False)

    def test_duplicate_pairing_is_rejected(self) -> None:
        rows = _linear_records()
        with self.assertRaisesRegex(ValueError, "Duplicate product/point"):
            evaluate_methods(rows + [dict(rows[0])])


if __name__ == "__main__":
    unittest.main()
