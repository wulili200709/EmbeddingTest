from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from algorithms.embedding import predict_one_with_model, train_register_model_from_samples
from algorithms.traditional import train_threshold_model_from_samples


class SharedRoiSampleModelTests(unittest.TestCase):
    def test_embedding_trains_each_crop_as_a_sample_and_keeps_roi_labels_separate(self):
        # The same image contains an OK ROI and an NG ROI. Its image path
        # cannot be used as the class label for both crops.
        with patch("algorithms.embedding._embed_many_cached", side_effect=[
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.9, 0.1, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        ]) as embed:
            model = train_register_model_from_samples(
                [("ok.png", "roi1"), ("mixed.png", "roi2")],
                [("mixed.png", "roi1")],
                label_names=["roi1", "roi2"], device="cpu", feat_net=object(),
            )
        self.assertEqual([(call.args[0], call.args[2]) for call in embed.call_args_list], [
            ("ok.png", ["roi1"]), ("mixed.png", ["roi2"]), ("mixed.png", ["roi1"]),
        ])
        self.assertEqual(model.ok_bank.shape, (2, 3))
        self.assertEqual(model.ng_bank.shape, (1, 3))
        np.testing.assert_allclose(model.ng_bank[0], [0.0, 0.0, 1.0])
        self.assertTrue(model.is_ready())
        self.assertEqual(predict_one_with_model(np.array([1.0, 0.0, 0.0], dtype=np.float32), model)[0], "OK")
        self.assertEqual(predict_one_with_model(np.array([0.0, 0.0, 1.0], dtype=np.float32), model)[0], "NG")

    def test_embedding_requires_both_sample_classes_after_normalization(self):
        with self.assertRaisesRegex(RuntimeError, "Both OK and NG"):
            train_register_model_from_samples([("ok.png", "roi1")], [], feat_net=object())

    def test_traditional_threshold_uses_the_label_of_each_crop(self):
        with patch("algorithms.traditional.compute_roi_metrics", side_effect=[
            {"meanintensity": 100.0}, {"meanintensity": 110.0}, {"meanintensity": 10.0},
        ]) as metrics:
            model, rows = train_threshold_model_from_samples(
                [("ok.png", "roi1"), ("mixed.png", "roi2")],
                [("mixed.png", "roi1")], "meanintensity",
            )
        self.assertEqual([(call.args[0], call.kwargs["preferred_label"]) for call in metrics.call_args_list], [
            ("ok.png", "roi1"), ("mixed.png", "roi2"), ("mixed.png", "roi1"),
        ])
        self.assertEqual([row["gt"] for row in rows], ["OK", "OK", "NG"])
        self.assertEqual(model.predict(100.0)[0], "OK")
        self.assertEqual(model.predict(10.0)[0], "NG")
        self.assertEqual(model.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
