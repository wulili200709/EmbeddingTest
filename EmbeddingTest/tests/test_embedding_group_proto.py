from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


import algorithms.embedding as embedding


class EmbeddingGroupProtoTest(unittest.TestCase):
    def test_group_training_collapses_bank_to_single_proto(self) -> None:
        sample_vectors = [
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.8, 0.2], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([0.2, 0.8], dtype=np.float32),
        ]

        with mock.patch.object(embedding, "load_backbone", return_value=(object(), 2)), mock.patch.object(
            embedding,
            "embed_one",
            side_effect=sample_vectors,
        ):
            model = embedding.train_register_model_from_samples(
                ok_samples=[("ok_1.png", "roi1"), ("ok_2.png", "roi2")],
                ng_samples=[("ng_1.png", "roi1"), ("ng_2.png", "roi2")],
                backbone="efficientnet_b0",
                score_mode="topk",
                margin=0.02,
                topk=3,
                label_name="pusher",
                label_names=["pusher"],
                collapse_to_proto=True,
                device="cpu",
            )

        self.assertTrue(model.grouped_proto_only)
        self.assertEqual(model.score_mode, "proto")
        self.assertEqual(model.ok_bank.shape, (1, 2))
        self.assertEqual(model.ng_bank.shape, (1, 2))
        self.assertEqual(model.ok_analysis_bank.shape, (2, 2))
        self.assertEqual(model.ng_analysis_bank.shape, (2, 2))
        self.assertEqual(model.ok_analysis_names, ["ok_1.png [roi1]", "ok_2.png [roi2]"])
        self.assertEqual(model.ng_analysis_names, ["ng_1.png [roi1]", "ng_2.png [roi2]"])
        np.testing.assert_allclose(model.ok_bank, model.ok_proto, atol=1e-6)
        np.testing.assert_allclose(model.ng_bank, model.ng_proto, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
