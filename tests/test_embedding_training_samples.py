from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from algorithms.embedding import train_register_model_from_samples


class EmbeddingTrainingSampleTests(unittest.TestCase):
    def test_single_roi_vectors_are_stacked_as_sample_by_feature_matrices(self) -> None:
        cached_vectors = [
            np.arange(4, dtype=np.float32),
            np.arange(4, dtype=np.float32) + 10,
            np.arange(4, dtype=np.float32) + 20,
        ]

        with (
            patch(
                "algorithms.embedding._embed_many_cached",
                side_effect=cached_vectors,
            ),
            patch(
                "algorithms.embedding.compute_prototypes",
                return_value=(
                    np.zeros((1, 4), dtype=np.float32),
                    np.ones((1, 4), dtype=np.float32),
                ),
            ) as compute_prototypes,
        ):
            model = train_register_model_from_samples(
                [("ok-1.png", "roi1"), ("ok-2.png", "roi1")],
                [("ng-1.png", "roi1")],
                label_names=["roi1"],
                device="cpu",
                feat_net=object(),
            )

        ok_embeddings, ng_embeddings = compute_prototypes.call_args.args
        self.assertEqual(ok_embeddings.shape, (2, 4))
        self.assertEqual(ng_embeddings.shape, (1, 4))
        self.assertEqual(model.ok_bank.shape, (2, 4))
        self.assertEqual(model.ng_bank.shape, (1, 4))

    def test_singleton_matrix_embedding_is_normalized_to_one_vector(self) -> None:
        singleton = np.arange(4, dtype=np.float32).reshape(1, 4)

        with (
            patch(
                "algorithms.embedding._embed_many_cached",
                side_effect=[singleton, singleton + 10],
            ),
            patch(
                "algorithms.embedding.compute_prototypes",
                return_value=(
                    np.zeros((1, 4), dtype=np.float32),
                    np.ones((1, 4), dtype=np.float32),
                ),
            ),
        ):
            model = train_register_model_from_samples(
                [("ok.png", "roi1")],
                [("ng.png", "roi1")],
                label_names=["roi1"],
                device="cpu",
                feat_net=object(),
            )

        self.assertEqual(model.ok_bank.shape, (1, 4))
        self.assertEqual(model.ng_bank.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
