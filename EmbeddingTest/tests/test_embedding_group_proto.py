from __future__ import annotations

import sys
import tempfile
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
            "embed_batch",
            side_effect=[vector.reshape(1, -1) for vector in sample_vectors],
        ):
            model = embedding.train_register_model_from_samples(
                ok_samples=[("ok_1.png", "roi1"), ("ok_2.png", "roi2")],
                ng_samples=[("ng_1.png", "roi1"), ("ng_2.png", "roi2")],
                backbone="b0",
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

    def test_sample_training_batches_rois_by_image_inside_group(self) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []

        def _fake_embed_batch(img_path, feat_net, label_names, device=None, roi_xywhs=None):
            labels = tuple(str(label) for label in label_names)
            calls.append((Path(str(img_path)).name, labels))
            vectors = []
            for label in labels:
                value = float(int(label.replace("roi", "")))
                vectors.append(np.asarray([value, value + 10.0], dtype=np.float32))
            return np.stack(vectors)

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            embedding,
            "load_backbone",
            return_value=(object(), 2),
        ), mock.patch.object(
            embedding,
            "embed_batch",
            side_effect=_fake_embed_batch,
        ):
            progress: list[str] = []
            model = embedding.train_register_model_from_samples(
                ok_samples=[
                    (str(Path(tmpdir) / "img_a.png"), "roi1"),
                    (str(Path(tmpdir) / "img_a.png"), "roi2"),
                    (str(Path(tmpdir) / "img_b.png"), "roi3"),
                ],
                ng_samples=[
                    (str(Path(tmpdir) / "img_c.png"), "roi4"),
                    (str(Path(tmpdir) / "img_c.png"), "roi5"),
                ],
                backbone="b0",
                score_mode="topk",
                margin=0.02,
                topk=3,
                label_name="pusher",
                label_names=["pusher"],
                collapse_to_proto=False,
                device="cpu",
                cache_dir=str(Path(tmpdir) / "cache"),
                progress_callback=progress.append,
            )

            second_progress: list[str] = []
            embedding.train_register_model_from_samples(
                ok_samples=[
                    (str(Path(tmpdir) / "img_a.png"), "roi1"),
                    (str(Path(tmpdir) / "img_a.png"), "roi2"),
                    (str(Path(tmpdir) / "img_b.png"), "roi3"),
                ],
                ng_samples=[
                    (str(Path(tmpdir) / "img_c.png"), "roi4"),
                    (str(Path(tmpdir) / "img_c.png"), "roi5"),
                ],
                backbone="b0",
                score_mode="topk",
                margin=0.02,
                topk=3,
                label_name="pusher",
                label_names=["pusher"],
                collapse_to_proto=False,
                device="cpu",
                cache_dir=str(Path(tmpdir) / "cache"),
                progress_callback=second_progress.append,
            )

        self.assertEqual(
            calls,
            [
                ("img_a.png", ("roi1", "roi2")),
                ("img_b.png", ("roi3",)),
                ("img_c.png", ("roi4", "roi5")),
            ],
        )
        self.assertEqual(model.ok_analysis_bank.shape, (3, 2))
        self.assertTrue(any("embedding OK 1-2/3" in message for message in progress))
        self.assertTrue(all(message.startswith("cache ") for message in second_progress))


if __name__ == "__main__":
    unittest.main()
