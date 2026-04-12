from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


import algorithms.embedding as embedding


class _IdentityFeatureNet:
    def __call__(self, batch):
        return batch


@unittest.skipIf(embedding.cv2 is None, "cv2 is required for the optimized array preprocessing path")
class EmbeddingArrayPreprocessTest(unittest.TestCase):
    def _manual_embed_reference(
        self,
        image_bgr: np.ndarray,
        shape_by_label: dict[str, dict],
        labels: list[str],
    ) -> np.ndarray:
        image_rgb = embedding._rgb_array_from_bgr(image_bgr)
        context = embedding._ImageRoiContext(
            img_path="",
            image=Image.fromarray(image_rgb),
            width=int(image_rgb.shape[1]),
            height=int(image_rgb.shape[0]),
            image_np=None,
            shape_by_label=shape_by_label,
        )
        tensors = [embedding.TF(embedding._resolve_roi_image(context, label_name=label)) for label in labels]
        batch = torch.stack(tensors, dim=0)
        feat = _IdentityFeatureNet()(batch)
        feat = embedding.F.adaptive_avg_pool2d(feat, 1).flatten(1)
        feat = embedding.F.normalize(feat, dim=1)
        return feat.detach().cpu().numpy()

    def test_embed_batch_from_array_matches_reference_for_rectangles(self) -> None:
        image_rgb = np.zeros((40, 52, 3), dtype=np.uint8)
        image_rgb[:, :] = (40, 90, 160)
        image_rgb[4:18, 6:22] = (220, 50, 60)
        image_rgb[20:36, 28:46] = (25, 210, 90)
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        shape_by_label = {
            "roi1": {
                "label": "roi1",
                "shape_type": "rectangle",
                "points": [[6, 4], [22, 18]],
            },
            "roi2": {
                "label": "roi2",
                "shape_type": "rectangle",
                "points": [[28, 20], [46, 36]],
            },
        }
        labels = ["roi1", "roi2"]

        actual = embedding.embed_batch_from_array(
            image_bgr,
            _IdentityFeatureNet(),
            labels,
            shape_by_label=shape_by_label,
            device="cpu",
        )
        expected = self._manual_embed_reference(image_bgr, shape_by_label, labels)
        np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=2e-2)

    def test_embed_batch_from_array_matches_reference_for_polygon(self) -> None:
        image_rgb = np.zeros((48, 48, 3), dtype=np.uint8)
        image_rgb[:, :] = (15, 20, 25)
        image_rgb[8:34, 10:36] = (180, 120, 40)
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        shape_by_label = {
            "roi_poly": {
                "label": "roi_poly",
                "shape_type": "polygon",
                "points": [[12, 10], [35, 12], [30, 32], [14, 34]],
            },
        }
        labels = ["roi_poly"]

        actual = embedding.embed_batch_from_array(
            image_bgr,
            _IdentityFeatureNet(),
            labels,
            shape_by_label=shape_by_label,
            device="cpu",
        )
        expected = self._manual_embed_reference(image_bgr, shape_by_label, labels)
        np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
