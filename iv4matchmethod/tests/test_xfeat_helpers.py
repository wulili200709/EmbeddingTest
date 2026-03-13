import numpy as np
import torch
from PIL import Image

from iv4matchmethod.xfeat_lighterglue import XFeatMatchConfig, apply_homography, match_descriptors_mnn, resize_to_max_dim


def test_resize_to_max_dim_preserves_aspect_ratio():
    image = Image.new("RGB", (2400, 1200), (0, 0, 0))
    resized, scale = resize_to_max_dim(image, 1200)

    assert resized.size == (1200, 600)
    assert abs(scale - 0.5) < 1e-6


def test_apply_homography_translation():
    points = [[1, 2], [3, 4]]
    homography = np.array(
        [
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 20.0],
            [0.0, 0.0, 1.0],
        ]
    )

    transformed = apply_homography(points, homography)
    np.testing.assert_allclose(transformed, np.array([[11.0, 22.0], [13.0, 24.0]]), atol=1e-6)


def test_match_descriptors_mnn_returns_mutual_matches_sorted_by_score():
    feature0 = {
        "descriptors": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.6, 0.8],
            ],
            dtype=torch.float32,
        )
    }
    feature1 = {
        "descriptors": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.6, 0.8],
                [0.8, 0.6],
            ],
            dtype=torch.float32,
        )
    }

    matches = match_descriptors_mnn(feature0, feature1, min_confidence=0.75)

    np.testing.assert_array_equal(matches, np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64))


def test_match_descriptors_mnn_handles_empty_descriptors():
    feature0 = {"descriptors": torch.empty((0, 64), dtype=torch.float32)}
    feature1 = {"descriptors": torch.randn(3, 64, dtype=torch.float32)}

    matches = match_descriptors_mnn(feature0, feature1, min_confidence=0.1)

    assert matches.shape == (0, 2)


def test_xfeat_match_config_uses_matcher_specific_default_thresholds():
    assert XFeatMatchConfig(matcher="lighterglue").min_confidence == 0.1
    assert XFeatMatchConfig(matcher="mnn").min_confidence == 0.82
