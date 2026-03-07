import numpy as np
from PIL import Image

from iv4matchmethod.xfeat_lighterglue import apply_homography, resize_to_max_dim


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

