import math

import numpy as np

from iv4matchmethod.geometry import transform_polygon


def test_transform_polygon_translation_only():
    roi = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    mapped = transform_polygon(roi, cx=5.0, cy=7.0, theta=0.0, sx=2.0, sy=3.0)
    expected = np.array([[3.0, 4.0], [7.0, 4.0], [7.0, 10.0], [3.0, 10.0]])
    np.testing.assert_allclose(mapped, expected, atol=1e-5)


def test_transform_polygon_rotation():
    roi = [[1, 0]]
    mapped = transform_polygon(roi, cx=0.0, cy=0.0, theta=math.pi / 2.0, sx=1.0, sy=1.0)
    np.testing.assert_allclose(mapped, np.array([[0.0, 1.0]]), atol=1e-5)

