import numpy as np

from iv4matchmethod.shape_match import bbox_to_relative_polygon, iter_pose_grid, transform_relative_points


def test_transform_relative_points_rotates_and_scales_about_center():
    points = [[1.0, 0.0], [0.0, 2.0]]

    transformed = transform_relative_points(points, center_xy=[10.0, 20.0], angle_deg=90.0, scale=2.0)

    np.testing.assert_allclose(
        transformed,
        np.array([[10.0, 22.0], [6.0, 20.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_bbox_to_relative_polygon_uses_bbox_center():
    polygon = bbox_to_relative_polygon([100, 200, 40, 20])

    np.testing.assert_allclose(
        polygon,
        np.array([[-20.0, -10.0], [20.0, -10.0], [20.0, 10.0], [-20.0, 10.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_iter_pose_grid_includes_endpoints():
    poses = iter_pose_grid(-4.0, 4.0, 4.0, 0.9, 1.0, 0.1)

    assert poses == [(-4.0, 0.9), (-4.0, 1.0), (0.0, 0.9), (0.0, 1.0), (4.0, 0.9), (4.0, 1.0)]
