from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from .labelme import (
    labelme_json_of_image,
    read_roi_from_labelme,
    try_read_polygon_points_from_labelme,
    upsert_labelme_rect,
)


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for localization")


def localize_anchor_template(
    ref_img_path: str,
    ref_anchor_xywh: Tuple[int, int, int, int],
    tgt_img_path: str,
    ref_exclude_poly_points: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Tuple[int, int, int, int], float]:
    _require_cv2()
    assert cv2 is not None

    ref = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    tgt = cv2.imread(tgt_img_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise FileNotFoundError(ref_img_path)
    if tgt is None:
        raise FileNotFoundError(tgt_img_path)

    x, y, w, h = ref_anchor_xywh
    patch = ref[y : y + h, x : x + w]
    if patch.size == 0:
        raise ValueError("Reference anchor patch is empty")

    if ref_exclude_poly_points:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        points = np.array(ref_exclude_poly_points, dtype=np.float32)
        points[:, 0] -= float(x)
        points[:, 1] -= float(y)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 0)
        result = cv2.matchTemplate(tgt, patch, cv2.TM_CCORR_NORMED, mask=mask)
    else:
        result = cv2.matchTemplate(tgt, patch, cv2.TM_CCOEFF_NORMED)

    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    tx, ty = int(max_loc[0]), int(max_loc[1])
    return (tx, ty, int(w), int(h)), float(max_val)


def _xywh_to_corners(xywh: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = xywh
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def _bbox_of_points(points: np.ndarray) -> Tuple[int, int, int, int]:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x = int(round(float(x_min)))
    y = int(round(float(y_min)))
    w = int(round(float(x_max - x_min)))
    h = int(round(float(y_max - y_min)))
    return x, y, max(1, w), max(1, h)


def localize_anchor_orb(
    ref_img_path: str,
    ref_anchor_xywh: Tuple[int, int, int, int],
    tgt_img_path: str,
    ref_exclude_poly_points: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Tuple[int, int, int, int], np.ndarray, int]:
    _require_cv2()
    assert cv2 is not None

    ref_gray = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    tgt_gray = cv2.imread(tgt_img_path, cv2.IMREAD_GRAYSCALE)
    if ref_gray is None:
        raise FileNotFoundError(ref_img_path)
    if tgt_gray is None:
        raise FileNotFoundError(tgt_img_path)

    x, y, w, h = ref_anchor_xywh
    patch = ref_gray[y : y + h, x : x + w]
    if patch.size == 0:
        raise ValueError("Reference anchor patch is empty")

    orb = cv2.ORB_create(nfeatures=1500)
    mask = None
    if ref_exclude_poly_points:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        points = np.array(ref_exclude_poly_points, dtype=np.float32)
        points[:, 0] -= float(x)
        points[:, 1] -= float(y)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 0)

    kp1, des1 = orb.detectAndCompute(patch, mask)
    kp2, des2 = orb.detectAndCompute(tgt_gray, None)
    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        raise RuntimeError("Too few ORB features to localize")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda item: item.distance)[:200]
    if len(matches) < 8:
        raise RuntimeError("Too few ORB matches to localize")

    src = np.float32([kp1[item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
    transform, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if transform is None or mask is None:
        raise RuntimeError("Failed to estimate homography")
    inliers = int(mask.ravel().sum())
    if inliers < 8:
        raise RuntimeError(f"Too few homography inliers: {inliers}")

    patch_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(patch_corners, transform).reshape(-1, 2)
    anchor_xywh = _bbox_of_points(projected)
    return anchor_xywh, transform, inliers


def transfer_roi_by_translation(
    ref_anchor_xywh: Tuple[int, int, int, int],
    ref_roi_xywh: Tuple[int, int, int, int],
    tgt_anchor_xywh: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    rx, ry, _rw, _rh = ref_roi_xywh
    ax, ay, _aw, _ah = ref_anchor_xywh
    tx, ty, _tw, _th = tgt_anchor_xywh
    dx = rx - ax
    dy = ry - ay
    return int(tx + dx), int(ty + dy), int(ref_roi_xywh[2]), int(ref_roi_xywh[3])


def transfer_roi_by_homography(
    ref_anchor_xywh: Tuple[int, int, int, int],
    ref_roi_xywh: Tuple[int, int, int, int],
    transform_patch_to_tgt: np.ndarray,
) -> Tuple[int, int, int, int]:
    _require_cv2()
    assert cv2 is not None

    ax, ay, _aw, _ah = ref_anchor_xywh
    roi_corners_ref = _xywh_to_corners(ref_roi_xywh)
    roi_corners_patch = roi_corners_ref - np.array([[ax, ay]], dtype=np.float32)
    points = roi_corners_patch.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(points, transform_patch_to_tgt).reshape(-1, 2)
    return _bbox_of_points(projected)


def autogen_roi_json_from_reference(
    tgt_img_path: str,
    ref_img_path: str,
    method: str = "template",
    anchor_label: str = "anchor",
    roi_label: str = "roi",
) -> str:
    ref_json = labelme_json_of_image(ref_img_path)
    if not os.path.exists(ref_json):
        raise FileNotFoundError(f"Missing reference json: {ref_json}")

    ref_anchor = read_roi_from_labelme(ref_json, label_name=anchor_label)
    ref_roi = read_roi_from_labelme(ref_json, label_name=roi_label)
    ref_exclude = try_read_polygon_points_from_labelme(ref_json, "anchor_mask")

    if method == "template":
        tgt_anchor, _score = localize_anchor_template(
            ref_img_path,
            ref_anchor,
            tgt_img_path,
            ref_exclude_poly_points=ref_exclude,
        )
        tgt_roi = transfer_roi_by_translation(ref_anchor, ref_roi, tgt_anchor)
        upsert_labelme_rect(tgt_img_path, tgt_anchor, label_name=anchor_label)
        return upsert_labelme_rect(tgt_img_path, tgt_roi, label_name=roi_label)

    if method == "orb":
        tgt_anchor, transform, _inliers = localize_anchor_orb(
            ref_img_path,
            ref_anchor,
            tgt_img_path,
            ref_exclude_poly_points=ref_exclude,
        )
        tgt_roi = transfer_roi_by_homography(ref_anchor, ref_roi, transform)
        upsert_labelme_rect(tgt_img_path, tgt_anchor, label_name=anchor_label)
        return upsert_labelme_rect(tgt_img_path, tgt_roi, label_name=roi_label)

    raise ValueError(f"Unknown method: {method}")


__all__ = [
    "autogen_roi_json_from_reference",
    "localize_anchor_orb",
    "localize_anchor_template",
    "transfer_roi_by_homography",
    "transfer_roi_by_translation",
]

