"""
qr_core.py

把 quick_register_embed.py 的核心逻辑抽出来，供 GUI / CLI 复用：
- Backbone 选择（efficientnet_b0 / mobilenet_v3_small / mobilenet_v3_large）
- labelme ROI 读写（无 json 时也能生成）
- ROI embedding
- 两种评分：proto / topk
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import models, transforms

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from shape_model_like import ScaledShapeModel
except Exception:  # pragma: no cover
    ScaledShapeModel = None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(name: str, device: Optional[str] = None):
    """
    返回 feature extractor（不含分类头）和输出通道数。
    """
    device = device or get_device()
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat = m.features
        out_ch = 1280
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        feat = m.features
        out_ch = 576
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        feat = m.features
        out_ch = 960
    else:
        raise ValueError(f"Unknown backbone: {name}")

    feat.eval().to(device)
    return feat, out_ch


TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_images(folder: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files: List[str] = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)


def labelme_json_of_image(img_path: str) -> str:
    base, _ = os.path.splitext(img_path)
    return base + ".json"


def clamp_roi_xywh(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def roi_xywh_to_labelme_shape(x: int, y: int, w: int, h: int, label_name: str = "roi") -> dict:
    # labelme rectangle: points = [[x1, y1], [x2, y2]]
    return {
        "label": label_name,
        "points": [[float(x), float(y)], [float(x + w), float(y + h)]],
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {},
    }

def polygon_points_to_labelme_shape(points_xy: List[Tuple[float, float]], label_name: str) -> dict:
    return {
        "label": label_name,
        "points": [[float(x), float(y)] for x, y in points_xy],
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
    }


def _new_labelme_base(img_path: str) -> dict:
    with Image.open(img_path) as im:
        W, H = im.size
    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }


def read_labelme_json_or_create(img_path: str, json_path: Optional[str] = None) -> Tuple[str, dict]:
    """
    读取 labelme json；若不存在则创建一个空的基础结构（不写盘，交给调用者决定写不写）。
    """
    jpath = json_path or labelme_json_of_image(img_path)
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # minimal sanitize
        data.setdefault("shapes", [])
        data.setdefault("flags", {})
        return jpath, data
    return jpath, _new_labelme_base(img_path)


def upsert_labelme_shape(
    img_path: str,
    label_name: str,
    shape: dict,
    json_path: Optional[str] = None,
) -> str:
    """
    在同一个 json 里新增/更新一个 shape（按 label 覆盖同名 shape）。
    这样同一张图可以同时保存 roi + anchor（不会互相覆盖）。
    """
    jpath, data = read_labelme_json_or_create(img_path, json_path=json_path)
    with Image.open(img_path) as im:
        W, H = im.size

    shapes = list(data.get("shapes", []))
    replaced = False
    for i, s in enumerate(shapes):
        if s.get("label") == label_name:
            shapes[i] = shape
            replaced = True
            break
    if not replaced:
        shapes.append(shape)
    data["shapes"] = shapes
    # ensure image size/path correct
    data["imagePath"] = os.path.basename(img_path)
    data["imageHeight"] = H
    data["imageWidth"] = W

    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return jpath


def upsert_labelme_rect(
    img_path: str,
    xywh: Tuple[int, int, int, int],
    label_name: str,
    json_path: Optional[str] = None,
) -> str:
    with Image.open(img_path) as im:
        W, H = im.size
    x, y, w, h = clamp_roi_xywh(*xywh, W=W, H=H)
    shape = roi_xywh_to_labelme_shape(x, y, w, h, label_name=label_name)
    return upsert_labelme_shape(img_path, label_name=label_name, shape=shape, json_path=json_path)


def upsert_labelme_polygon(
    img_path: str,
    points_xy: List[Tuple[float, float]],
    label_name: str,
    json_path: Optional[str] = None,
) -> str:
    # points 不做强校验，读取 bbox 时会自动取外接矩形
    shape = polygon_points_to_labelme_shape(points_xy, label_name=label_name)
    return upsert_labelme_shape(img_path, label_name=label_name, shape=shape, json_path=json_path)


def delete_labelme_shape(img_path: str, label_name: str, json_path: Optional[str] = None) -> bool:
    """
    从同名 labelme json 中删除指定 label 的 shape。
    返回：是否真的删除了（找到并删除=True）。
    """
    jpath = json_path or labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        return False
    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    shapes = list(data.get("shapes", []))
    new_shapes = [s for s in shapes if s.get("label") != label_name]
    if len(new_shapes) == len(shapes):
        return False
    data["shapes"] = new_shapes
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def write_labelme_json_for_roi(
    img_path: str,
    roi_xywh: Tuple[int, int, int, int],
    label_name: str = "roi",
    out_json_path: Optional[str] = None,
) -> str:
    """
    给单张图片写一个最小可用的 labelme json（rectangle）。
    """
    # 兼容旧调用：现在改为 upsert，不会覆盖同文件其它 label（比如 anchor）
    return upsert_labelme_rect(img_path, roi_xywh, label_name=label_name, json_path=out_json_path)


def read_roi_from_labelme(labelme_json_path: str, label_name: str = "roi") -> Tuple[int, int, int, int]:
    """
    从 labelme 的标注读取 ROI。
    - rectangle: points=[[x1,y1],[x2,y2]]
    - polygon: points=[[x1,y1],...[xn,yn]] -> 取外接矩形 bbox
    """
    with open(labelme_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for s in data.get("shapes", []):
        if s.get("label") != label_name:
            continue
        pts = np.array(s["points"], dtype=np.float32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        x = int(round(float(x_min)))
        y = int(round(float(y_min)))
        w = int(round(float(x_max - x_min)))
        h = int(round(float(y_max - y_min)))
        if w <= 0 or h <= 0:
            raise ValueError(f"ROI 宽/高不合法：{labelme_json_path}")
        return x, y, w, h

    raise RuntimeError(f"在 {labelme_json_path} 中找不到 label='{label_name}' 的 ROI")


def try_read_xywh_from_labelme(labelme_json_path: str, label_name: str) -> Optional[Tuple[int, int, int, int]]:
    try:
        return read_roi_from_labelme(labelme_json_path, label_name=label_name)
    except Exception:
        return None


def read_shape_from_labelme(labelme_json_path: str, label_name: str) -> Optional[dict]:
    with open(labelme_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for s in data.get("shapes", []):
        if s.get("label") == label_name:
            return s
    return None


def list_shapes_from_labelme(labelme_json_path: str, label_prefix: Optional[str] = None) -> List[dict]:
    with open(labelme_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shapes = []
    for s in data.get("shapes", []):
        if not isinstance(s, dict):
            continue
        label = str(s.get("label", ""))
        if label_prefix and not label.startswith(label_prefix):
            continue
        shapes.append(s)
    return shapes


def sorted_label_names_from_labelme(labelme_json_path: str, label_prefix: str = "roi") -> List[str]:
    labels = [str(s.get("label", "")) for s in list_shapes_from_labelme(labelme_json_path, label_prefix=label_prefix)]

    def _sort_key(name: str):
        suffix = name[len(label_prefix) :]
        if suffix.isdigit():
            return (0, int(suffix))
        if name == label_prefix:
            return (0, 0)
        return (1, name)

    return sorted([name for name in labels if name], key=_sort_key)


def try_read_polygon_points_from_labelme(labelme_json_path: str, label_name: str) -> Optional[List[Tuple[float, float]]]:
    """
    读取 polygon 的 points；如果不是 polygon 或不存在，返回 None。
    """
    try:
        s = read_shape_from_labelme(labelme_json_path, label_name=label_name)
        if not s:
            return None
        if s.get("shape_type") != "polygon":
            return None
        pts = s.get("points", [])
        return [(float(x), float(y)) for x, y in pts]
    except Exception:
        return None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("未安装 OpenCV（cv2），无法使用模板匹配/特征点定位")


def localize_anchor_template(
    ref_img_path: str,
    ref_anchor_xywh: Tuple[int, int, int, int],
    tgt_img_path: str,
    ref_exclude_poly_points: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Tuple[int, int, int, int], float]:
    """
    模板匹配：用参考图的 anchor patch 在目标图中找最相似位置。
    返回：目标图 anchor 的 xywh（与参考 anchor 同宽高）以及匹配分数。
    """
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
        raise ValueError("ref anchor patch 为空")

    mask = None
    if ref_exclude_poly_points:
        # keep mask = 255, exclude region = 0
        mask = np.ones((h, w), dtype=np.uint8) * 255
        pts = np.array(ref_exclude_poly_points, dtype=np.float32)
        pts[:, 0] -= float(x)
        pts[:, 1] -= float(y)
        pts_i = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts_i], 0)
        # matchTemplate(mask=) 支持的 method 有限制，这里用 TM_CCORR_NORMED
        method = cv2.TM_CCORR_NORMED
        res = cv2.matchTemplate(tgt, patch, method, mask=mask)
    else:
        res = cv2.matchTemplate(tgt, patch, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    tx, ty = int(max_loc[0]), int(max_loc[1])
    return (tx, ty, int(w), int(h)), float(max_val)


def _xywh_to_corners(xywh: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = xywh
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def _bbox_of_points(pts: np.ndarray) -> Tuple[int, int, int, int]:
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
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
    """
    特征点(ORB)匹配：在参考 anchor patch 和目标图之间匹配，估计单应性 H（patch->target）。
    返回：目标图 anchor bbox、H、inliers 数。
    """
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
        raise ValueError("ref anchor patch 为空")

    orb = cv2.ORB_create(nfeatures=1500)
    m = None
    if ref_exclude_poly_points:
        m = np.ones((h, w), dtype=np.uint8) * 255
        pts = np.array(ref_exclude_poly_points, dtype=np.float32)
        pts[:, 0] -= float(x)
        pts[:, 1] -= float(y)
        pts_i = np.round(pts).astype(np.int32)
        cv2.fillPoly(m, [pts_i], 0)
    kp1, des1 = orb.detectAndCompute(patch, m)
    kp2, des2 = orb.detectAndCompute(tgt_gray, None)
    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        raise RuntimeError("ORB 特征点过少，无法定位（可换模板匹配或增大/换 anchor 区域）")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[:200]
    if len(matches) < 8:
        raise RuntimeError("ORB 匹配点过少，无法定位")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        raise RuntimeError("单应性估计失败")
    inliers = int(mask.ravel().sum())
    if inliers < 8:
        raise RuntimeError(f"RANSAC 内点太少：{inliers}")

    # anchor corners in patch coordinates: (0,0)-(w,h)
    patch_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(patch_corners, H).reshape(-1, 2)
    anchor_xywh = _bbox_of_points(proj)
    return anchor_xywh, H, inliers


def transfer_roi_by_translation(
    ref_anchor_xywh: Tuple[int, int, int, int],
    ref_roi_xywh: Tuple[int, int, int, int],
    tgt_anchor_xywh: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """
    仅用平移：ROI 相对 anchor 的偏移保持不变。
    """
    rx, ry, _rw, _rh = ref_roi_xywh
    ax, ay, _aw, _ah = ref_anchor_xywh
    tx, ty, _tw, _th = tgt_anchor_xywh
    dx = rx - ax
    dy = ry - ay
    x = tx + dx
    y = ty + dy
    w = ref_roi_xywh[2]
    h = ref_roi_xywh[3]
    return int(x), int(y), int(w), int(h)


def transfer_roi_by_homography(
    ref_anchor_xywh: Tuple[int, int, int, int],
    ref_roi_xywh: Tuple[int, int, int, int],
    H_patch_to_tgt: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    用单应性：把 ROI（相对 anchor 的坐标）映射到目标图，再取 bbox。
    注意：H 是 anchor patch 坐标系(0..w,0..h) -> 目标图坐标系。
    """
    _require_cv2()
    assert cv2 is not None

    ax, ay, _aw, _ah = ref_anchor_xywh
    # ROI corners in patch coordinates
    roi_corners_ref = _xywh_to_corners(ref_roi_xywh)
    roi_corners_patch = roi_corners_ref - np.array([[ax, ay]], dtype=np.float32)
    pts = roi_corners_patch.reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(pts, H_patch_to_tgt).reshape(-1, 2)
    return _bbox_of_points(proj)


def autogen_roi_json_from_reference(
    tgt_img_path: str,
    ref_img_path: str,
    method: str = "template",  # "template" | "orb"
    anchor_label: str = "anchor",
    roi_label: str = "roi",
) -> str:
    """
    用参考图(ref)的 anchor+roi，在目标图(tgt)上自动定位 anchor 并生成 roi，写入 tgt 的 labelme json。
    - method=template: 快，平移为主
    - method=orb: 更鲁棒，可处理一定旋转/透视
    """
    ref_json = labelme_json_of_image(ref_img_path)
    if not os.path.exists(ref_json):
        raise FileNotFoundError(f"参考图缺少 json：{ref_json}")

    ref_anchor = read_roi_from_labelme(ref_json, label_name=anchor_label)
    ref_roi = read_roi_from_labelme(ref_json, label_name=roi_label)
    ref_exclude = try_read_polygon_points_from_labelme(ref_json, "anchor_mask")

    if method == "template":
        tgt_anchor, score = localize_anchor_template(
            ref_img_path, ref_anchor, tgt_img_path, ref_exclude_poly_points=ref_exclude
        )
        # translation-only
        tgt_roi = transfer_roi_by_translation(ref_anchor, ref_roi, tgt_anchor)
        # 写入 json（anchor + roi）
        upsert_labelme_rect(tgt_img_path, tgt_anchor, label_name=anchor_label)
        jpath = upsert_labelme_rect(tgt_img_path, tgt_roi, label_name=roi_label)
        return jpath
    if method == "orb":
        tgt_anchor, H, _inliers = localize_anchor_orb(
            ref_img_path, ref_anchor, tgt_img_path, ref_exclude_poly_points=ref_exclude
        )
        tgt_roi = transfer_roi_by_homography(ref_anchor, ref_roi, H)
        upsert_labelme_rect(tgt_img_path, tgt_anchor, label_name=anchor_label)
        jpath = upsert_labelme_rect(tgt_img_path, tgt_roi, label_name=roi_label)
        return jpath
    raise ValueError(f"Unknown method: {method}")


def _require_shape_model():
    _require_cv2()
    if ScaledShapeModel is None:
        raise RuntimeError("shape_model_like 不可用，无法使用 shape_model 模式")


def _rect_xywh_to_points(xywh: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
    x, y, w, h = xywh
    return [
        (float(x), float(y)),
        (float(x + w), float(y)),
        (float(x + w), float(y + h)),
        (float(x), float(y + h)),
    ]


def _shape_points_from_labelme(labelme_json_path: str, label_name: str) -> Optional[List[Tuple[float, float]]]:
    poly_pts = try_read_polygon_points_from_labelme(labelme_json_path, label_name)
    if poly_pts and len(poly_pts) >= 3:
        return poly_pts
    xywh = try_read_xywh_from_labelme(labelme_json_path, label_name)
    if xywh:
        return _rect_xywh_to_points(xywh)
    return None


def _mask_from_points(h: int, w: int, points: Sequence[Tuple[float, float]], fill_value: int = 255) -> np.ndarray:
    _require_cv2()
    assert cv2 is not None
    mask = np.zeros((int(h), int(w)), dtype=np.uint8)
    if not points:
        return mask
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Invalid polygon points")
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts_i], int(fill_value))
    return mask


def _transform_points(
    points_xy: Sequence[Tuple[float, float]],
    origin_rc: Tuple[float, float],
    row: float,
    col: float,
    angle: float,
    scale: float,
) -> List[Tuple[float, float]]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    orow, ocol = origin_rc
    out: List[Tuple[float, float]] = []
    for x, y in points_xy:
        dcol = float(x) - float(ocol)
        drow = float(y) - float(orow)
        col_f = float(col) + scale * (cos_a * dcol - sin_a * drow)
        row_f = float(row) + scale * (sin_a * dcol + cos_a * drow)
        out.append((col_f, row_f))
    return out


def create_shape_model_from_reference(
    ref_img_path: str,
    model_path: str,
    *,
    anchor_label: str = "anchor",
    anchor_mask_label: str = "anchor_mask",
    nbins: int = 30,
    canny1: int = 50,
    canny2: int = 150,
) -> str:
    _require_shape_model()
    assert cv2 is not None

    ref = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise FileNotFoundError(ref_img_path)

    ref_json = labelme_json_of_image(ref_img_path)
    if not os.path.exists(ref_json):
        raise FileNotFoundError(f"参考图缺少 json：{ref_json}")

    anchor_pts = _shape_points_from_labelme(ref_json, anchor_label)
    if not anchor_pts:
        raise RuntimeError("参考图缺少 anchor 标注")

    mask = _mask_from_points(ref.shape[0], ref.shape[1], anchor_pts, fill_value=255)
    exclude_pts = _shape_points_from_labelme(ref_json, anchor_mask_label)
    if exclude_pts:
        exclude = _mask_from_points(ref.shape[0], ref.shape[1], exclude_pts, fill_value=255)
        mask[exclude > 0] = 0

    model = ScaledShapeModel.create(ref, mask=mask, nbins=nbins, canny1=canny1, canny2=canny2)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
    return model_path


def autogen_roi_json_from_shape_model(
    tgt_img_path: str,
    ref_img_path: str,
    model_path: str,
    *,
    anchor_label: str = "anchor",
    roi_label: str = "roi",
    anchor_mask_label: str = "anchor_mask",
    angle_start: float = -math.pi,
    angle_extent: float = math.pi * 2.0,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
    min_score: float = 0.18,
    num_matches: int = 1,
    max_overlap: float = 0.3,
) -> str:
    _require_shape_model()
    assert cv2 is not None

    if not os.path.exists(model_path):
        create_shape_model_from_reference(
            ref_img_path,
            model_path,
            anchor_label=anchor_label,
            anchor_mask_label=anchor_mask_label,
        )

    model = ScaledShapeModel.load(model_path)

    tgt = cv2.imread(tgt_img_path, cv2.IMREAD_GRAYSCALE)
    if tgt is None:
        raise FileNotFoundError(tgt_img_path)

    rows, cols, angs, scs, _scores = model.find(
        tgt,
        angle_start=angle_start,
        angle_extent=angle_extent,
        scale_min=scale_min,
        scale_max=scale_max,
        min_score=min_score,
        num_matches=num_matches,
        max_overlap=max_overlap,
    )
    if rows.size == 0:
        raise RuntimeError("shape_model 未匹配到目标")

    row = float(rows[0])
    col = float(cols[0])
    angle = float(angs[0])
    scale = float(scs[0])

    ref_json = labelme_json_of_image(ref_img_path)
    if not os.path.exists(ref_json):
        raise FileNotFoundError(f"参考图缺少 json：{ref_json}")

    roi_pts = _shape_points_from_labelme(ref_json, roi_label)
    if not roi_pts:
        raise RuntimeError("参考图缺少 roi 标注")

    anchor_pts = _shape_points_from_labelme(ref_json, anchor_label)

    tgt_roi_pts = _transform_points(roi_pts, model.origin_rc, row, col, angle, scale)
    if anchor_pts:
        tgt_anchor_pts = _transform_points(anchor_pts, model.origin_rc, row, col, angle, scale)
        upsert_labelme_polygon(tgt_img_path, tgt_anchor_pts, label_name=anchor_label)

    jpath = upsert_labelme_polygon(tgt_img_path, tgt_roi_pts, label_name=roi_label)
    return jpath


@torch.no_grad()
def embed_one(
    img_path: str,
    feat_net,
    label_name: str = "roi",
    device: Optional[str] = None,
    roi_xywh: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    - 默认从同名 labelme json 读取 ROI
    - 若提供 roi_xywh，则直接用该 ROI（并可由 GUI 负责保存 json）
    """
    device = device or get_device()
    with Image.open(img_path) as img_raw:
        img = img_raw.convert("RGB")
    W, H = img.size

    roi_img: Optional[Image.Image] = None
    if roi_xywh is None:
        jpath = labelme_json_of_image(img_path)
        if not os.path.exists(jpath):
            raise FileNotFoundError(f"缺少 labelme json：{jpath}")
        shape = read_shape_from_labelme(jpath, label_name=label_name)
        if shape and str(shape.get("shape_type", "rectangle")) == "polygon":
            pts = np.asarray(shape.get("points", []), dtype=np.float32)
            if pts.shape[0] >= 3 and cv2 is not None:
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                x = int(round(float(x_min)))
                y = int(round(float(y_min)))
                w = int(round(float(x_max - x_min)))
                h = int(round(float(y_max - y_min)))
                x, y, w, h = clamp_roi_xywh(x, y, w, h, W=W, H=H)
                img_np = np.array(img)
                crop = img_np[y : y + h, x : x + w].copy()
                rel_pts = np.round(pts - np.array([[x, y]], dtype=np.float32)).astype(np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [rel_pts], 255)
                crop[mask == 0] = 0
                roi_img = Image.fromarray(crop)
            else:
                x, y, w, h = read_roi_from_labelme(jpath, label_name=label_name)
        else:
            x, y, w, h = read_roi_from_labelme(jpath, label_name=label_name)
    else:
        x, y, w, h = roi_xywh

    if roi_img is None:
        x, y, w, h = clamp_roi_xywh(x, y, w, h, W=W, H=H)
        roi_img = img.crop((x, y, x + w, y + h))
    t = TF(roi_img).unsqueeze(0).to(device)  # [1,3,224,224]
    f = feat_net(t)  # [1,C,H,W]
    f = F.adaptive_avg_pool2d(f, 1).flatten(1)  # [1,C]
    f = F.normalize(f, dim=1)  # L2 normalize
    return f[0].detach().cpu().numpy()


def embed_many(
    img_path: str,
    feat_net,
    label_names: Sequence[str],
    device: Optional[str] = None,
) -> np.ndarray:
    labels = [str(name) for name in label_names if str(name).strip()]
    if not labels:
        raise ValueError("label_names cannot be empty")
    parts = [embed_one(img_path, feat_net, label_name=label, device=device) for label in labels]
    if len(parts) == 1:
        return parts[0]
    vec = np.concatenate(parts, axis=0)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


def score_topk(e: np.ndarray, bank: np.ndarray, k: int = 3) -> float:
    sims = bank @ e  # [N]
    if sims.size == 0:
        return float("-inf")
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k >= sims.size:
        return float(np.mean(sims))
    return float(np.mean(np.sort(sims)[-k:]))


@dataclass
class RegisterModel:
    backbone: str
    score_mode: str  # "proto" or "topk"
    margin: float
    topk: int
    label_name: str = "roi"
    label_names: Optional[List[str]] = None
    device: str = "cpu"

    ok_proto: Optional[np.ndarray] = None  # [1,C]
    ng_proto: Optional[np.ndarray] = None  # [1,C]
    ok_bank: Optional[np.ndarray] = None  # [N,C]
    ng_bank: Optional[np.ndarray] = None  # [N,C]

    def is_ready(self) -> bool:
        return self.ok_bank is not None and self.ng_bank is not None and self.ok_bank.size > 0 and self.ng_bank.size > 0

    def effective_label_names(self) -> List[str]:
        labels = [str(name) for name in (self.label_names or []) if str(name).strip()]
        if labels:
            return labels
        return [str(self.label_name or "roi")]


def save_register_model_npz(model: RegisterModel, npz_path: str):
    """
    保存“注册模型”（不含 torch backbone 权重；backbone 使用 torchvision 预训练，随用随载）。
    """
    if not model.is_ready():
        raise RuntimeError("model 未就绪，无法保存")
    ok_proto = model.ok_proto
    ng_proto = model.ng_proto
    ok_bank = model.ok_bank
    ng_bank = model.ng_bank
    assert ok_proto is not None and ng_proto is not None and ok_bank is not None and ng_bank is not None

    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez_compressed(
        npz_path,
        backbone=np.array([model.backbone]),
        score_mode=np.array([model.score_mode]),
        margin=np.array([float(model.margin)], dtype=np.float32),
        topk=np.array([int(model.topk)], dtype=np.int32),
        label_name=np.array([model.label_name]),
        label_names=np.array(model.effective_label_names()),
        device=np.array([model.device]),
        ok_proto=ok_proto.astype(np.float32),
        ng_proto=ng_proto.astype(np.float32),
        ok_bank=ok_bank.astype(np.float32),
        ng_bank=ng_bank.astype(np.float32),
    )


def load_register_model_npz(npz_path: str) -> RegisterModel:
    z = np.load(npz_path, allow_pickle=False)
    m = RegisterModel(
        backbone=str(z["backbone"][0]),
        score_mode=str(z["score_mode"][0]),
        margin=float(z["margin"][0]),
        topk=int(z["topk"][0]),
        label_name=str(z["label_name"][0]),
        label_names=[str(v) for v in z["label_names"]] if "label_names" in z.files else [str(z["label_name"][0])],
        device=str(z["device"][0]),
        ok_proto=z["ok_proto"],
        ng_proto=z["ng_proto"],
        ok_bank=z["ok_bank"],
        ng_bank=z["ng_bank"],
    )
    return m


def train_register_model(
    ok_files: Sequence[str],
    ng_files: Sequence[str],
    backbone: str = "efficientnet_b0",
    score_mode: str = "proto",
    margin: float = 0.02,
    topk: int = 3,
    label_name: str = "roi",
    label_names: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
) -> RegisterModel:
    if not ok_files or not ng_files:
        raise RuntimeError("OK/NG 都需要至少 1 张图片")
    device = device or get_device()
    feat_net, _ = load_backbone(backbone, device=device)
    labels = [str(name) for name in (label_names or [label_name]) if str(name).strip()]
    if not labels:
        labels = ["roi"]

    ok_emb = np.stack([embed_many(p, feat_net, labels, device=device) for p in ok_files])
    ng_emb = np.stack([embed_many(p, feat_net, labels, device=device) for p in ng_files])

    ok_proto = ok_emb.mean(axis=0, keepdims=True)
    ok_proto = ok_proto / np.linalg.norm(ok_proto, axis=1, keepdims=True)
    ng_proto = ng_emb.mean(axis=0, keepdims=True)
    ng_proto = ng_proto / np.linalg.norm(ng_proto, axis=1, keepdims=True)

    m = RegisterModel(
        backbone=backbone,
        score_mode=score_mode,
        margin=float(margin),
        topk=int(topk),
        label_name=labels[0],
        label_names=labels,
        device=device,
        ok_proto=ok_proto,
        ng_proto=ng_proto,
        ok_bank=ok_emb,
        ng_bank=ng_emb,
    )
    return m


def predict_one_with_model(
    e: np.ndarray,
    model: RegisterModel,
) -> Tuple[str, float, float, float]:
    if not model.is_ready():
        raise RuntimeError("model 未训练/未就绪")
    ok_proto = model.ok_proto
    ng_proto = model.ng_proto
    ok_bank = model.ok_bank
    ng_bank = model.ng_bank
    assert ok_proto is not None and ng_proto is not None and ok_bank is not None and ng_bank is not None

    if model.score_mode == "proto":
        sim_ok = float(e @ ok_proto[0])
        sim_ng = float(e @ ng_proto[0])
    elif model.score_mode == "topk":
        k_ok = min(model.topk, len(ok_bank))
        k_ng = min(model.topk, len(ng_bank))
        sim_ok = score_topk(e, ok_bank, k=k_ok)
        sim_ng = score_topk(e, ng_bank, k=k_ng)
    else:
        raise ValueError(f"Unknown score mode: {model.score_mode}")

    diff = sim_ok - sim_ng
    pred = "OK" if diff >= model.margin else "NG"
    return pred, float(diff), sim_ok, sim_ng


# ---- backward-compatible API (for quick_register_embed.py) ----
def predict_one(
    e: np.ndarray,
    ok_proto: np.ndarray,
    ng_proto: np.ndarray,
    ok_bank: np.ndarray,
    ng_bank: np.ndarray,
    mode: str,
    margin: float,
    topk: int,
) -> Tuple[str, float, float, float]:
    """
    兼容 quick_register_embed.py 的旧签名。
    """
    if mode == "proto":
        sim_ok = float(e @ ok_proto[0])
        sim_ng = float(e @ ng_proto[0])
    elif mode == "topk":
        k_ok = min(topk, len(ok_bank))
        k_ng = min(topk, len(ng_bank))
        sim_ok = score_topk(e, ok_bank, k=k_ok)
        sim_ng = score_topk(e, ng_bank, k=k_ng)
    else:
        raise ValueError(f"Unknown score mode: {mode}")

    diff = sim_ok - sim_ng
    pred = "OK" if diff >= margin else "NG"
    return pred, float(diff), sim_ok, sim_ng
