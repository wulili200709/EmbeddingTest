from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def labelme_json_of_image(img_path: str) -> str:
    base, _ = os.path.splitext(img_path)
    return base + ".json"


def clamp_roi_xywh(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def roi_xywh_to_labelme_shape(
    x: int,
    y: int,
    w: int,
    h: int,
    label_name: str = "roi",
) -> dict:
    return {
        "label": label_name,
        "points": [[float(x), float(y)], [float(x + w), float(y + h)]],
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {},
    }


def polygon_points_to_labelme_shape(
    points_xy: List[Tuple[float, float]],
    label_name: str,
) -> dict:
    return {
        "label": label_name,
        "points": [[float(x), float(y)] for x, y in points_xy],
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
    }


def _new_labelme_base(img_path: str) -> dict:
    with Image.open(img_path) as image:
        width, height = image.size
    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def read_labelme_json_or_create(
    img_path: str,
    json_path: Optional[str] = None,
) -> Tuple[str, dict]:
    jpath = json_path or labelme_json_of_image(img_path)
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
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
    jpath, data = read_labelme_json_or_create(img_path, json_path=json_path)
    with Image.open(img_path) as image:
        width, height = image.size

    shapes = list(data.get("shapes", []))
    replaced = False
    for index, current in enumerate(shapes):
        if current.get("label") == label_name:
            shapes[index] = shape
            replaced = True
            break
    if not replaced:
        shapes.append(shape)

    data["shapes"] = shapes
    data["imagePath"] = os.path.basename(img_path)
    data["imageHeight"] = height
    data["imageWidth"] = width

    with open(jpath, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return jpath


def upsert_labelme_rect(
    img_path: str,
    xywh: Tuple[int, int, int, int],
    label_name: str,
    json_path: Optional[str] = None,
) -> str:
    with Image.open(img_path) as image:
        width, height = image.size
    x, y, w, h = clamp_roi_xywh(*xywh, W=width, H=height)
    shape = roi_xywh_to_labelme_shape(x, y, w, h, label_name=label_name)
    return upsert_labelme_shape(img_path, label_name=label_name, shape=shape, json_path=json_path)


def upsert_labelme_polygon(
    img_path: str,
    points_xy: List[Tuple[float, float]],
    label_name: str,
    json_path: Optional[str] = None,
) -> str:
    shape = polygon_points_to_labelme_shape(points_xy, label_name=label_name)
    return upsert_labelme_shape(img_path, label_name=label_name, shape=shape, json_path=json_path)


def delete_labelme_shape(
    img_path: str,
    label_name: str,
    json_path: Optional[str] = None,
) -> bool:
    jpath = json_path or labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        return False
    with open(jpath, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    shapes = list(data.get("shapes", []))
    filtered = [shape for shape in shapes if shape.get("label") != label_name]
    if len(filtered) == len(shapes):
        return False
    data["shapes"] = filtered
    with open(jpath, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return True


def write_labelme_json_for_roi(
    img_path: str,
    roi_xywh: Tuple[int, int, int, int],
    label_name: str = "roi",
    out_json_path: Optional[str] = None,
) -> str:
    return upsert_labelme_rect(img_path, roi_xywh, label_name=label_name, json_path=out_json_path)


def read_roi_from_labelme(
    labelme_json_path: str,
    label_name: str = "roi",
) -> Tuple[int, int, int, int]:
    with open(labelme_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    for shape in data.get("shapes", []):
        if shape.get("label") != label_name:
            continue
        points = np.array(shape["points"], dtype=np.float32)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        x = int(round(float(x_min)))
        y = int(round(float(y_min)))
        w = int(round(float(x_max - x_min)))
        h = int(round(float(y_max - y_min)))
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid ROI size: {labelme_json_path}")
        return x, y, w, h

    raise RuntimeError(f"Label '{label_name}' not found in {labelme_json_path}")


def try_read_xywh_from_labelme(
    labelme_json_path: str,
    label_name: str,
) -> Optional[Tuple[int, int, int, int]]:
    try:
        return read_roi_from_labelme(labelme_json_path, label_name=label_name)
    except Exception:
        return None


def read_shape_from_labelme(labelme_json_path: str, label_name: str) -> Optional[dict]:
    with open(labelme_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for shape in data.get("shapes", []):
        if shape.get("label") == label_name:
            return shape
    return None


def list_shapes_from_labelme(
    labelme_json_path: str,
    label_prefix: Optional[str] = None,
) -> List[dict]:
    with open(labelme_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    shapes: List[dict] = []
    for shape in data.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", ""))
        if label_prefix and not label.startswith(label_prefix):
            continue
        shapes.append(shape)
    return shapes


def sorted_label_names_from_labelme(
    labelme_json_path: str,
    label_prefix: str = "roi",
) -> List[str]:
    labels = [
        str(shape.get("label", ""))
        for shape in list_shapes_from_labelme(labelme_json_path, label_prefix=label_prefix)
    ]

    def _sort_key(name: str) -> tuple[int, int | str]:
        suffix = name[len(label_prefix) :]
        if suffix.isdigit():
            return (0, int(suffix))
        if name == label_prefix:
            return (0, 0)
        return (1, name)

    return sorted([label for label in labels if label], key=_sort_key)


def try_read_polygon_points_from_labelme(
    labelme_json_path: str,
    label_name: str,
) -> Optional[List[Tuple[float, float]]]:
    try:
        shape = read_shape_from_labelme(labelme_json_path, label_name=label_name)
        if not shape or shape.get("shape_type") != "polygon":
            return None
        return [(float(x), float(y)) for x, y in shape.get("points", [])]
    except Exception:
        return None


__all__ = [
    "clamp_roi_xywh",
    "delete_labelme_shape",
    "labelme_json_of_image",
    "list_shapes_from_labelme",
    "polygon_points_to_labelme_shape",
    "read_labelme_json_or_create",
    "read_roi_from_labelme",
    "read_shape_from_labelme",
    "roi_xywh_to_labelme_shape",
    "sorted_label_names_from_labelme",
    "try_read_polygon_points_from_labelme",
    "try_read_xywh_from_labelme",
    "upsert_labelme_polygon",
    "upsert_labelme_rect",
    "upsert_labelme_shape",
    "write_labelme_json_for_roi",
]

