from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from algorithms.labelme import labelme_json_of_image
from safe_io import atomic_write_json, load_json_with_backup


@dataclass(frozen=True)
class RuntimePreviewShape:
    label: str
    shape_type: str
    points: tuple[tuple[float, float], ...]


@dataclass
class RuntimePreviewFrame:
    role: str
    image_bgr: np.ndarray
    source_path: str = ""
    product_dir: str = ""
    camera_role: str = "cam1"
    roi_shapes: tuple[RuntimePreviewShape, ...] = ()
    measurements: tuple[dict, ...] = ()


def build_runtime_preview_frame(
    *,
    role: str,
    image_bgr,
    source_path: str = "",
    product_dir: str = "",
    camera_role: str = "cam1",
    roi_shapes: tuple[RuntimePreviewShape, ...] = (),
    measurements: tuple[dict, ...] = (),
) -> RuntimePreviewFrame:
    image = np.asarray(image_bgr)
    if image.ndim not in {2, 3}:
        raise ValueError(f"unsupported runtime preview image shape: {image.shape!r}")
    copied = np.ascontiguousarray(image.copy())
    if copied.ndim == 3 and copied.shape[2] > 3:
        copied = copied[:, :, :3]
    return RuntimePreviewFrame(
        role=str(role or "").strip() or "cam1",
        image_bgr=copied,
        source_path=str(source_path or "").strip(),
        product_dir=str(product_dir or "").strip(),
        camera_role=str(camera_role or "").strip() or "cam1",
        roi_shapes=tuple(roi_shapes or ()),
        measurements=tuple(dict(item) for item in tuple(measurements or ()) if isinstance(item, dict)),
    )


def read_exported_runtime_preview_shapes(image_path: str) -> tuple[RuntimePreviewShape, ...]:
    path_text = str(image_path or "").strip()
    if not path_text:
        return ()
    json_path = Path(labelme_json_of_image(path_text))
    payload = load_json_with_backup(json_path, default=None)
    if not isinstance(payload, dict):
        return ()

    shapes: list[RuntimePreviewShape] = []
    for shape in payload.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip()
        if not label:
            continue
        points_raw = shape.get("points", [])
        points: list[tuple[float, float]] = []
        for point in points_raw:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                points.append((float(point[0]), float(point[1])))
            except Exception:
                continue
        if not points:
            continue
        shapes.append(
            RuntimePreviewShape(
                label=label,
                shape_type=str(shape.get("shape_type", "rectangle") or "rectangle"),
                points=tuple(points),
            )
        )
    return tuple(shapes)


def read_exported_runtime_preview_measurements(image_path: str) -> tuple[dict, ...]:
    path_text = str(image_path or "").strip()
    if not path_text:
        return ()
    json_path = Path(labelme_json_of_image(path_text))
    payload = load_json_with_backup(json_path, default=None)
    if not isinstance(payload, dict):
        return ()
    flags = payload.get("flags", {})
    if not isinstance(flags, dict):
        return ()
    measurements = flags.get("runtime_measurements", [])
    if not isinstance(measurements, list):
        return ()
    return tuple(dict(item) for item in measurements if isinstance(item, dict))


def load_runtime_preview_shapes(image_path: str) -> tuple[RuntimePreviewShape, ...]:
    # Backward-compatible alias for older callers that still read exported sidecars.
    return read_exported_runtime_preview_shapes(image_path)


def export_runtime_preview_frame(
    frame: RuntimePreviewFrame,
    capture_dir: str | Path,
    *,
    stamp: str | None = None,
) -> RuntimePreviewFrame:
    output_dir = Path(capture_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp_text = str(stamp or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    role = str(frame.role or "").strip() or "cam1"
    image_path = output_dir / f"{stamp_text}_{role}.png"

    image = np.asarray(frame.image_bgr)
    writable = np.ascontiguousarray(image)
    if writable.ndim == 3 and writable.shape[2] > 3:
        writable = writable[:, :, :3]
    if not cv2.imwrite(str(image_path), writable):
        raise RuntimeError(f"failed to save runtime capture: {image_path}")

    _write_runtime_preview_json(image_path, frame)
    return build_runtime_preview_frame(
        role=frame.role,
        image_bgr=frame.image_bgr,
        source_path=str(image_path),
        product_dir=frame.product_dir,
        camera_role=frame.camera_role,
        roi_shapes=frame.roi_shapes,
        measurements=frame.measurements,
    )


def _write_runtime_preview_json(image_path: Path, frame: RuntimePreviewFrame) -> None:
    image = np.asarray(frame.image_bgr)
    height, width = image.shape[:2]
    payload = {
        "version": "5.5.0",
        "flags": {
            "runtime_measurements": [
                dict(item)
                for item in tuple(getattr(frame, "measurements", ()) or ())
                if isinstance(item, dict)
            ],
        },
        "shapes": [
            {
                "label": str(shape.label or "").strip(),
                "points": [
                    [float(point[0]), float(point[1])]
                    for point in tuple(shape.points or ())
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ],
                "group_id": None,
                "shape_type": str(shape.shape_type or "rectangle"),
                "flags": {},
            }
            for shape in tuple(frame.roi_shapes or ())
            if str(shape.label or "").strip()
        ],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }
    json_path = Path(labelme_json_of_image(str(image_path)))
    atomic_write_json(json_path, payload, ensure_ascii=False, indent=2)


__all__ = [
    "export_runtime_preview_frame",
    "RuntimePreviewFrame",
    "RuntimePreviewShape",
    "build_runtime_preview_frame",
    "read_exported_runtime_preview_measurements",
    "read_exported_runtime_preview_shapes",
    "load_runtime_preview_shapes",
]
