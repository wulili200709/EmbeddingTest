from __future__ import annotations

from collections.abc import Iterable, Mapping

from PySide6 import QtGui


ROI_DEFAULT_COLOR = QtGui.QColor(55, 155, 55)
ROI_NG_COLOR = QtGui.QColor(220, 30, 30)
ROI_UNLABELED_COLOR = QtGui.QColor(55, 132, 255)
ROI_STROKE_WIDTH = 2.0
SEARCH_REGION_COLOR = QtGui.QColor(0, 0, 255)
SEARCH_REGION_WIDTH = 0.5
ANCHOR_COLOR = QtGui.QColor(0, 255, 255)
ANCHOR_MASK_COLOR = QtGui.QColor(255, 0, 0)
REFERENCE_ROI_COLOR = QtGui.QColor(255, 165, 0)


def is_roi_label(label: object) -> bool:
    return str(label or "").strip().lower().startswith("roi")


def is_ng_status(status: object) -> bool:
    return str(status or "").strip().lower() == "ng"


def is_ok_status(status: object) -> bool:
    return str(status or "").strip().lower() == "ok"


def color_for_roi_status(status: object) -> QtGui.QColor:
    if is_ng_status(status):
        return QtGui.QColor(ROI_NG_COLOR)
    if is_ok_status(status):
        return QtGui.QColor(ROI_DEFAULT_COLOR)
    return QtGui.QColor(ROI_UNLABELED_COLOR)


def overlay_style_for_label(
    label: object,
    *,
    status: object = "",
) -> tuple[QtGui.QColor, float, bool]:
    label_text = str(label or "").strip()
    if is_roi_label(label_text):
        return color_for_roi_status(status), ROI_STROKE_WIDTH, False
    if label_text == "anchor":
        return QtGui.QColor(ANCHOR_COLOR), ROI_STROKE_WIDTH, True
    if label_text == "anchor_mask":
        return QtGui.QColor(ANCHOR_MASK_COLOR), ROI_STROKE_WIDTH, True
    if label_text == "roi":
        return QtGui.QColor(REFERENCE_ROI_COLOR), ROI_STROKE_WIDTH, False
    return QtGui.QColor(ROI_DEFAULT_COLOR), ROI_STROKE_WIDTH, False


def search_region_style() -> tuple[QtGui.QColor, float, bool]:
    return QtGui.QColor(SEARCH_REGION_COLOR), SEARCH_REGION_WIDTH, False


def merge_roi_statuses(
    rows: Iterable[Mapping[str, object]],
    *,
    camera_id: object = "",
) -> dict[str, str]:
    wanted_camera = str(camera_id or "").strip()
    merged: dict[str, str] = {}
    for row in rows:
        label = str(row.get("roi_label", "") or "").strip()
        if not is_roi_label(label):
            continue
        row_camera = str(row.get("camera_id", "") or "").strip()
        if wanted_camera and row_camera != wanted_camera:
            continue
        status = str(row.get("status_kind", "") or "").strip().lower()
        if not status:
            continue
        current = merged.get(label, "")
        if current == "ng":
            continue
        merged[label] = "ng" if status == "ng" else status
    return merged


__all__ = [
    "ROI_DEFAULT_COLOR",
    "ROI_NG_COLOR",
    "ROI_UNLABELED_COLOR",
    "ROI_STROKE_WIDTH",
    "SEARCH_REGION_COLOR",
    "SEARCH_REGION_WIDTH",
    "ANCHOR_COLOR",
    "ANCHOR_MASK_COLOR",
    "REFERENCE_ROI_COLOR",
    "color_for_roi_status",
    "is_ng_status",
    "is_ok_status",
    "is_roi_label",
    "merge_roi_statuses",
    "overlay_style_for_label",
    "search_region_style",
]
