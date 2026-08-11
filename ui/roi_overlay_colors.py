from __future__ import annotations

from collections.abc import Iterable, Mapping

from PySide6 import QtGui


ROI_DEFAULT_COLOR = QtGui.QColor(55, 155, 55)
ROI_NG_COLOR = QtGui.QColor(220, 30, 30)
ROI_PASS_COLOR = QtGui.QColor(47, 143, 70)
ROI_UNLABELED_COLOR = QtGui.QColor(55, 132, 255)
ROI_DISABLED_COLOR = QtGui.QColor(140, 140, 140)
ROI_STROKE_WIDTH = 2.0
SEARCH_REGION_COLOR = QtGui.QColor(0, 0, 255)
SEARCH_REGION_WIDTH = 0.5
ANCHOR_COLOR = QtGui.QColor(0, 255, 255)
ANCHOR_MASK_COLOR = QtGui.QColor(255, 0, 0)
REFERENCE_ROI_COLOR = QtGui.QColor(255, 165, 0)

_DISTANCE_HELPER_KEYS = {
    "line_distance": ("line_a_item_id", "line_b_item_id"),
    "line_distance_ref_normal": ("line_a_item_id", "line_b_item_id"),
    "center_distance": ("center_a_item_id", "center_b_item_id"),
}


def is_roi_label(label: object) -> bool:
    return str(label or "").strip().lower().startswith("roi")


def is_ng_status(status: object) -> bool:
    return str(status or "").strip().lower() == "ng"


def is_ok_status(status: object) -> bool:
    return str(status or "").strip().lower() == "ok"


def is_pass_status(status: object) -> bool:
    return str(status or "").strip().lower() == "pass"


def is_disabled_status(status: object) -> bool:
    return str(status or "").strip().lower() in {"disabled", "inactive"}


def color_for_roi_status(status: object) -> QtGui.QColor:
    if is_disabled_status(status):
        return QtGui.QColor(ROI_DISABLED_COLOR)
    if is_ng_status(status):
        return QtGui.QColor(ROI_NG_COLOR)
    if is_pass_status(status):
        return QtGui.QColor(ROI_PASS_COLOR)
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
    camera_rows = [
        row
        for row in rows
        if not wanted_camera or str(row.get("camera_id", "") or "").strip() == wanted_camera
    ]
    merged: dict[str, str] = {}

    def merge_status(label: str, status: str) -> None:
        current = merged.get(label, "")
        if current == "ng":
            return
        merged[label] = "ng" if status == "ng" else status

    roi_label_by_item_id = {
        str(row.get("item_id", "") or "").strip(): str(row.get("roi_label", "") or "").strip()
        for row in camera_rows
        if str(row.get("item_id", "") or "").strip()
        and is_roi_label(row.get("roi_label", ""))
    }
    for row in camera_rows:
        label = str(row.get("roi_label", "") or "").strip()
        if not is_roi_label(label):
            continue
        status = str(row.get("status_kind", "") or "").strip().lower()
        if not status:
            continue
        merge_status(label, status)

    # A distance result owns the visual status of the ROI tools that feed it.
    # The helper line/center detection may succeed while the measured dimension
    # is out of tolerance; in that case both related ROI boxes must be red.
    for row in camera_rows:
        algorithm = str(row.get("algorithm_code", "") or "").strip()
        helper_keys = _DISTANCE_HELPER_KEYS.get(algorithm)
        if not helper_keys:
            continue
        status = str(row.get("status_kind", "") or "").strip().lower()
        if not status:
            continue
        params = row.get("params", {})
        if not isinstance(params, Mapping):
            continue
        for key in helper_keys:
            helper_id = str(params.get(key, "") or "").strip()
            label = roi_label_by_item_id.get(helper_id, "")
            if label:
                merge_status(label, status)
    return merged


__all__ = [
    "ROI_DEFAULT_COLOR",
    "ROI_NG_COLOR",
    "ROI_PASS_COLOR",
    "ROI_UNLABELED_COLOR",
    "ROI_DISABLED_COLOR",
    "ROI_STROKE_WIDTH",
    "SEARCH_REGION_COLOR",
    "SEARCH_REGION_WIDTH",
    "ANCHOR_COLOR",
    "ANCHOR_MASK_COLOR",
    "REFERENCE_ROI_COLOR",
    "color_for_roi_status",
    "is_ng_status",
    "is_ok_status",
    "is_pass_status",
    "is_disabled_status",
    "is_roi_label",
    "merge_roi_statuses",
    "overlay_style_for_label",
    "search_region_style",
]
