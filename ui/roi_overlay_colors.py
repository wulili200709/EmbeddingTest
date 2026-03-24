from __future__ import annotations

from collections.abc import Iterable, Mapping

from PySide6 import QtGui


ROI_DEFAULT_COLOR = QtGui.QColor("#22c55e")
ROI_NG_COLOR = QtGui.QColor("#ef4444")


def is_roi_label(label: object) -> bool:
    return str(label or "").strip().lower().startswith("roi")


def is_ng_status(status: object) -> bool:
    return str(status or "").strip().lower() == "ng"


def color_for_roi_status(status: object) -> QtGui.QColor:
    return QtGui.QColor(ROI_NG_COLOR if is_ng_status(status) else ROI_DEFAULT_COLOR)


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
    "color_for_roi_status",
    "is_ng_status",
    "is_roi_label",
    "merge_roi_statuses",
]
