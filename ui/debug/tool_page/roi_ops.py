"""ROI-related ToolPage helpers."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PySide6 import QtGui

import algorithms.proxy as qr_core

from domain import output_labels_from_line2dup_recipe
from line2dup.core import locator as line2dup_locator
from ui.debug import OverlayShape


def _load_canvas_image(tool_page, path: str) -> None:
    from . import page as page_module

    tool_page.canvas.set_image(path, pixmap=page_module._pixmap_from_path(path))
    tool_page._load_shape_for_label(path, tool_page._current_label())


def _set_status_for_current_image(tool_page, path: str) -> None:
    match_ms = tool_page._line2dup_match_ms_by_image.get(path)
    total_ms = tool_page._line2dup_autogen_ms_by_image.get(path)
    if match_ms is None and total_ms is None:
        return
    parts = [f"当前图像: {os.path.basename(path)}"]
    if match_ms is not None:
        parts.append(f"模板匹配={match_ms:.1f}ms")
    if total_ms is not None:
        parts.append(f"生成ROI={total_ms:.1f}ms")
    tool_page.lbl_status.setText("状态: " + "  ".join(parts))


def _current_label(tool_page) -> str:
    return tool_page.cmb_label.currentText()


def _update_save_label_text(tool_page) -> None:
    label = tool_page._current_label()
    tool_page.btn_save.setText(f"保存标注({label}) -> labelme json")


def _set_overlay_shapes(tool_page, img_path: str, current_label: str) -> None:
    from . import page as page_module

    j = qr_core.labelme_json_of_image(img_path)
    overlays: List[OverlayShape] = []
    visible_roi_labels: Optional[set[str]] = None

    recipe = tool_page.line2dup_recipe
    if recipe is None and os.path.exists(tool_page.session.line2dup_recipe_path):
        try:
            recipe = line2dup_locator.load_recipe_for_product(tool_page.session.product_dir)
            tool_page.line2dup_recipe = recipe
        except Exception:
            recipe = None
    if tool_page.loc_method == "line2dup":
        labels = [str(label).strip() for label in output_labels_from_line2dup_recipe(recipe) if str(label).strip()]
        visible_roi_labels = set(labels) if labels else None

    if tool_page.loc_method == "line2dup" and recipe is not None and recipe.search_points:
        points = [
            (float(pt[0]), float(pt[1]))
            for pt in (recipe.search_points or [])
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        if len(points) >= 2:
            if str(recipe.search_shape_type or "rectangle") == "rectangle" and len(points) == 2:
                (x0, y0), (x1, y1) = points[:2]
                x = int(round(min(x0, x1)))
                y = int(round(min(y0, y1)))
                w = max(1, int(round(abs(x1 - x0))))
                h = max(1, int(round(abs(y1 - y0))))
                overlays.append(
                    OverlayShape(
                        shape_type="rect",
                        xywh=(x, y, w, h),
                        color=QtGui.QColor(0, 0, 255),
                        width=0.5,
                        dash=False,
                    )
                )
            elif len(points) >= 3:
                overlays.append(
                    OverlayShape(
                        shape_type="polygon",
                        points=points,
                        color=QtGui.QColor(0, 0, 255),
                        width=0.5,
                        dash=False,
                    )
                )

    if not os.path.exists(j):
        tool_page.canvas.set_overlays(overlays)
        return

    def add_shape(label: str, color: QtGui.QColor, *, width: int = 2, dash: bool = False) -> None:
        poly_pts = qr_core.try_read_polygon_points_from_labelme(j, label)
        if poly_pts and len(poly_pts) >= 3:
            overlays.append(OverlayShape(shape_type="polygon", points=poly_pts, color=color, width=width, dash=dash))
            return
        xywh = qr_core.try_read_xywh_from_labelme(j, label)
        if xywh:
            overlays.append(OverlayShape(shape_type="rect", xywh=xywh, color=color, width=width, dash=dash))

    seen_labels: set[str] = set()
    for idx, label in enumerate(qr_core.sorted_label_names_from_labelme(j, label_prefix="roi")):
        if visible_roi_labels is not None and label not in visible_roi_labels:
            continue
        if label == current_label:
            continue
        seen_labels.add(label)
        add_shape(label, page_module.ROI_OVERLAY_PALETTE[idx % len(page_module.ROI_OVERLAY_PALETTE)], width=2, dash=False)

    for label, color, dash in [
        ("anchor", QtGui.QColor(0, 255, 255), True),
        ("roi", QtGui.QColor(255, 165, 0), False),
        ("anchor_mask", QtGui.QColor(255, 0, 0), True),
    ]:
        if label == current_label or label in seen_labels:
            continue
        add_shape(label, color, width=2, dash=dash)

    tool_page.canvas.set_overlays(overlays)


def _load_shape_for_label(tool_page, img_path: str, label_name: str) -> None:
    tool_page.canvas.clear_roi()
    j = qr_core.labelme_json_of_image(img_path)
    loaded = False
    if os.path.exists(j):
        poly_pts = qr_core.try_read_polygon_points_from_labelme(j, label_name)
        if poly_pts and len(poly_pts) >= 3:
            tool_page.canvas.set_roi_polygon(poly_pts)
            tool_page.cmb_shape.setCurrentText("polygon")
            loaded = True
        xywh = qr_core.try_read_xywh_from_labelme(j, label_name)
        if xywh:
            tool_page.canvas.set_roi_rect(xywh)
            tool_page.cmb_shape.setCurrentText("rect")
            loaded = True
    tool_page._set_overlay_shapes(img_path, label_name)
    if not loaded:
        tool_page._on_shapes_changed()


def _on_shapes_changed(tool_page) -> None:
    p = tool_page.canvas.image_path()
    if p is None:
        tool_page.btn_save.setEnabled(False)
        return
    st = tool_page.canvas.roi
    ok = (st.shape_type == "rect" and st.xywh is not None) or (st.shape_type == "polygon" and st.points is not None)
    tool_page.btn_save.setEnabled(ok)


def _roi_xywh_from_canvas(tool_page) -> Optional[Tuple[int, int, int, int]]:
    roi = tool_page.canvas.roi_xywh()
    if roi is not None:
        return roi
    p = tool_page.canvas.image_path()
    if p:
        j = qr_core.labelme_json_of_image(p)
        if os.path.exists(j):
            xywh = qr_core.try_read_xywh_from_labelme(j, "roi")
            if xywh:
                return xywh
    return None
