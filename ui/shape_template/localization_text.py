"""Render stored localization diagnostics in the current UI language."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

from ui.i18n import tr


_LEGACY_REASONS = {
    "未绑定 ROI": "unused",
    "ROI 绑定的定位组不存在": "missing_group",
    "参考图已更换，请重新确认并保存此组定位区域": "reference_changed",
    "参考图在读取时发生变化": "reference_changed_during_read",
    "无法读取定位组参考图": "reference_read_failed",
    "请先画出有效的矩形定位区域": "invalid_anchor",
    "定位区域必须是轴对齐矩形": "anchor_not_axis_aligned",
    "定位区域超出参考图范围，请重新画区": "anchor_outside_reference",
    "定位区域纹理不足，请选择清晰且稳定的结构": "low_texture",
    "搜索半径必须是 1 到 512 的整数": "invalid_radius",
    "匹配分数和峰值差阈值必须在 0 到 1 之间": "invalid_thresholds",
    "预测定位区域在图像外，没有完整有效的搜索候选": "no_valid_candidate",
    "最佳位置到达搜索边界，请检查定位区域或搜索半径": "search_boundary",
    "最佳位置贴近有效图像边界，无法可靠细化": "image_boundary",
    "全局定位矩阵无效": "invalid_transform",
    "定位组需要有效的 8 位检测图": "invalid_scene",
    "检测图通道数无效": "invalid_scene_channels",
    "全局匹配四角无效": "invalid_global_quad",
    "定位组当前只支持仿射全局定位": "unsupported_global_transform",
    "绑定 ROI 缺少有效名称或坐标": "invalid_bound_roi",
    "绑定 ROI 矩形面积为零": "empty_bound_roi",
    "绑定 ROI 缺少有效的闭合区域": "invalid_bound_roi_shape",
}


def group_reason(record: Mapping[str, Any]) -> str:
    """Translate a raw cached record; never cache the returned language string."""
    code = str(record.get("reason_code") or "")
    params = record.get("reason_params")
    params = dict(params) if isinstance(params, Mapping) else {}
    legacy = str(record.get("reason") or "")
    if not code:
        code = _LEGACY_REASONS.get(legacy, "")
        if not code and legacy.startswith("匹配分数不足"):
            match = re.search(r"\(([\d.]+) < ([\d.]+)\)", legacy)
            if match:
                code, params = "low_score", {"score": float(match[1]), "minimum": float(match[2])}
        if not code and legacy.startswith("存在相似候选"):
            match = re.search(r"峰值差 ([\d.]+) < ([\d.]+)", legacy)
            if match:
                code, params = "ambiguous_peak", {"margin": float(match[1]), "minimum": float(match[2])}
        if not code and legacy == "定位区域至少需要 8 × 8 像素":
            code, params = "anchor_too_small", {"minimum": 8}
        if not code and legacy.startswith("绑定 ROI ") and legacy.endswith(" 超出检测图范围"):
            code, params = "bound_roi_outside_scene", {"label": legacy[len("绑定 ROI "):-len(" 超出检测图范围")]}
    if not code and not record.get("used", True):
        code = "unused"
    if not code and record.get("valid", False):
        code = "ok"
    if code:
        key = "template.groups.reason." + code
        translated = tr(key, **params)
        if translated != key:
            return translated
    return legacy or tr("template.groups.reason.unexpected_error")


def group_error_message(records: Iterable[Mapping[str, Any]]) -> str:
    """Build a localized strict-failure message without changing validity."""
    failures = [record for record in records if record.get("used", True) and not record.get("valid", False)]
    details = []
    for record in failures:
        name = (tr("template.groups.configuration_name") if record.get("id") == "__configuration__"
                else str(record.get("name") or record.get("id") or tr("template.groups.result_group")))
        details.append(f"{name}: {group_reason(record)}")
    return tr("template.groups.failure_message", details="; ".join(details))


__all__ = ["group_reason", "group_error_message"]
