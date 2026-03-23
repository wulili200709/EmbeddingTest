"""
recipe_manager.py

配方/检测项的轻量辅助层。
当前先提供 line2dup recipe 中 ROI 标签提取能力，后续逐步扩展。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from line2dup.core.recipe import Line2DupRecipe


def inspection_item_specs_from_line2dup_recipe(recipe: Line2DupRecipe | None) -> List[Dict[str, str]]:
    if recipe is None or not recipe.reference_regions:
        return [{"roi_label": "roi", "display_name": "roi"}]

    specs: List[Dict[str, str]] = []
    seen_labels: set[str] = set()
    for region in (recipe.reference_regions or []):
        if not isinstance(region, dict):
            continue
        roi_label = str(region.get("output_label") or region.get("reference_label") or "").strip()
        if not roi_label or roi_label in seen_labels:
            continue
        display_name = str(
            region.get("display_name")
            or region.get("name")
            or region.get("output_label")
            or region.get("reference_label")
            or roi_label
        ).strip() or roi_label
        specs.append({
            "roi_label": roi_label,
            "display_name": display_name,
        })
        seen_labels.add(roi_label)

    return specs or [{"roi_label": "roi", "display_name": "roi"}]


def output_labels_from_line2dup_recipe(recipe: Line2DupRecipe | None) -> List[str]:
    return [spec["roi_label"] for spec in inspection_item_specs_from_line2dup_recipe(recipe)]


def clearable_roi_labels(
    current_labels: Iterable[str],
    existing_labels: Iterable[str],
    *,
    prefer_stale_only: bool = False,
) -> Tuple[List[str], str]:
    """
    Decide which ROI labels should be removed from dataset images.

    When the reference ROI list changed, images may still contain stale labels
    that no longer exist in the current recipe. In that case:
      - prefer_stale_only=True: delete only stale labels to keep valid ROIs.
      - prefer_stale_only=False: delete all ROI-like labels for a full rebuild.

    Returns:
      (labels_to_clear, mode)
      mode in {"stale_only", "all_existing", "current_only"}
    """

    def _normalize(labels: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        seen: set[str] = set()
        for raw in labels:
            label = str(raw).strip()
            if not label or label in {"anchor", "anchor_mask"} or label in seen:
                continue
            normalized.append(label)
            seen.add(label)
        return normalized

    def _sort_key(label: str):
        if label == "roi":
            return (0, 0, label)
        if label.startswith("roi") and label[3:].isdigit():
            return (0, int(label[3:]), label)
        return (1, label.lower(), label)

    current = _normalize(current_labels)
    existing = _normalize(existing_labels)
    current_set = set(current)
    existing_set = set(existing)
    stale = existing_set - current_set

    if stale:
        if prefer_stale_only:
            selected = stale
            mode = "stale_only"
        else:
            selected = existing_set | current_set
            mode = "all_existing"
    else:
        selected = current_set or {"roi"}
        mode = "current_only"

    return sorted(selected, key=_sort_key), mode


__all__ = [
    "inspection_item_specs_from_line2dup_recipe",
    "output_labels_from_line2dup_recipe",
    "clearable_roi_labels",
]
