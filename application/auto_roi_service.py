from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Mapping, Optional

import algorithms.lazy_api as qr_core
from common import labelme_io
from common.app_logging import get_app_logger
from shape.core import locator as shape_locator


LOGGER = get_app_logger(__name__)


@dataclass
class AutoRoiIssue:
    message_key: str
    message_args: dict[str, object] = field(default_factory=dict)
    fallback: str = ""


@dataclass
class AutoRoiValidation:
    ok: bool
    labels: List[str] = field(default_factory=list)
    issue: Optional[AutoRoiIssue] = None


def recipe_region_labels(reference_regions: Iterable[Mapping[str, object]] | None) -> set[str]:
    labels = {
        str(region.get("output_label") or region.get("reference_label") or "").strip()
        for region in (reference_regions or [])
        if isinstance(region, Mapping)
    }
    labels.discard("")
    return labels


def missing_roi_files(paths: Iterable[str], labels: Iterable[str]) -> List[str]:
    wanted = [str(label).strip() for label in labels if str(label).strip()] or ["roi"]
    missing: List[str] = []
    for path in paths:
        image_path = str(path or "")
        json_path = labelme_io.labelme_json_of_image(image_path)
        if not os.path.exists(json_path):
            missing.append(image_path)
            continue
        try:
            existing_labels = {
                str(shape.get("label", "") or "").strip()
                for shape in labelme_io.list_shapes_from_labelme(json_path)
                if isinstance(shape, dict)
            }
        except Exception as exc:
            LOGGER.exception("Failed to read ROI labels from %s: %s", json_path, exc)
            missing.append(image_path)
            continue
        if any(label not in existing_labels for label in wanted):
            missing.append(image_path)
    return missing


def validate_autogen_reference(
    *,
    method: str,
    ref_image: str,
    shape_model_path: str = "",
    shape_labels: Iterable[str] = (),
    reference_regions: Iterable[Mapping[str, object]] | None = None,
) -> AutoRoiValidation:
    method = str(method or "").strip()
    ref_image = str(ref_image or "").strip()
    if method == "shape":
        if shape_model_path and not os.path.exists(shape_model_path):
            return AutoRoiValidation(
                ok=False,
                issue=AutoRoiIssue(
                    message_key="template.no_model",
                    fallback="Current product has no template model. Create a template first.",
                ),
            )
        labels = [str(label).strip() for label in shape_labels if str(label).strip()]
        region_labels = recipe_region_labels(reference_regions)
        if (not ref_image or not os.path.exists(ref_image)) and not region_labels:
            return AutoRoiValidation(ok=False, labels=labels, issue=AutoRoiIssue("auto.need_reference_or_saved"))
        missing_labels = [label for label in labels if label not in region_labels]
        if missing_labels:
            ref_json = labelme_io.labelme_json_of_image(ref_image) if ref_image else ""
            if not ref_json or not os.path.exists(ref_json):
                return AutoRoiValidation(
                    ok=False,
                    labels=labels,
                    issue=AutoRoiIssue("auto.missing_reference_json", {"labels": ", ".join(missing_labels)}),
                )
            missing_labels = [
                label for label in missing_labels
                if labelme_io.read_shape_from_labelme(ref_json, label) is None
            ]
            if missing_labels:
                return AutoRoiValidation(
                    ok=False,
                    labels=labels,
                    issue=AutoRoiIssue("auto.missing_reference_roi", {"labels": ", ".join(missing_labels)}),
                )
        return AutoRoiValidation(ok=True, labels=labels)

    if not ref_image or not os.path.exists(ref_image):
        return AutoRoiValidation(ok=False, labels=["roi"], issue=AutoRoiIssue("auto.set_reference_first"))
    ref_json = labelme_io.labelme_json_of_image(ref_image)
    if not os.path.exists(ref_json):
        return AutoRoiValidation(ok=False, labels=["roi"], issue=AutoRoiIssue("auto.reference_missing_json"))
    if labelme_io.try_read_xywh_from_labelme(ref_json, "anchor") is None:
        return AutoRoiValidation(ok=False, labels=["roi"], issue=AutoRoiIssue("auto.reference_missing_anchor"))
    if labelme_io.try_read_xywh_from_labelme(ref_json, "roi") is None:
        return AutoRoiValidation(ok=False, labels=["roi"], issue=AutoRoiIssue("auto.reference_missing_roi"))
    return AutoRoiValidation(ok=True, labels=["roi"])


def run_auto_roi_batch(
    *,
    paths: Iterable[str],
    method: str,
    ref_image: str,
    product_dir: str,
    camera_role: str,
    labels: Iterable[str] = (),
    only_missing: bool = True,
    pre_resolved: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    image_paths = [str(path) for path in paths if str(path)]
    wanted_labels = [str(label).strip() for label in labels if str(label).strip()] or ["roi"]
    todo = list(image_paths) if pre_resolved else (missing_roi_files(image_paths, wanted_labels) if only_missing else list(image_paths))
    if not todo:
        return {
            "ok": 0,
            "errs": [],
            "ok_paths": [],
            "todo_paths": [],
            "timings": {},
            "no_work": True,
        }

    ok = 0
    ok_paths: List[str] = []
    timings: dict[str, float] = {}
    errs: List[str] = []
    total = len(todo)
    method = str(method or "").strip()
    for index, path in enumerate(todo, start=1):
        if progress is not None:
            progress(f"{index}/{total} {os.path.basename(path)}")
        try:
            if method == "shape":
                run = shape_locator.autogen_roi_json_from_shape_timed(
                    tgt_img_path=path,
                    ref_img_path=str(ref_image or ""),
                    product_dir=str(product_dir or ""),
                    camera_role=str(camera_role or "cam1"),
                )
                timings[path] = float(run.total_ms)
            else:
                qr_core.autogen_roi_json_from_reference(
                    tgt_img_path=path,
                    ref_img_path=str(ref_image or ""),
                    method=method,
                    anchor_label="anchor",
                    roi_label="roi",
                )
            ok += 1
            ok_paths.append(path)
        except Exception as exc:
            LOGGER.exception("Auto ROI failed for %s: %s", path, exc)
            errs.append(f"{os.path.basename(path)}: {exc}")

    return {
        "ok": ok,
        "errs": errs,
        "ok_paths": ok_paths,
        "todo_paths": todo,
        "timings": timings,
        "no_work": False,
    }
