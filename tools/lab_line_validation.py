"""Exploratory comparison of opening measurements against paired lab values.

All splits hold out a whole product (``image``), including its 14 positions.
The common scale is the arithmetic mean of the 14 position-specific ratios
``mean(lab_mm) / mean(px)`` in the training products.  Each position then gets
its own training-mean ``lab_mm - scale * px`` offset.  The two methods are
fitted separately; these fitted scales are comparison parameters, not an
independent physical camera calibration.

No report files or workbooks are read or modified by this module.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping


POINT_NAMES = tuple(f"{row}{i}" for row in ("U", "L") for i in range(1, 8))
METHOD_FIELDS = {"legacy": "legacy_px", "auto": "auto_px"}
EVALUATION_NOTES = [
    "仅6件且方法在这些图开发，LOO是探索性不是独立验收。",
    "按整件产品留一：每轮用其余5件拟合，用留出的1件预测全部14个点，禁止随机拆分84个点。",
    "每个方法分别重估共同比例和逐点固定补偿，仅用于方法比较，不是独立物理标定。",
    "共同比例=训练集14个点位的[实验室均值/像素均值]的算术平均；补偿=各点训练集mean(实验室-比例×像素)。",
    "same_batch使用同批数据拟合并回算，不能作为泛化或验收结果。",
    "所有有限正数测量均参与探索统计，包括review和有数值的failed；状态保留，不择优、不插补缺值。",
    "缺值会减少对应点的训练样本数；若训练集任一点无有效配对数据，则整轮无法拟合。缺失预测逐条保留，部分结果不代表全量结果。",
]


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _image_order(image: str) -> tuple:
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower())
                 for part in re.split(r"(\d+)", image))


def _normalise(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict], list[str], list[dict]]:
    indexed: dict[tuple[str, str], dict] = {}
    for row_number, record in enumerate(records, start=1):
        image, name = record.get("image"), record.get("name")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"Record {row_number} needs a nonempty image/product identifier")
        if name not in POINT_NAMES:
            raise ValueError(f"Record {row_number} has unknown point {name!r}")
        status = record.get("status", "ok")
        if status not in {"ok", "review", "failed"}:
            raise ValueError(f"Record {row_number} has unknown status {status!r}")
        key = (image, name)
        if key in indexed:
            raise ValueError(f"Duplicate product/point pair: {image}/{name}")
        indexed[key] = {
            "image": image,
            "name": name,
            "status": status,
            "lab_mm": _number(record.get("lab_mm")),
            "legacy_px": _number(record.get("legacy_px")),
            "auto_px": _number(record.get("auto_px")),
            "missing_record": False,
        }
    images = sorted({image for image, _ in indexed}, key=_image_order)
    complete_rows, missing_pairs = [], []
    for image in images:
        for name in POINT_NAMES:
            row = indexed.get((image, name))
            if row is None:
                missing_pairs.append({"image": image, "name": name})
                row = {"image": image, "name": name, "status": "failed",
                       "lab_mm": None, "legacy_px": None, "auto_px": None,
                       "missing_record": True}
            complete_rows.append(row)
    return complete_rows, images, missing_pairs


def _fit(training: list[dict], field: str, training_images: list[str]) -> dict:
    by_point = {
        name: [row for row in training if row["name"] == name
               and row[field] is not None and row["lab_mm"] is not None]
        for name in POINT_NAMES
    }
    counts = {name: len(rows) for name, rows in by_point.items()}
    missing = [name for name in POINT_NAMES if not counts[name]]
    result = {
        "evaluable": False,
        "scale": None,
        "offsets": None,
        "point_ratios": None,
        "training_images": list(training_images),
        "training_product_count": len(training_images),
        "training_count_by_point": counts,
        "training_pair_count": sum(counts.values()),
        "missing_training_points": missing,
        "training_has_missing_pairs": any(count != len(training_images) for count in counts.values()),
        "reason": None,
    }
    if missing:
        result["reason"] = "No usable training pair for: " + ", ".join(missing)
        return result
    try:
        ratios = {name: _mean([row["lab_mm"] for row in rows])
                  / _mean([row[field] for row in rows]) for name, rows in by_point.items()}
        scale = _mean(list(ratios.values()))
        offsets = {name: _mean([row["lab_mm"] - scale * row[field] for row in rows])
                   for name, rows in by_point.items()}
        if not all(math.isfinite(value) for value in [scale, *ratios.values(), *offsets.values()]):
            raise ArithmeticError("Non-finite fitted parameters")
    except (ArithmeticError, ValueError) as exc:
        result["reason"] = f"Training failed: {exc}"
        return result
    result.update(evaluable=True, scale=scale, offsets=offsets, point_ratios=ratios)
    return result


def _predict(rows: list[dict], field: str, fit: dict, held_out_image: str | None) -> list[dict]:
    predictions = []
    for row in rows:
        prediction = {
            "image": row["image"], "name": row["name"], "status": row["status"],
            "missing_record": row["missing_record"], "px": row[field], "lab_mm": row["lab_mm"],
            "predicted_mm": None, "error_mm": None, "abs_error_mm": None,
            "evaluable": False, "reason": None, "held_out_image": held_out_image,
        }
        if not fit["evaluable"]:
            prediction["reason"] = fit["reason"]
        elif row[field] is None:
            prediction["reason"] = "Missing, non-finite or nonpositive pixel measurement"
        elif row["lab_mm"] is None:
            prediction["reason"] = "Missing, non-finite or nonpositive lab reference"
        else:
            value = fit["scale"] * row[field] + fit["offsets"][row["name"]]
            error = value - row["lab_mm"]
            if math.isfinite(value) and math.isfinite(error):
                prediction.update(predicted_mm=value, error_mm=error, abs_error_mm=abs(error), evaluable=True)
            else:
                prediction["reason"] = "Non-finite prediction"
        predictions.append(prediction)
    return predictions


def _metrics(predictions: list[dict]) -> dict:
    usable = [row for row in predictions if row["evaluable"]]
    count = len(usable)
    worst = max(usable, key=lambda row: row["abs_error_mm"]) if usable else None
    errors = [row["error_mm"] for row in usable]
    status_counts = Counter(row["status"] for row in predictions)
    return {
        "evaluable": bool(count),
        "complete": bool(predictions) and count == len(predictions),
        "evaluation_status": "complete" if count and count == len(predictions) else "partial" if count else "not_evaluable",
        "n": count,
        "expected_count": len(predictions),
        "unevaluable_count": len(predictions) - count,
        "coverage": count / len(predictions) if predictions else 0.0,
        "mae": _mean([abs(value) for value in errors]) if count else None,
        "rmse": math.hypot(*errors) / math.sqrt(count) if count else None,
        "bias": _mean(errors) if count else None,
        "max_abs": worst["abs_error_mm"] if worst else None,
        "max_abs_image": worst["image"] if worst else None,
        "max_abs_point": worst["name"] if worst else None,
        "max_abs_at": {key: worst[key] for key in ("image", "name", "error_mm")} if worst else None,
        "review_count": status_counts["review"],
        "failed_count": status_counts["failed"],
        "evaluated_review_count": sum(row["status"] == "review" for row in usable),
        "evaluated_failed_count": sum(row["status"] == "failed" for row in usable),
    }


def _evaluation(predictions: list[dict]) -> dict:
    result = _metrics(predictions)
    result["per_point"] = {
        name: _metrics([row for row in predictions if row["name"] == name]) for name in POINT_NAMES
    }
    result["predictions"] = predictions
    return result


def evaluate_methods(records: Iterable[Mapping[str, Any]]) -> dict:
    """Return JSON-safe same-batch and whole-product-held-out evaluations.

    Required record keys: ``image``, ``name`` (U1..U7/L1..L7), ``legacy_px``,
    ``auto_px``, ``lab_mm`` and ``status`` (ok/review/failed).  Missing numeric
    measurements remain visible as unevaluable predictions.  Review status
    never excludes a finite positive measurement.  Duplicate product/point
    identities raise ValueError because their pairing would be ambiguous.

    The expected experiment has six products; other product counts are
    explicitly flagged, and leave-one-product-out still uses whole products.
    LOO parameters live in ``folds`` because each held-out product has a
    different training fit.  Error signs are prediction minus laboratory.
    """
    rows, images, missing_pairs = _normalise(records)
    counts = Counter(row["status"] for row in rows)
    warnings = []
    if len(images) != 6:
        warnings.append(f"Expected 6 products; received {len(images)}. Each LOO fold trains on {max(len(images) - 1, 0)} products.")
    if missing_pairs:
        warnings.append(f"{len(missing_pairs)} product/point records are missing; they are retained as failed entries.")
    result = {
        "schema_version": 1,
        "notes": list(EVALUATION_NOTES),
        "warnings": warnings,
        "product_count": len(images),
        "images": images,
        "point_names": list(POINT_NAMES),
        "record_count": len(rows),
        "input_record_count": len(rows) - len(missing_pairs),
        "missing_records": missing_pairs,
        "review_count": counts["review"],
        "failed_count": counts["failed"],
        "status_scope": "Status is the supplied record quality flag, preserved for both methods; it is not a pass/fail tolerance judgment.",
        "methods": {},
    }
    for method, field in METHOD_FIELDS.items():
        fit = _fit(rows, field, images)
        same_batch = _evaluation(_predict(rows, field, fit, None))
        same_batch.update({key: value for key, value in fit.items() if key != "evaluable"})
        same_batch["training_evaluable"] = fit["evaluable"]
        folds, all_predictions = [], []
        for image in images:
            training_images = [other for other in images if other != image]
            training = [row for row in rows if row["image"] != image]
            test = [row for row in rows if row["image"] == image]
            fold_fit = _fit(training, field, training_images)
            predictions = _predict(test, field, fold_fit, image)
            fold = _evaluation(predictions)
            fold.update({key: value for key, value in fold_fit.items() if key != "evaluable"})
            fold["held_out_image"] = image
            fold["training_evaluable"] = fold_fit["evaluable"]
            folds.append(fold)
            all_predictions.extend(predictions)
        loo = _evaluation(all_predictions)
        loo["folds"] = folds
        loo["failed_training_fold_count"] = sum(not fold["training_evaluable"] for fold in folds)
        loo["parameter_scope"] = "Each fold has its own training-only scale and offsets; see folds."
        result["methods"][method] = {
            "source_field": field,
            "review_count": counts["review"],
            "failed_count": counts["failed"],
            "missing_pixel_count": sum(row[field] is None for row in rows),
            "missing_lab_count": sum(row["lab_mm"] is None for row in rows),
            "same_batch": same_batch,
            "leave_one_product_out": loo,
        }
    return result
