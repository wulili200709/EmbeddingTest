"""Run the configured 0916 product through shape location and 14 gap tools."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from application.algorithm_controller import AlgorithmController
from application.product_session import ProductSession
from application.runtime_context import ProductRuntimeContext


def main() -> None:
    session_root = ROOT / "dist-opening-gap/LC System V3.0.5/EmbeddingTest/.qr_session"
    session = ProductSession(str(session_root))
    session.load()
    session.switch_product("0916")
    context = ProductRuntimeContext(session, AlgorithmController())
    product = Path(session.product_dir)
    session_data = json.loads((product / "session.json").read_text(encoding="utf-8"))
    recipe = json.loads((product / "shape/cam1/recipe.json").read_text(encoding="utf-8"))
    expected_roi_count = len(recipe["reference_regions"])
    paths = [product / value for value in session_data["train_files"]]
    names = [f"U{i}" for i in range(1, 8)] + [f"L{i}" for i in range(1, 8)]
    rows: list[dict] = []
    errors: list[dict] = []
    stats: dict[str, list[float]] = {name: [] for name in names}
    minimum_valid = 49
    for index, path in enumerate(paths, 1):
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            errors.append({"image": path.name, "error": "image read failed"})
            continue
        try:
            result = context.predict_items_batch_from_frame(
                image, camera_role="cam1", items=context.inspection_items,
            )
        except Exception as exc:
            errors.append({"image": path.name, "error": str(exc)})
            continue
        row: dict[str, object] = {"image": path.name, "roi_count": len(result.roi_shapes)}
        if len(result.roi_shapes) != expected_roi_count or len(result.rows) != len(names):
            errors.append({"image": path.name, "error": f"rois={len(result.roi_shapes)}, rows={len(result.rows)}"})
        for item in result.rows:
            name = str(item.get("roi_label", "")).removeprefix("opening_")
            if name not in stats:
                errors.append({"image": path.name, "error": f"unknown result {name}"})
                continue
            measurement = item.get("measurement") or {}
            if item.get("pred") != "OK" or item.get("value") is None:
                errors.append({"image": path.name, "region": name, "error": item.get("detail", "NG")})
                continue
            value = float(item["value"])
            row[name] = round(value, 3)
            stats[name].append(value)
            minimum_valid = min(minimum_valid, int(measurement.get("valid_columns", 0)))
        rows.append(row)
        if index % 10 == 0:
            print(f"processed {index}/{len(paths)}", flush=True)
    report = ROOT / "reports"
    report.mkdir(exist_ok=True)
    with (report / "0916_opening_gap_runtime_50.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "roi_count", *names])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "images": len(paths),
        "measurements_expected": len(paths) * len(names),
        "measurements_valid": sum(len(values) for values in stats.values()),
        "minimum_valid_columns": minimum_valid,
        "errors": errors,
        "regions": {
            name: {
                "min_px": round(float(np.min(values)), 3),
                "median_px": round(float(np.median(values)), 3),
                "max_px": round(float(np.max(values)), 3),
            }
            for name, values in stats.items() if values
        },
    }
    (report / "0916_opening_gap_runtime_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
