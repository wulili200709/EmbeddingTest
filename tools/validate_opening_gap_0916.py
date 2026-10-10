"""Validate the opening gap algorithm against the 0916 product's 50 images."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.measurement import measure_opening_gap_from_array
from algorithms.measurement_opening_gap import regions_from_params


def main() -> None:
    config = json.loads((PROJECT_ROOT / "config/examples/0916_opening_gaps.json").read_text(encoding="utf-8"))
    product = PROJECT_ROOT / "dist/LC System V3.0.4/EmbeddingTest/.qr_session/0916"
    session = json.loads((product / "session.json").read_text(encoding="utf-8"))
    paths = [product / value for value in session["train_files"]]
    regions = regions_from_params(config)
    region_configs = {item["name"]: item for item in config["regions"]}
    roi_half_width = int(config.get("roi_width", 180)) // 2
    output = PROJECT_ROOT / "reports"
    output.mkdir(exist_ok=True)
    rows = []
    details = []
    errors = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            errors.append({"image": path.name, "error": "cannot read image"})
            continue
        if [image.shape[1], image.shape[0]] != config["image_size"]:
            errors.append({"image": path.name, "error": f"unexpected image size {image.shape[:2]}"})
            continue
        row = {"image": path.name}
        for region in regions:
            region_config = region_configs[region.name]
            try:
                y0 = region.upper_y[0]
                shape = {
                    "shape_type": "rectangle",
                    "points": [[region.center_x - roi_half_width, y0], [region.center_x + roi_half_width, region.lower_y[1] + 1]],
                }
                result = measure_opening_gap_from_array(
                    image,
                    shape_by_label={region.name: shape},
                    preferred_label=region.name,
                    params={
                        "width": region.width,
                        "upper_y": [value - y0 for value in region.upper_y],
                        "lower_y": [value - y0 for value in region.lower_y],
                        "blur_ksize": config["blur_ksize"],
                        "min_strength": config["min_strength"],
                        "min_valid_ratio": config["min_valid_ratio"],
                        "edge_mode": config["edge_mode"],
                        "upper_dark_fraction": region_config.get("upper_dark_fraction", config["upper_dark_fraction"]),
                        "lower_dark_fraction": region_config.get("lower_dark_fraction", config["lower_dark_fraction"]),
                    },
                )
                row[region.name] = round(result.distance_px, 3)
                details.append({"image": path.name, **result.to_dict()})
            except (ValueError, RuntimeError) as exc:
                row[region.name] = ""
                errors.append({"image": path.name, "region": region.name, "error": str(exc)})
        rows.append(row)
    with (output / "0916_opening_gap_50.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", *[region.name for region in regions]])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "images": len(paths),
        "measurements_expected": len(paths) * len(regions),
        "measurements_valid": len(details),
        "errors": errors,
        "regions": {},
    }
    for region in regions:
        group = [item for item in details if item["name"] == region.name]
        if not group:
            continue
        values = np.asarray([item["distance_px"] for item in group])
        summary["regions"][region.name] = {
            "min_px": round(float(values.min()), 3),
            "median_px": round(float(np.median(values)), 3),
            "max_px": round(float(values.max()), 3),
            "min_valid_columns": min(item["valid_columns"] for item in group),
            "max_column_mad_px": round(max(item["mad_px"] for item in group), 3),
        }
    (output / "0916_opening_gap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
