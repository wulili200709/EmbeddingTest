"""Batch-test the provisional 50-2 opening regions and export one row per image."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.measurement_opening_gap import (  # noqa: E402
    measure_opening_gap,
    regions_from_params,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test 50-2's provisional 12 notch gaps in all PNG images in a folder."
    )
    parser.add_argument("input_dir", type=Path, help="folder containing the 50-2 PNG images")
    parser.add_argument(
        "--mode", choices=("dark_plateau", "peak", "both"), default="both",
        help="edge rule to test; both saves two separate CSV files",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/examples/50-2_opening_gaps.json",
        help="region and algorithm settings JSON",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="destination folder; default is a new timestamped folder under reports",
    )
    parser.add_argument(
        "--pattern", help="file name glob; default comes from config (the original 50 images)",
    )
    return parser.parse_args()


def _write_edge_viewer(output_dir: Path, images: list[dict], regions: list, image_size: tuple, config: dict) -> Path:
    template = (ROOT / "tools/50_2_edge_viewer_template.html").read_text(encoding="utf-8")
    payload = {
        "images": images,
        "regions": [
            {
                "name": region.name,
                "center_x": region.center_x,
                "upper_y": list(region.upper_y),
                "lower_y": list(region.lower_y),
                "width": region.width,
                "upper_dark_fraction": next(
                    float(item.get("upper_dark_fraction", config["upper_dark_fraction"]))
                    for item in config["regions"] if item["name"] == region.name
                ),
                "lower_dark_fraction": next(
                    float(item.get("lower_dark_fraction", config["lower_dark_fraction"]))
                    for item in config["regions"] if item["name"] == region.name
                ),
            }
            for region in regions
        ],
        "image_size": list(image_size),
    }
    # Keep even unusual file names from ending the inline script element.
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    viewer_path = output_dir / "edge_viewer.html"
    viewer_path.write_text(template.replace("__MEASUREMENT_DATA__", data), encoding="utf-8")
    return viewer_path


def main() -> int:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    config_path = args.config.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"image folder does not exist: {input_dir}")
    if not config_path.is_file():
        raise SystemExit(f"config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pattern = args.pattern or str(config.get("filename_glob", "*.png"))
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"no images matching {pattern!r} found: {input_dir}")
    regions = regions_from_params(config)
    region_settings = {item["name"]: item for item in config["regions"]}
    expected_size = tuple(config["image_size"])
    modes = ("dark_plateau", "peak") if args.mode == "both" else (args.mode,)
    output_dir = (
        args.output_dir.resolve() if args.output_dir
        else ROOT / "reports" / f"50-2_test_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = {mode: [] for mode in modes}
    values = {mode: {region.name: [] for region in regions} for mode in modes}
    errors = {mode: [] for mode in modes}
    viewer_images: list[dict] = []
    for index, path in enumerate(paths, 1):
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            problem = "cannot read image"
        elif (image.shape[1], image.shape[0]) != expected_size:
            problem = f"unexpected image size {image.shape[1]}x{image.shape[0]}"
        else:
            problem = ""
        viewer_image: dict = {"name": path.name, "url": path.resolve().as_uri(), "modes": {}}
        for mode in modes:
            row: dict[str, object] = {"image": path.name}
            viewer_image["modes"][mode] = {}
            if problem:
                errors[mode].append({"image": path.name, "error": problem})
            else:
                for region in regions:
                    local_settings = region_settings[region.name]
                    try:
                        result = measure_opening_gap(
                            image,
                            region,
                            blur_ksize=int(config["blur_ksize"]),
                            min_strength=float(config["min_strength"]),
                            min_valid_ratio=float(config["min_valid_ratio"]),
                            edge_mode=mode,
                            upper_dark_fraction=float(local_settings.get("upper_dark_fraction", config["upper_dark_fraction"])),
                            lower_dark_fraction=float(local_settings.get("lower_dark_fraction", config["lower_dark_fraction"])),
                            include_column_edges=True,
                        )
                    except (RuntimeError, ValueError) as exc:
                        errors[mode].append({"image": path.name, "region": region.name, "error": str(exc)})
                        continue
                    row[region.name] = round(result.distance_px, 3)
                    values[mode][region.name].append(result.distance_px)
                    viewer_image["modes"][mode][region.name] = {
                        "distance_px": round(result.distance_px, 3),
                        "measurement_x": result.measurement_x,
                        "upper_y": round(result.upper_y, 3),
                        "lower_y": round(result.lower_y, 3),
                        "valid_columns": result.valid_columns,
                        "total_columns": result.total_columns,
                        "mad_px": round(result.mad_px, 3),
                        "column_edges": [
                            [x, round(upper, 2), round(lower, 2)]
                            for x, upper, lower in result.column_edges
                        ],
                    }
            rows[mode].append(row)
        viewer_images.append(viewer_image)
        if index % 10 == 0 or index == len(paths):
            print(f"processed {index}/{len(paths)}", flush=True)

    summary: dict[str, object] = {
        "input_dir": str(input_dir),
        "config": str(config_path),
        "filename_glob": pattern,
        "images": len(paths),
        "expected_size": list(expected_size),
        "measurement_unit": "px",
        "warning": "Provisional fixed ROIs; the correct physical edges have not been confirmed.",
        "modes": {},
    }
    for mode in modes:
        csv_path = output_dir / f"50-2_{mode}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["image", *[region.name for region in regions]])
            writer.writeheader()
            writer.writerows(rows[mode])
        stats = {}
        for region in regions:
            readings = values[mode][region.name]
            if not readings:
                continue
            stats[region.name] = {
                "valid_images": len(readings),
                "min_px": round(min(readings), 3),
                "median_px": round(float(np.median(readings)), 3),
                "max_px": round(max(readings), 3),
                "range_px": round(max(readings) - min(readings), 3),
            }
        summary["modes"][mode] = {
            "csv": str(csv_path),
            "measurements_valid": sum(map(len, values[mode].values())),
            "errors": errors[mode],
            "regions": stats,
        }
        print(f"{mode}: {csv_path} ({len(errors[mode])} errors)")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    viewer_path = _write_edge_viewer(output_dir, viewer_images, regions, expected_size, config)
    print(f"summary: {summary_path}")
    print(f"edge viewer: {viewer_path}")
    return 0 if all(not failures for failures in errors.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
