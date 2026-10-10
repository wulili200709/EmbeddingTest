"""Copy the 0916 product into a new release and add 14 opening gap ROIs.

The existing release is read only. Run after build_py312.ps1 completes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dist/LC System V3.0.4/EmbeddingTest/.qr_session/0916"
DEFAULT_DEST = ROOT / "dist-opening-gap/LC System V3.0.5/EmbeddingTest/.qr_session"
CONFIG = ROOT / "config/examples/0916_opening_gaps.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    session_root = args.session_root.resolve()
    if not SOURCE.is_dir() or not session_root.is_dir():
        raise RuntimeError("source product or built release session folder is missing")
    destination = session_root / "0916"
    if destination.exists():
        raise RuntimeError(f"destination already exists: {destination}")
    shutil.copytree(SOURCE, destination)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    roi_half_width = int(config.get("roi_width", 180)) // 2
    recipe_path = destination / "shape/cam1/recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    items_path = destination / "inspection_items.json"
    items = json.loads(items_path.read_text(encoding="utf-8"))
    for item in items:
        item["enabled"] = False
    for region in config["regions"]:
        name = region["name"]
        label = f"opening_{name}"
        x = int(region["center_x"])
        y0 = int(region["upper_y"][0])
        y1 = int(region["lower_y"][1]) + 1
        recipe["reference_regions"].append({
            "reference_label": label,
            "output_label": label,
            "display_name": name,
            "shape_type": "rectangle",
            "points": [[x - roi_half_width, y0], [x + roi_half_width, y1]],
        })
        items.append({
            "item_id": label,
            "display_name": name,
            "camera_id": "cam1",
            "roi_label": label,
            "algorithm_code": "opening_gap_median",
            "algorithm_type": "opening_gap_median",
            "enabled": True,
            "params": {
                "width": 49,
                "upper_y": [int(value) - y0 for value in region["upper_y"]],
                "lower_y": [int(value) - y0 for value in region["lower_y"]],
                "blur_ksize": config["blur_ksize"],
                "min_strength": config["min_strength"],
                "min_valid_ratio": config["min_valid_ratio"],
                "edge_mode": config["edge_mode"],
                "upper_dark_fraction": region.get("upper_dark_fraction", config["upper_dark_fraction"]),
                "lower_dark_fraction": region.get("lower_dark_fraction", config["lower_dark_fraction"]),
                "limit_unit": "px",
            },
        })
    _write_json(recipe_path, recipe)
    _write_json(items_path, items)
    session_path = destination / "session.json"
    session_data = json.loads(session_path.read_text(encoding="utf-8"))
    session_data["test_files"] = list(dict.fromkeys([
        *session_data.get("test_files", []),
        *session_data.get("train_files", []),
    ]))
    _write_json(session_path, session_data)
    products_path = session_root / "products.json"
    products = json.loads(products_path.read_text(encoding="utf-8"))
    products["products"] = list(dict.fromkeys([*products.get("products", []), "0916"]))
    products["current_product"] = "0916"
    _write_json(products_path, products)
    print(f"Configured {len(config['regions'])} opening measurements in {destination}")


if __name__ == "__main__":
    main()
