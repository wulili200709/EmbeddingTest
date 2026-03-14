from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

from sift_bf_match import (
    create_sift,
    draw_scene_overlay,
    estimate_homography,
    get_template_outline,
    load_image,
    load_template_image_and_mask,
    prepare_template_visualization,
    save_visualization,
)


SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".pbm",
    ".pgm",
    ".ppm",
    ".sr",
    ".ras",
    ".tiff",
    ".tif",
    ".webp",
}


@dataclass
class TemplatePreparation:
    image: np.ndarray
    mask: np.ndarray | None
    mask_source: str | None
    outline: np.ndarray
    visualization: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None
    load_ms: float
    outline_ms: float
    gray_ms: float
    feature_ms: float


@dataclass
class SceneTiming:
    scene_load_ms: float = 0.0
    resize_ms: float = 0.0
    gray_ms: float = 0.0
    feature_ms: float = 0.0
    knn_match_ms: float = 0.0
    ratio_filter_ms: float = 0.0
    homography_ms: float = 0.0
    overlay_draw_ms: float = 0.0
    save_match_ms: float = 0.0
    save_overlay_ms: float = 0.0
    total_ms: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch template matching with SIFT + BFMatcher(L2)."
    )
    parser.add_argument("--template", required=True, help="Path to the template image.")
    parser.add_argument(
        "--template-mask",
        help=(
            "Optional binary mask for the template. If omitted, the script first tries "
            "the template alpha channel, then <template_stem>_mask.png."
        ),
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing scene images to match against the template.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where batch outputs will be written.",
    )
    parser.add_argument(
        "--include-template",
        action="store_true",
        help="Also run matching for the template image itself if it is inside --input-dir.",
    )
    parser.add_argument(
        "--include-mask-images",
        action="store_true",
        help="Include files whose names look like masks, such as *_mask.png.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=0,
        help="Resize each scene image so its largest side is at most this value. 0 disables resizing. Default: 0",
    )
    parser.add_argument(
        "--no-write-visuals",
        action="store_true",
        help="Skip writing *_matches.png and *_overlay.png to focus on timing.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Maximum SIFT keypoints. 0 means unlimited. Default: 0",
    )
    parser.add_argument(
        "--n-octave-layers",
        type=int,
        default=3,
        help="Number of layers in each octave. Default: 3",
    )
    parser.add_argument(
        "--contrast-threshold",
        type=float,
        default=0.04,
        help="SIFT contrast threshold. Lower values detect more points. Default: 0.04",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=10.0,
        help="SIFT edge threshold. Higher values keep more edge-like features. Default: 10.0",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.6,
        help="Gaussian sigma applied to octave 0. Default: 1.6",
    )
    parser.add_argument(
        "--enable-precise-upscale",
        action="store_true",
        help="Enable OpenCV SIFT precise upscale when available.",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.75,
        help="Lowe ratio threshold for BFMatcher.knnMatch. Default: 0.75",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=10,
        help="Minimum good matches required to estimate homography. Default: 10",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold for homography. Default: 5.0",
    )
    return parser.parse_args()


def collect_scene_paths(
    input_dir: Path,
    template_path: Path,
    template_mask_path: str | None,
    include_template: bool,
    include_mask_images: bool,
) -> list[Path]:
    scene_paths: list[Path] = []
    resolved_template = template_path.resolve()
    resolved_template_mask = (
        Path(template_mask_path).resolve() if template_mask_path else None
    )

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if not include_template and path.resolve() == resolved_template:
            continue
        if resolved_template_mask is not None and path.resolve() == resolved_template_mask:
            continue
        if not include_mask_images and path.stem.lower().endswith("_mask"):
            continue
        scene_paths.append(path)
    return scene_paths


def sanitize_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def resize_to_max_dim(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    if max_dim <= 0:
        return image, 1.0

    height, width = image.shape[:2]
    scale = min(float(max_dim) / max(height, width), 1.0)
    if scale == 1.0:
        return image, scale

    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def extract_sift_features(
    image: np.ndarray,
    sift: cv2.Feature2D,
    mask: np.ndarray | None = None,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None, float, float]:
    gray_start = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_ms = (time.perf_counter() - gray_start) * 1000.0

    feature_start = time.perf_counter()
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    feature_ms = (time.perf_counter() - feature_start) * 1000.0
    if keypoints is None:
        return [], None, gray_ms, feature_ms
    return list(keypoints), descriptors, gray_ms, feature_ms


def match_descriptors(
    template_descriptors: np.ndarray | None,
    scene_descriptors: np.ndarray | None,
    ratio_threshold: float,
) -> tuple[list[cv2.DMatch], float, float]:
    if template_descriptors is None or scene_descriptors is None:
        return [], 0.0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    knn_start = time.perf_counter()
    knn_matches = matcher.knnMatch(template_descriptors, scene_descriptors, k=2)
    knn_match_ms = (time.perf_counter() - knn_start) * 1000.0

    ratio_start = time.perf_counter()
    good_matches: list[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_threshold * second.distance:
            good_matches.append(first)
    good_matches.sort(key=lambda match: match.distance)
    ratio_filter_ms = (time.perf_counter() - ratio_start) * 1000.0
    return good_matches, knn_match_ms, ratio_filter_ms


def prepare_template(
    template_path: str,
    template_mask_path: str | None,
    sift: cv2.Feature2D,
    args: argparse.Namespace,
) -> TemplatePreparation:
    load_start = time.perf_counter()
    template_image, template_mask, template_mask_source = load_template_image_and_mask(
        template_path,
        template_mask_path,
    )
    load_ms = (time.perf_counter() - load_start) * 1000.0

    outline_start = time.perf_counter()
    outline = get_template_outline(template_image.shape, template_mask)
    outline_ms = (time.perf_counter() - outline_start) * 1000.0

    keypoints, descriptors, gray_ms, feature_ms = extract_sift_features(
        template_image,
        sift,
        template_mask,
    )
    visualization = prepare_template_visualization(template_image, template_mask)
    return TemplatePreparation(
        image=template_image,
        mask=template_mask,
        mask_source=template_mask_source,
        outline=outline,
        visualization=visualization,
        keypoints=keypoints,
        descriptors=descriptors,
        load_ms=load_ms,
        outline_ms=outline_ms,
        gray_ms=gray_ms,
        feature_ms=feature_ms,
    )


def compute_stage_stats(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        "avg_ms": round(sum(values) / len(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def process_scene(
    scene_path: Path,
    prepared_template: TemplatePreparation,
    sift: cv2.Feature2D,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    timings = SceneTiming()
    scene_total_start = time.perf_counter()

    scene_load_start = time.perf_counter()
    scene_image = load_image(str(scene_path))
    timings.scene_load_ms = (time.perf_counter() - scene_load_start) * 1000.0

    resize_start = time.perf_counter()
    processed_scene_image, scene_scale = resize_to_max_dim(scene_image, args.max_dim)
    timings.resize_ms = (time.perf_counter() - resize_start) * 1000.0

    keypoints_scene, descriptors_scene, timings.gray_ms, timings.feature_ms = (
        extract_sift_features(processed_scene_image, sift)
    )

    good_matches, timings.knn_match_ms, timings.ratio_filter_ms = match_descriptors(
        prepared_template.descriptors,
        descriptors_scene,
        args.ratio_threshold,
    )

    homography_start = time.perf_counter()
    homography, inlier_mask, projected_outline = estimate_homography(
        prepared_template.keypoints,
        keypoints_scene,
        good_matches,
        args.min_matches,
        args.ransac_threshold,
        prepared_template.outline,
    )
    timings.homography_ms = (time.perf_counter() - homography_start) * 1000.0

    match_output_path = ""
    overlay_output_path = ""
    if not args.no_write_visuals:
        overlay_start = time.perf_counter()
        scene_overlay = draw_scene_overlay(processed_scene_image, projected_outline)
        timings.overlay_draw_ms = (time.perf_counter() - overlay_start) * 1000.0

        stem = sanitize_stem(scene_path)
        match_output = output_dir / f"{stem}_matches.png"
        overlay_output = output_dir / f"{stem}_overlay.png"

        save_match_start = time.perf_counter()
        save_visualization(
            prepared_template.visualization,
            scene_overlay,
            prepared_template.keypoints,
            keypoints_scene,
            good_matches,
            inlier_mask,
            match_output,
        )
        timings.save_match_ms = (time.perf_counter() - save_match_start) * 1000.0

        save_overlay_start = time.perf_counter()
        cv2.imwrite(str(overlay_output), scene_overlay)
        timings.save_overlay_ms = (time.perf_counter() - save_overlay_start) * 1000.0

        match_output_path = str(match_output)
        overlay_output_path = str(overlay_output)

    timings.total_ms = (time.perf_counter() - scene_total_start) * 1000.0
    inlier_count = sum(inlier_mask) if inlier_mask is not None else 0
    homography_ok = homography is not None

    row: dict[str, object] = {
        "scene": scene_path.name,
        "status": "ok" if homography_ok else "no_homography",
        "scale": round(scene_scale, 6),
        "template_mask": prepared_template.mask_source or "none",
        "template_keypoints": len(prepared_template.keypoints),
        "scene_keypoints": len(keypoints_scene),
        "good_matches": len(good_matches),
        "inliers": inlier_count,
        "scene_load_ms": round(timings.scene_load_ms, 3),
        "resize_ms": round(timings.resize_ms, 3),
        "gray_ms": round(timings.gray_ms, 3),
        "feature_ms": round(timings.feature_ms, 3),
        "knn_match_ms": round(timings.knn_match_ms, 3),
        "ratio_filter_ms": round(timings.ratio_filter_ms, 3),
        "homography_ms": round(timings.homography_ms, 3),
        "overlay_draw_ms": round(timings.overlay_draw_ms, 3),
        "save_match_ms": round(timings.save_match_ms, 3),
        "save_overlay_ms": round(timings.save_overlay_ms, 3),
        "total_ms": round(timings.total_ms, 3),
        "match_output": match_output_path,
        "overlay_output": overlay_output_path,
    }
    return row


def write_summary_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    rows: list[dict[str, object]],
    prepared_template: TemplatePreparation,
) -> dict[str, object]:
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    summary = {
        "images_processed": len(rows),
        "homography_estimated": ok_count,
        "homography_failed": len(rows) - ok_count,
        "success_rate": round(ok_count / len(rows), 4),
        "template": {
            "template_keypoints": len(prepared_template.keypoints),
            "mask_source": prepared_template.mask_source or "none",
            "load_ms": round(prepared_template.load_ms, 3),
            "outline_ms": round(prepared_template.outline_ms, 3),
            "gray_ms": round(prepared_template.gray_ms, 3),
            "feature_ms": round(prepared_template.feature_ms, 3),
        },
        "stage_stats": {
            "scene_load_ms": compute_stage_stats(rows, "scene_load_ms"),
            "resize_ms": compute_stage_stats(rows, "resize_ms"),
            "gray_ms": compute_stage_stats(rows, "gray_ms"),
            "feature_ms": compute_stage_stats(rows, "feature_ms"),
            "knn_match_ms": compute_stage_stats(rows, "knn_match_ms"),
            "ratio_filter_ms": compute_stage_stats(rows, "ratio_filter_ms"),
            "homography_ms": compute_stage_stats(rows, "homography_ms"),
            "overlay_draw_ms": compute_stage_stats(rows, "overlay_draw_ms"),
            "save_match_ms": compute_stage_stats(rows, "save_match_ms"),
            "save_overlay_ms": compute_stage_stats(rows, "save_overlay_ms"),
            "total_ms": compute_stage_stats(rows, "total_ms"),
        },
        "slowest_total_ms": sorted(
            (
                {
                    "scene": row["scene"],
                    "status": row["status"],
                    "total_ms": row["total_ms"],
                }
                for row in rows
            ),
            key=lambda item: float(item["total_ms"]),
            reverse=True,
        )[:5],
        "best_inliers": sorted(
            (
                {
                    "scene": row["scene"],
                    "good_matches": row["good_matches"],
                    "inliers": row["inliers"],
                }
                for row in rows
            ),
            key=lambda item: (int(item["inliers"]), int(item["good_matches"])),
            reverse=True,
        )[:5],
    }
    return summary


def main() -> int:
    process_start = time.perf_counter()
    args = parse_args()
    template_path = Path(args.template)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    scene_paths = collect_scene_paths(
        input_dir,
        template_path,
        args.template_mask,
        args.include_template,
        args.include_mask_images,
    )
    if not scene_paths:
        print("No scene images found.", file=sys.stderr)
        return 1

    try:
        sift = create_sift(args)
        prepared_template = prepare_template(args.template, args.template_mask, sift, args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    for scene_path in scene_paths:
        row = process_scene(scene_path, prepared_template, sift, args, output_dir)
        rows.append(row)
        print(
            f"{row['scene']}: status={row['status']}, "
            f"keypoints={row['scene_keypoints']}, good_matches={row['good_matches']}, "
            f"inliers={row['inliers']}, total={float(row['total_ms']):.3f}ms"
        )

    summary_csv_path = output_dir / "summary.csv"
    write_summary_csv(summary_csv_path, rows)

    summary = build_summary(rows, prepared_template)
    summary["elapsed_ms"] = round((time.perf_counter() - process_start) * 1000.0, 3)
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"processed: {summary['images_processed']}")
    print(f"homography estimated: {summary['homography_estimated']}")
    print(f"success rate: {summary['success_rate']:.2%}")
    print(f"template keypoints: {len(prepared_template.keypoints)}")
    print(f"summary csv: {summary_csv_path}")
    print(f"summary json: {summary_json_path}")
    print(f"elapsed ms: {summary['elapsed_ms']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
