from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from akaze_bf_match import (
    detect_and_compute,
    draw_scene_overlay,
    estimate_homography,
    load_image,
    ratio_test_match,
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
class PreparedImage:
    path: Path
    image: np.ndarray
    scale: float
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch template matching with AKAZE + BFMatcher(Hamming)."
    )
    parser.add_argument("--template", required=True, help="Path to the template image.")
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
        "--max-dim",
        type=int,
        default=0,
        help="Resize each image so its largest side is at most this value before matching. 0 disables resizing. Default: 0",
    )
    parser.add_argument(
        "--no-write-visuals",
        action="store_true",
        help="Skip writing *_matches.png and *_overlay.png to focus on matching throughput.",
    )
    parser.add_argument(
        "--descriptor-type",
        choices=("MLDB", "UPRIGHT_MLDB"),
        default="MLDB",
        help="Binary AKAZE descriptor type used with Hamming distance. Default: MLDB",
    )
    parser.add_argument(
        "--descriptor-size",
        type=int,
        default=0,
        help="AKAZE descriptor size in bits. 0 lets OpenCV choose. Default: 0",
    )
    parser.add_argument(
        "--descriptor-channels",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="AKAZE descriptor channels. Default: 3",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5e-06,
        help="AKAZE detector response threshold. Smaller values detect more points. Default: 5e-06",
    )
    parser.add_argument(
        "--octaves",
        type=int,
        default=4,
        help="AKAZE octave count. Default: 4",
    )
    parser.add_argument(
        "--octave-layers",
        type=int,
        default=4,
        help="AKAZE layers per octave. Default: 4",
    )
    parser.add_argument(
        "--diffusivity",
        choices=("CHARBONNIER", "PM_G1", "PM_G2", "WEICKERT"),
        default="PM_G2",
        help="AKAZE nonlinear diffusion model. Default: PM_G2",
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


def collect_scene_paths(input_dir: Path, template_path: Path, include_template: bool) -> list[Path]:
    scene_paths: list[Path] = []
    resolved_template = template_path.resolve()
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if not include_template and path.resolve() == resolved_template:
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


def prepare_image(path: Path, args: argparse.Namespace, image_flags: int) -> PreparedImage:
    image = load_image(str(path), image_flags)
    processed_image, scale = resize_to_max_dim(image, args.max_dim)
    keypoints, descriptors = detect_and_compute(
        processed_image,
        args.descriptor_type,
        args.descriptor_size,
        args.descriptor_channels,
        args.threshold,
        args.octaves,
        args.octave_layers,
        args.diffusivity,
    )
    return PreparedImage(
        path=path,
        image=processed_image,
        scale=scale,
        keypoints=keypoints,
        descriptors=descriptors,
    )


def main() -> int:
    process_start = time.perf_counter()
    args = parse_args()
    template_path = Path(args.template)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    scene_paths = collect_scene_paths(input_dir, template_path, args.include_template)
    if not scene_paths:
        print("No scene images found.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    rows: list[dict[str, str | int | float]] = []
    scene_load_seconds_total = 0.0
    scene_match_seconds_total = 0.0
    scene_write_seconds_total = 0.0
    scene_total_seconds_total = 0.0
    image_flags = cv2.IMREAD_GRAYSCALE if args.no_write_visuals else cv2.IMREAD_COLOR

    template_prepare_start = time.perf_counter()
    try:
        prepared_template = prepare_image(template_path, args, image_flags)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    template_prepare_seconds = time.perf_counter() - template_prepare_start

    for scene_path in scene_paths:
        scene_total_start = time.perf_counter()

        scene_load_start = time.perf_counter()
        scene_image = load_image(str(scene_path), image_flags)
        scene_load_seconds = time.perf_counter() - scene_load_start

        scene_match_start = time.perf_counter()
        processed_scene_image, scene_scale = resize_to_max_dim(scene_image, args.max_dim)
        keypoints_scene, descriptors_scene = detect_and_compute(
            processed_scene_image,
            args.descriptor_type,
            args.descriptor_size,
            args.descriptor_channels,
            args.threshold,
            args.octaves,
            args.octave_layers,
            args.diffusivity,
        )
        good_matches = ratio_test_match(
            prepared_template.descriptors,
            descriptors_scene,
            args.ratio_threshold,
        )
        homography, inlier_mask, projected_corners = estimate_homography(
            prepared_template.keypoints,
            keypoints_scene,
            good_matches,
            args.min_matches,
            args.ransac_threshold,
            prepared_template.image.shape,
        )
        scene_match_seconds = time.perf_counter() - scene_match_start

        scene_write_start = time.perf_counter()
        match_output_path = ""
        overlay_output_path = ""
        if not args.no_write_visuals:
            scene_overlay = draw_scene_overlay(processed_scene_image, projected_corners)
            stem = sanitize_stem(scene_path)
            match_output = output_dir / f"{stem}_matches.png"
            overlay_output = output_dir / f"{stem}_overlay.png"
            save_visualization(
                prepared_template.image,
                scene_overlay,
                prepared_template.keypoints,
                keypoints_scene,
                good_matches,
                inlier_mask,
                match_output,
            )
            cv2.imwrite(str(overlay_output), scene_overlay)
            match_output_path = str(match_output)
            overlay_output_path = str(overlay_output)
        scene_write_seconds = time.perf_counter() - scene_write_start

        inlier_count = sum(inlier_mask) if inlier_mask is not None else 0
        homography_ok = homography is not None
        scene_total_seconds = time.perf_counter() - scene_total_start
        scene_load_seconds_total += scene_load_seconds
        scene_match_seconds_total += scene_match_seconds
        scene_write_seconds_total += scene_write_seconds
        scene_total_seconds_total += scene_total_seconds
        rows.append(
            {
                "scene": scene_path.name,
                "status": "ok" if homography_ok else "no_homography",
                "template_keypoints": len(prepared_template.keypoints),
                "scene_keypoints": len(keypoints_scene),
                "good_matches": len(good_matches),
                "inliers": inlier_count,
                "scale": round(scene_scale, 6),
                "scene_load_seconds": round(scene_load_seconds, 6),
                "scene_match_seconds": round(scene_match_seconds, 6),
                "scene_write_seconds": round(scene_write_seconds, 6),
                "scene_total_seconds": round(scene_total_seconds, 6),
                "match_output": match_output_path,
                "overlay_output": overlay_output_path,
            }
        )
        print(
            f"{scene_path.name}: status={'ok' if homography_ok else 'no_homography'}, "
            f"good_matches={len(good_matches)}, inliers={inlier_count}, "
            f"scene_match={scene_match_seconds:.3f}s"
        )

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "scene",
                "status",
                "template_keypoints",
                "scene_keypoints",
                "good_matches",
                "inliers",
                "scale",
                "scene_load_seconds",
                "scene_match_seconds",
                "scene_write_seconds",
                "scene_total_seconds",
                "match_output",
                "overlay_output",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    total_seconds = time.perf_counter() - process_start
    print(f"processed: {len(rows)}")
    print(f"homography estimated: {ok_count}")
    print(f"template prepare seconds: {template_prepare_seconds:.3f}")
    print(f"scene load seconds: {scene_load_seconds_total:.3f}")
    print(f"scene match seconds: {scene_match_seconds_total:.3f}")
    print(f"scene write seconds: {scene_write_seconds_total:.3f}")
    print(f"scene total seconds: {scene_total_seconds_total:.3f}")
    print(f"total seconds: {total_seconds:.3f}")
    print(f"average scene match seconds: {scene_match_seconds_total / len(rows):.3f}")
    print(f"summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
