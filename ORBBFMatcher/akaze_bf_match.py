from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np


AKAZE_DESCRIPTOR_TYPES = {
    "MLDB": cv2.AKAZE_DESCRIPTOR_MLDB,
    "UPRIGHT_MLDB": cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT,
}

AKAZE_DIFFUSIVITY = {
    "PM_G1": cv2.KAZE_DIFF_PM_G1,
    "PM_G2": cv2.KAZE_DIFF_PM_G2,
    "WEICKERT": cv2.KAZE_DIFF_WEICKERT,
    "CHARBONNIER": cv2.KAZE_DIFF_CHARBONNIER,
}


@dataclass
class MatchResult:
    keypoints_template: list[cv2.KeyPoint]
    keypoints_scene: list[cv2.KeyPoint]
    good_matches: list[cv2.DMatch]
    homography: np.ndarray | None
    inlier_mask: list[int] | None
    projected_corners: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Template matching with AKAZE + BFMatcher(Hamming)."
    )
    parser.add_argument("--template", required=True, help="Path to the template image.")
    parser.add_argument("--scene", required=True, help="Path to the scene image.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output match visualization image.",
    )
    parser.add_argument(
        "--overlay-output",
        help="Optional path to save the scene image with the detected quadrilateral.",
    )
    parser.add_argument(
        "--descriptor-type",
        choices=sorted(AKAZE_DESCRIPTOR_TYPES),
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
        choices=sorted(AKAZE_DIFFUSIVITY),
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


def load_image(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(path, flags)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def detect_and_compute(
    image: np.ndarray,
    descriptor_type: str,
    descriptor_size: int,
    descriptor_channels: int,
    threshold: float,
    octaves: int,
    octave_layers: int,
    diffusivity: str,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create(
        descriptor_type=AKAZE_DESCRIPTOR_TYPES[descriptor_type],
        descriptor_size=descriptor_size,
        descriptor_channels=descriptor_channels,
        threshold=threshold,
        nOctaves=octaves,
        nOctaveLayers=octave_layers,
        diffusivity=AKAZE_DIFFUSIVITY[diffusivity],
    )
    return akaze.detectAndCompute(gray, None)


def ratio_test_match(
    template_descriptors: np.ndarray | None,
    scene_descriptors: np.ndarray | None,
    ratio_threshold: float,
) -> list[cv2.DMatch]:
    if template_descriptors is None or scene_descriptors is None:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(template_descriptors, scene_descriptors, k=2)

    good_matches: list[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_threshold * second.distance:
            good_matches.append(first)

    good_matches.sort(key=lambda match: match.distance)
    return good_matches


def estimate_homography(
    keypoints_template: list[cv2.KeyPoint],
    keypoints_scene: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    min_matches: int,
    ransac_threshold: float,
    template_shape: tuple[int, int, int],
) -> tuple[np.ndarray | None, list[int] | None, np.ndarray | None]:
    if len(good_matches) < min_matches:
        return None, None, None

    src_pts = np.float32(
        [keypoints_template[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    if homography is None or mask is None:
        return None, None, None

    height, width = template_shape[:2]
    corners = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    ).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(corners, homography)
    return homography, mask.ravel().astype(int).tolist(), projected_corners


def draw_scene_overlay(scene_image: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    overlay = scene_image.copy()
    if corners is not None:
        points = np.int32(corners)
        cv2.polylines(overlay, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
    return overlay


def save_visualization(
    template_image: np.ndarray,
    scene_overlay: np.ndarray,
    keypoints_template: list[cv2.KeyPoint],
    keypoints_scene: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    inlier_mask: list[int] | None,
    output_path: Path,
) -> None:
    draw_params = {
        "matchColor": (0, 255, 0),
        "singlePointColor": (255, 0, 0),
        "flags": cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    }
    if inlier_mask is not None:
        draw_params["matchesMask"] = inlier_mask

    visualization = cv2.drawMatches(
        template_image,
        keypoints_template,
        scene_overlay,
        keypoints_scene,
        matches,
        None,
        **draw_params,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)


def build_result(
    template_image: np.ndarray,
    scene_image: np.ndarray,
    args: argparse.Namespace,
) -> MatchResult:
    keypoints_template, descriptors_template = detect_and_compute(
        template_image,
        args.descriptor_type,
        args.descriptor_size,
        args.descriptor_channels,
        args.threshold,
        args.octaves,
        args.octave_layers,
        args.diffusivity,
    )
    keypoints_scene, descriptors_scene = detect_and_compute(
        scene_image,
        args.descriptor_type,
        args.descriptor_size,
        args.descriptor_channels,
        args.threshold,
        args.octaves,
        args.octave_layers,
        args.diffusivity,
    )

    good_matches = ratio_test_match(
        descriptors_template, descriptors_scene, args.ratio_threshold
    )
    homography, inlier_mask, projected_corners = estimate_homography(
        keypoints_template,
        keypoints_scene,
        good_matches,
        args.min_matches,
        args.ransac_threshold,
        template_image.shape,
    )

    return MatchResult(
        keypoints_template=keypoints_template,
        keypoints_scene=keypoints_scene,
        good_matches=good_matches,
        homography=homography,
        inlier_mask=inlier_mask,
        projected_corners=projected_corners,
    )


def main() -> int:
    args = parse_args()
    try:
        template_image = load_image(args.template)
        scene_image = load_image(args.scene)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    result = build_result(template_image, scene_image, args)
    scene_overlay = draw_scene_overlay(scene_image, result.projected_corners)

    output_path = Path(args.output)
    save_visualization(
        template_image,
        scene_overlay,
        result.keypoints_template,
        result.keypoints_scene,
        result.good_matches,
        result.inlier_mask,
        output_path,
    )

    if args.overlay_output:
        overlay_path = Path(args.overlay_output)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), scene_overlay)

    inlier_count = sum(result.inlier_mask) if result.inlier_mask is not None else 0
    print(f"template keypoints: {len(result.keypoints_template)}")
    print(f"scene keypoints: {len(result.keypoints_scene)}")
    print(f"good matches: {len(result.good_matches)}")
    print(f"inliers: {inlier_count}")
    print(f"visualization saved to: {output_path}")

    if result.homography is None:
        print(
            "Homography was not estimated. Try lowering --ratio-threshold or --threshold, or use images with more texture.",
            file=sys.stderr,
        )
        return 2

    if args.overlay_output:
        print(f"overlay saved to: {Path(args.overlay_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
