from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import numpy as np


DETECTOR_BEBLID_SCALE = {
    "orb": 1.0,
    "brisk": 5.0,
}


@dataclass
class MatchResult:
    keypoints_template: list[cv2.KeyPoint]
    keypoints_scene: list[cv2.KeyPoint]
    good_matches: list[cv2.DMatch]
    homography: np.ndarray | None
    inlier_mask: list[int] | None
    projected_outline: np.ndarray | None
    detector_name: str
    beblid_scale_factor: float
    beblid_bits: int
    template_mask_source: str | None


@dataclass
class MatchTimings:
    template_load_ms: float = 0.0
    template_prepare_ms: float = 0.0
    scene_load_ms: float = 0.0
    scene_feature_ms: float = 0.0
    descriptor_match_ms: float = 0.0
    homography_ms: float = 0.0
    save_output_ms: float = 0.0

    @property
    def template_setup_ms(self) -> float:
        return self.template_load_ms + self.template_prepare_ms

    @property
    def matching_time_ms(self) -> float:
        return (
            self.scene_load_ms
            + self.scene_feature_ms
            + self.descriptor_match_ms
            + self.homography_ms
        )

    @property
    def pipeline_total_ms(self) -> float:
        return self.template_setup_ms + self.matching_time_ms + self.save_output_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Template matching with ORB/BRISK keypoints + BEBLID descriptors."
    )
    parser.add_argument("--template", required=True, help="Path to the template image.")
    parser.add_argument(
        "--template-mask",
        help=(
            "Optional binary mask for the template. If omitted, the script first tries "
            "the template alpha channel, then <template_stem>_mask.png."
        ),
    )
    parser.add_argument("--scene", required=True, help="Path to the scene image.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output match visualization image.",
    )
    parser.add_argument(
        "--overlay-output",
        help="Optional path to save the scene image with the detected outline.",
    )
    parser.add_argument(
        "--detector",
        choices=sorted(DETECTOR_BEBLID_SCALE),
        default="orb",
        help="Keypoint detector to use. Default: orb",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=1500,
        help="Maximum ORB keypoints. Only used when --detector orb. Default: 1500",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.2,
        help="ORB pyramid scale factor. Only used when --detector orb. Default: 1.2",
    )
    parser.add_argument(
        "--nlevels",
        type=int,
        default=8,
        help="ORB pyramid levels. Only used when --detector orb. Default: 8",
    )
    parser.add_argument(
        "--edge-threshold",
        type=int,
        default=5,
        help="ORB edge threshold. Only used when --detector orb. Default: 5",
    )
    parser.add_argument(
        "--fast-threshold",
        type=int,
        default=5,
        help="ORB FAST threshold. Only used when --detector orb. Default: 5",
    )
    parser.add_argument(
        "--brisk-thresh",
        type=int,
        default=5,
        help="BRISK AGAST threshold. Only used when --detector brisk. Default: 5",
    )
    parser.add_argument(
        "--brisk-octaves",
        type=int,
        default=0,
        help="BRISK octaves. Only used when --detector brisk. Default: 0",
    )
    parser.add_argument(
        "--brisk-pattern-scale",
        type=float,
        default=1.0,
        help="BRISK pattern scale. Only used when --detector brisk. Default: 1.0",
    )
    parser.add_argument(
        "--beblid-scale-factor",
        type=float,
        default=None,
        help=(
            "Override BEBLID sampling window scale. By default it is chosen automatically "
            "for the selected detector."
        ),
    )
    parser.add_argument(
        "--beblid-bits",
        type=int,
        choices=(256, 512),
        default=512,
        help="BEBLID descriptor size. Default: 512",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.8,
        help="Lowe ratio threshold for BFMatcher.knnMatch. Default: 0.8",
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
    parser.add_argument(
        "--print-timing",
        action="store_true",
        help="Print machine-readable matching timing in milliseconds.",
    )
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def load_template_image_and_mask(
    template_path_text: str, template_mask_path_text: str | None
) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    template_path = Path(template_path_text)
    raw_template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if raw_template is None:
        raise FileNotFoundError(f"Unable to read image: {template_path}")

    template_mask: np.ndarray | None = None
    template_mask_source: str | None = None

    if raw_template.ndim == 2:
        template_image = cv2.cvtColor(raw_template, cv2.COLOR_GRAY2BGR)
    elif raw_template.shape[2] == 4:
        template_image = cv2.cvtColor(raw_template, cv2.COLOR_BGRA2BGR)
        alpha = raw_template[:, :, 3]
        if np.any(alpha < 255):
            template_mask = alpha
            template_mask_source = f"{template_path} (alpha)"
    else:
        template_image = raw_template

    if template_mask_path_text:
        explicit_mask = cv2.imread(template_mask_path_text, cv2.IMREAD_GRAYSCALE)
        if explicit_mask is None:
            raise FileNotFoundError(f"Unable to read template mask: {template_mask_path_text}")
        template_mask = explicit_mask
        template_mask_source = str(Path(template_mask_path_text))
    elif template_mask is None:
        auto_mask_path = template_path.with_name(f"{template_path.stem}_mask.png")
        if auto_mask_path.exists():
            auto_mask = cv2.imread(str(auto_mask_path), cv2.IMREAD_GRAYSCALE)
            if auto_mask is None:
                raise FileNotFoundError(f"Unable to read template mask: {auto_mask_path}")
            template_mask = auto_mask
            template_mask_source = str(auto_mask_path)

    if template_mask is not None:
        if template_mask.shape[:2] != template_image.shape[:2]:
            raise RuntimeError("Template mask size does not match the template image size")
        _, template_mask = cv2.threshold(template_mask, 1, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(template_mask) == 0:
            raise RuntimeError("Template mask is empty")

    return template_image, template_mask, template_mask_source


def ensure_beblid_available() -> None:
    if not hasattr(cv2, "xfeatures2d") or not hasattr(
        cv2.xfeatures2d, "BEBLID_create"
    ):
        raise RuntimeError(
            "BEBLID is unavailable in this environment. Install opencv-contrib-python "
            "inside the active virtual environment."
        )


def create_detector(args: argparse.Namespace) -> cv2.Feature2D:
    if args.detector == "orb":
        return cv2.ORB_create(
            nfeatures=args.max_features,
            scaleFactor=args.scale_factor,
            nlevels=args.nlevels,
            edgeThreshold=args.edge_threshold,
            fastThreshold=args.fast_threshold,
        )
    return cv2.BRISK_create(
        thresh=args.brisk_thresh,
        octaves=args.brisk_octaves,
        patternScale=args.brisk_pattern_scale,
    )


def get_beblid_scale_factor(args: argparse.Namespace) -> float:
    if args.beblid_scale_factor is not None:
        return args.beblid_scale_factor
    return DETECTOR_BEBLID_SCALE[args.detector]


def create_beblid_descriptor(
    args: argparse.Namespace,
) -> tuple[cv2.Feature2D, float]:
    ensure_beblid_available()
    beblid_scale_factor = get_beblid_scale_factor(args)
    if args.beblid_bits == 256:
        descriptor_size = cv2.xfeatures2d.BEBLID_SIZE_256_BITS
    else:
        descriptor_size = cv2.xfeatures2d.BEBLID_SIZE_512_BITS

    descriptor = cv2.xfeatures2d.BEBLID_create(
        beblid_scale_factor,
        descriptor_size,
    )
    return descriptor, beblid_scale_factor


def detect_and_compute(
    image: np.ndarray,
    detector: cv2.Feature2D,
    descriptor: cv2.Feature2D,
    mask: np.ndarray | None = None,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(gray, mask)
    if not keypoints:
        return [], None

    keypoints, descriptors = descriptor.compute(gray, keypoints)
    return list(keypoints), descriptors


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
    template_outline: np.ndarray,
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

    projected_outline = cv2.perspectiveTransform(template_outline, homography)
    return homography, mask.ravel().astype(int).tolist(), projected_outline


def get_template_outline(
    template_shape: tuple[int, int, int], template_mask: np.ndarray | None
) -> np.ndarray:
    if template_mask is None:
        height, width = template_shape[:2]
        return np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        ).reshape(-1, 1, 2)

    contours, _hierarchy = cv2.findContours(
        template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        height, width = template_shape[:2]
        return np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        ).reshape(-1, 1, 2)

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    return simplified.astype(np.float32)


def draw_scene_overlay(scene_image: np.ndarray, outline: np.ndarray | None) -> np.ndarray:
    overlay = scene_image.copy()
    if outline is not None:
        points = np.int32(outline)
        cv2.polylines(overlay, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
    return overlay


def prepare_template_visualization(
    template_image: np.ndarray, template_mask: np.ndarray | None
) -> np.ndarray:
    preview = template_image.copy()
    outline = get_template_outline(template_image.shape, template_mask)
    if template_mask is not None:
        dimmed = (preview.astype(np.float32) * 0.25).astype(np.uint8)
        preview = np.where(template_mask[:, :, None] > 0, preview, dimmed)
    cv2.polylines(preview, [np.int32(outline)], True, (0, 255, 255), 2, cv2.LINE_AA)
    return preview


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
    template_mask: np.ndarray | None,
    scene_image: np.ndarray,
    args: argparse.Namespace,
    template_mask_source: str | None,
) -> tuple[MatchResult, MatchTimings]:
    timings = MatchTimings()
    detector = create_detector(args)
    descriptor, beblid_scale_factor = create_beblid_descriptor(args)

    template_prepare_start = time.perf_counter()
    template_outline = get_template_outline(template_image.shape, template_mask)

    keypoints_template, descriptors_template = detect_and_compute(
        template_image,
        detector,
        descriptor,
        template_mask,
    )
    timings.template_prepare_ms = (
        time.perf_counter() - template_prepare_start
    ) * 1000.0

    scene_feature_start = time.perf_counter()
    keypoints_scene, descriptors_scene = detect_and_compute(
        scene_image,
        detector,
        descriptor,
    )
    timings.scene_feature_ms = (time.perf_counter() - scene_feature_start) * 1000.0

    descriptor_match_start = time.perf_counter()
    good_matches = ratio_test_match(
        descriptors_template, descriptors_scene, args.ratio_threshold
    )
    timings.descriptor_match_ms = (
        time.perf_counter() - descriptor_match_start
    ) * 1000.0

    homography_start = time.perf_counter()
    homography, inlier_mask, projected_outline = estimate_homography(
        keypoints_template,
        keypoints_scene,
        good_matches,
        args.min_matches,
        args.ransac_threshold,
        template_outline,
    )
    timings.homography_ms = (time.perf_counter() - homography_start) * 1000.0

    return (
        MatchResult(
            keypoints_template=keypoints_template,
            keypoints_scene=keypoints_scene,
            good_matches=good_matches,
            homography=homography,
            inlier_mask=inlier_mask,
            projected_outline=projected_outline,
            detector_name=args.detector,
            beblid_scale_factor=beblid_scale_factor,
            beblid_bits=args.beblid_bits,
            template_mask_source=template_mask_source,
        ),
        timings,
    )


def main() -> int:
    args = parse_args()
    timings = MatchTimings()

    def print_matching_timing() -> None:
        if not args.print_timing:
            return
        print(f"template load ms: {timings.template_load_ms:.3f}")
        print(f"template prepare ms: {timings.template_prepare_ms:.3f}")
        print(f"template setup ms: {timings.template_setup_ms:.3f}")
        print(f"scene load ms: {timings.scene_load_ms:.3f}")
        print(f"scene feature ms: {timings.scene_feature_ms:.3f}")
        print(f"descriptor match ms: {timings.descriptor_match_ms:.3f}")
        print(f"homography ms: {timings.homography_ms:.3f}")
        print(f"save output ms: {timings.save_output_ms:.3f}")
        print(f"matching time ms: {timings.matching_time_ms:.3f}")
        print(f"pipeline total ms: {timings.pipeline_total_ms:.3f}")

    try:
        template_load_start = time.perf_counter()
        template_image, template_mask, template_mask_source = load_template_image_and_mask(
            args.template, args.template_mask
        )
        timings.template_load_ms = (time.perf_counter() - template_load_start) * 1000.0

        scene_load_start = time.perf_counter()
        scene_image = load_image(args.scene)
        timings.scene_load_ms = (time.perf_counter() - scene_load_start) * 1000.0

        result, build_timings = build_result(
            template_image,
            template_mask,
            scene_image,
            args,
            template_mask_source,
        )
        timings.template_prepare_ms = build_timings.template_prepare_ms
        timings.scene_feature_ms = build_timings.scene_feature_ms
        timings.descriptor_match_ms = build_timings.descriptor_match_ms
        timings.homography_ms = build_timings.homography_ms
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        print_matching_timing()
        return 1

    template_visualization = prepare_template_visualization(template_image, template_mask)
    scene_overlay = draw_scene_overlay(scene_image, result.projected_outline)

    output_path = Path(args.output)
    save_output_start = time.perf_counter()
    save_visualization(
        template_visualization,
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
    timings.save_output_ms = (time.perf_counter() - save_output_start) * 1000.0

    inlier_count = sum(result.inlier_mask) if result.inlier_mask is not None else 0
    print(f"detector: {result.detector_name}")
    print(f"beblid scale factor: {result.beblid_scale_factor}")
    print(f"beblid bits: {result.beblid_bits}")
    print(f"template mask: {result.template_mask_source or 'none'}")
    print(f"template keypoints: {len(result.keypoints_template)}")
    print(f"scene keypoints: {len(result.keypoints_scene)}")
    print(f"good matches: {len(result.good_matches)}")
    print(f"inliers: {inlier_count}")
    print(f"visualization saved to: {output_path}")

    if result.homography is None:
        print(
            "Homography was not estimated. Try lowering --ratio-threshold, lowering the "
            "detector threshold, or using images with more texture.",
            file=sys.stderr,
        )
        print_matching_timing()
        return 2

    if args.overlay_output:
        print(f"overlay saved to: {Path(args.overlay_output)}")
    print_matching_timing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
