from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np


@dataclass
class TemplateVariant:
    template_id: int
    class_id: str
    angle: float
    scale: float
    image: np.ndarray
    mask: np.ndarray
    corners: np.ndarray
    feature_bbox: tuple[int, int, int, int]


@dataclass
class Detection:
    class_id: str
    template_id: int
    angle: float
    scale: float
    similarity: float
    origin: tuple[int, int]
    box: tuple[int, int, int, int]
    corners: np.ndarray


@dataclass
class RawMatchRecord:
    class_id: str
    template_id: int
    similarity: float
    x: int
    y: int


def parse_csv_floats(value: str) -> list[float]:
    numbers: list[float] = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        numbers.append(float(item))
    if not numbers:
        raise argparse.ArgumentTypeError("Expected at least one numeric value.")
    return numbers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "2D shape-based template matching with OpenCV LINE/LINEMOD "
            "(color-gradient modality)."
        )
    )
    parser.add_argument("--template", required=True, help="Path to the template image.")
    parser.add_argument("--scene", required=True, help="Path to the scene image.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the annotated scene image.",
    )
    parser.add_argument(
        "--mask",
        help="Optional binary mask for the template. Must match template size.",
    )
    parser.add_argument(
        "--template-preview-output",
        help="Optional path to save a contact sheet of generated LINE templates.",
    )
    parser.add_argument(
        "--class-id",
        default="object",
        help="LINE template class ID. Default: object",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Detector similarity threshold in [0, 100]. Default: 90.0",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Maximum raw matches to inspect before NMS. Default: 200",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=12,
        help="Maximum detections kept after NMS. Default: 12",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.25,
        help="IoU threshold for NMS de-duplication. Default: 0.25",
    )
    parser.add_argument(
        "--angles",
        type=parse_csv_floats,
        default=[0.0],
        help="Comma-separated template rotations in degrees. Default: 0",
    )
    parser.add_argument(
        "--scales",
        type=parse_csv_floats,
        default=[1.0],
        help="Comma-separated template scales. Default: 1.0",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=245,
        help=(
            "When --mask is omitted, pixels darker than this grayscale threshold are "
            "treated as foreground. Default: 245"
        ),
    )
    parser.add_argument(
        "--preview-variants",
        type=int,
        default=6,
        help="How many generated variants to include in the preview sheet. Default: 6",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=960,
        help=(
            "Tile size used for fallback matching on large scenes. Set 0 to disable "
            "tiling. Default: 960"
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=256,
        help="Overlap in pixels between fallback tiles. Default: 256",
    )
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def load_mask(path: str, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match template shape {expected_shape}."
        )
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if component_count <= 2:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return largest


def auto_mask_from_template(template_image: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray < threshold, 255, 0).astype(np.uint8)

    if not np.any(mask):
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = keep_largest_component(mask)

    if not np.any(mask):
        raise ValueError("Template mask is empty after auto-segmentation.")
    return mask


def crop_to_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, height = cv2.boundingRect(mask)
    if width == 0 or height == 0:
        raise ValueError("Template mask is empty.")
    return (
        image[y : y + height, x : x + width].copy(),
        mask[y : y + height, x : x + width].copy(),
    )


def build_variant(
    base_image: np.ndarray,
    base_mask: np.ndarray,
    angle: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}.")

    height, width = base_image.shape[:2]
    corners = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    center = (width / 2.0, height / 2.0)

    transform = cv2.getRotationMatrix2D(center, angle, scale)
    transformed_corners = cv2.transform(corners[None, :, :], transform)[0]

    min_xy = np.floor(transformed_corners.min(axis=0))
    max_xy = np.ceil(transformed_corners.max(axis=0))
    out_width = max(1, int(max_xy[0] - min_xy[0] + 1))
    out_height = max(1, int(max_xy[1] - min_xy[1] + 1))

    transform[0, 2] -= min_xy[0]
    transform[1, 2] -= min_xy[1]

    warped_image = cv2.warpAffine(
        base_image,
        transform,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    warped_mask = cv2.warpAffine(
        base_mask,
        transform,
        (out_width, out_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = np.where(warped_mask > 127, 255, 0).astype(np.uint8)
    warped_mask = keep_largest_component(warped_mask)

    cropped_image, cropped_mask = crop_to_mask(warped_image, warped_mask)
    crop_x, crop_y, _, _ = cv2.boundingRect(warped_mask)
    local_corners = (
        cv2.transform(corners[None, :, :], transform)[0]
        - np.float32([crop_x, crop_y])
    )

    return cropped_image, cropped_mask, local_corners


def build_detector(
    template_image: np.ndarray,
    template_mask: np.ndarray,
    class_id: str,
    angles: list[float],
    scales: list[float],
) -> tuple[cv2.linemod_Detector, np.ndarray, np.ndarray, dict[int, TemplateVariant]]:
    base_image, base_mask = crop_to_mask(template_image, template_mask)
    detector = cv2.linemod.getDefaultLINE()
    variants: dict[int, TemplateVariant] = {}

    for scale in scales:
        for angle in angles:
            variant_image, variant_mask, variant_corners = build_variant(
                base_image, base_mask, angle, scale
            )
            template_id, bbox = detector.addTemplate(
                [variant_image],
                class_id,
                variant_mask,
            )
            if template_id < 0:
                continue

            variants[template_id] = TemplateVariant(
                template_id=template_id,
                class_id=class_id,
                angle=angle,
                scale=scale,
                image=variant_image,
                mask=variant_mask,
                corners=variant_corners,
                feature_bbox=tuple(int(value) for value in bbox),
            )

    if not variants:
        raise RuntimeError(
            "Failed to extract any valid LINE templates. Try a cleaner mask or a "
            "template with stronger edges."
        )

    return detector, base_image, base_mask, variants


def resolve_detections(
    raw_matches: list[RawMatchRecord],
    variants: dict[int, TemplateVariant],
) -> list[Detection]:
    detections: list[Detection] = []
    seen: set[tuple[int, int, int]] = set()

    for match in raw_matches:
        variant = variants.get(match.template_id)
        if variant is None:
            continue

        match_key = (variant.template_id, int(match.x), int(match.y))
        if match_key in seen:
            continue
        seen.add(match_key)

        bbox_x, bbox_y, _, _ = variant.feature_bbox
        origin_x = int(match.x - bbox_x)
        origin_y = int(match.y - bbox_y)

        scene_corners = variant.corners + np.float32([origin_x, origin_y])
        min_xy = np.floor(scene_corners.min(axis=0)).astype(int)
        max_xy = np.ceil(scene_corners.max(axis=0)).astype(int)
        width = max(1, int(max_xy[0] - min_xy[0] + 1))
        height = max(1, int(max_xy[1] - min_xy[1] + 1))

        detections.append(
            Detection(
                class_id=variant.class_id,
                template_id=variant.template_id,
                angle=variant.angle,
                scale=variant.scale,
                similarity=float(match.similarity),
                origin=(origin_x, origin_y),
                box=(int(min_xy[0]), int(min_xy[1]), width, height),
                corners=scene_corners,
            )
        )

    return detections


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    a_right = ax + aw
    a_bottom = ay + ah
    b_right = bx + bw
    b_bottom = by + bh

    inter_left = max(ax, bx)
    inter_top = max(ay, by)
    inter_right = min(a_right, b_right)
    inter_bottom = min(a_bottom, b_bottom)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    intersection = float((inter_right - inter_left) * (inter_bottom - inter_top))
    union = float(aw * ah + bw * bh) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def non_max_suppression(
    detections: list[Detection],
    iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    kept: list[Detection] = []
    for detection in detections:
        if any(
            compute_iou(detection.box, selected.box) >= iou_threshold
            for selected in kept
        ):
            continue
        kept.append(detection)
        if len(kept) >= max_detections:
            break
    return kept


def draw_text_block(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    top_left = (x, y - text_size[1] - baseline - 4)
    bottom_right = (x + text_size[0] + 6, y + 2)
    cv2.rectangle(image, top_left, bottom_right, (10, 10, 10), -1)
    cv2.putText(
        image,
        text,
        (x + 3, y - 3),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(scene_image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    overlay = scene_image.copy()
    palette = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 220, 0),
        (255, 120, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for index, detection in enumerate(detections):
        color = palette[index % len(palette)]
        points = np.round(detection.corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [points], True, color, 2, cv2.LINE_AA)

        label_x = max(0, int(np.min(points[:, 0, 0])))
        label_y = max(18, int(np.min(points[:, 0, 1])) - 4)
        label = (
            f"#{index + 1} score={detection.similarity:.1f} "
            f"a={detection.angle:g} s={detection.scale:g}"
        )
        draw_text_block(overlay, label, (label_x, label_y), color)

    return overlay


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    new_width = max(1, int(round(image.shape[1] * scale)))
    new_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    offset_x = (width - new_width) // 2
    offset_y = (height - new_height) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized
    return canvas


def render_template_preview(
    detector: cv2.linemod_Detector,
    variants: dict[int, TemplateVariant],
    preview_variants: int,
) -> np.ndarray:
    selected = list(sorted(variants.values(), key=lambda item: item.template_id))[
        : max(1, preview_variants)
    ]
    previews: list[np.ndarray] = []

    for variant in selected:
        preview = variant.image.copy()
        templates = detector.getTemplates(variant.class_id, variant.template_id)
        preview = cv2.linemod.drawFeatures(
            preview,
            templates,
            (variant.feature_bbox[0], variant.feature_bbox[1]),
            3,
        )

        contours, _ = cv2.findContours(
            variant.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.polylines(
            preview,
            [np.round(variant.corners).astype(np.int32)],
            True,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        panel = fit_panel(preview, 220, 180)
        draw_text_block(
            panel,
            f"id={variant.template_id} a={variant.angle:g} s={variant.scale:g}",
            (8, 24),
            (0, 0, 0),
        )
        previews.append(panel)

    columns = min(3, len(previews))
    rows = (len(previews) + columns - 1) // columns
    sheet = np.full((rows * 180, columns * 220, 3), 255, dtype=np.uint8)

    for index, panel in enumerate(previews):
        row = index // columns
        column = index % columns
        y = row * 180
        x = column * 220
        sheet[y : y + 180, x : x + 220] = panel

    return sheet


def pad_scene_for_linemod(
    scene_image: np.ndarray,
    detector: cv2.linemod_Detector,
) -> tuple[np.ndarray, tuple[int, int]]:
    levels = max(1, int(detector.pyramidLevels()))
    step = 1
    for level in range(levels):
        step = math.lcm(step, int(detector.getT(level)))

    height, width = scene_image.shape[:2]
    padded_height = ((height + step - 1) // step) * step
    padded_width = ((width + step - 1) // step) * step

    pad_bottom = padded_height - height
    pad_right = padded_width - width
    if pad_bottom == 0 and pad_right == 0:
        return scene_image, (0, 0)

    padded = cv2.copyMakeBorder(
        scene_image,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return padded, (pad_right, pad_bottom)


def convert_matches(matches: list[cv2.linemod_Match]) -> list[RawMatchRecord]:
    return [
        RawMatchRecord(
            class_id=str(match.class_id),
            template_id=int(match.template_id),
            similarity=float(match.similarity),
            x=int(match.x),
            y=int(match.y),
        )
        for match in matches
    ]


def build_tile_starts(length: int, tile_size: int, tile_overlap: int) -> list[int]:
    if tile_size <= 0 or length <= tile_size:
        return [0]

    step = tile_size - tile_overlap
    if step <= 0:
        raise ValueError("--tile-overlap must be smaller than --tile-size.")

    starts = list(range(0, length - tile_size + 1, step))
    last_start = length - tile_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def match_scene_in_tiles(
    detector: cv2.linemod_Detector,
    scene_image: np.ndarray,
    threshold: float,
    tile_size: int,
    tile_overlap: int,
) -> tuple[list[RawMatchRecord], int]:
    height, width = scene_image.shape[:2]
    x_starts = build_tile_starts(width, tile_size, tile_overlap)
    y_starts = build_tile_starts(height, tile_size, tile_overlap)

    matches: list[RawMatchRecord] = []
    tile_count = 0
    for y_start in y_starts:
        for x_start in x_starts:
            tile = scene_image[
                y_start : min(y_start + tile_size, height),
                x_start : min(x_start + tile_size, width),
            ].copy()
            padded_tile, _ = pad_scene_for_linemod(tile, detector)
            tile_matches, _ = detector.match([padded_tile], threshold)
            tile_count += 1

            for match in tile_matches:
                matches.append(
                    RawMatchRecord(
                        class_id=str(match.class_id),
                        template_id=int(match.template_id),
                        similarity=float(match.similarity),
                        x=int(match.x) + x_start,
                        y=int(match.y) + y_start,
                    )
                )

    matches.sort(key=lambda item: item.similarity, reverse=True)
    return matches, tile_count


def match_scene(
    detector: cv2.linemod_Detector,
    scene_image: np.ndarray,
    threshold: float,
    tile_size: int,
    tile_overlap: int,
) -> tuple[list[RawMatchRecord], str]:
    padded_scene_image, scene_padding = pad_scene_for_linemod(scene_image, detector)
    try:
        raw_matches, _ = detector.match([padded_scene_image], threshold)
        match_records = convert_matches(raw_matches)
        mode = "full-frame"
        if scene_padding != (0, 0):
            mode += f" (padded right={scene_padding[0]} bottom={scene_padding[1]})"
        return match_records, mode
    except cv2.error:
        if tile_size <= 0:
            raise
        match_records, tile_count = match_scene_in_tiles(
            detector,
            scene_image,
            threshold,
            tile_size,
            tile_overlap,
        )
        return match_records, f"tiled ({tile_count} tiles)"


def main() -> int:
    args = parse_args()
    total_start = time.perf_counter()
    try:
        template_image = load_image(args.template)
        scene_image = load_image(args.scene)
        template_mask = (
            load_mask(args.mask, template_image.shape[:2])
            if args.mask
            else auto_mask_from_template(template_image, args.mask_threshold)
        )
        detector, cropped_template, cropped_mask, variants = build_detector(
            template_image,
            template_mask,
            args.class_id,
            args.angles,
            args.scales,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    match_start = time.perf_counter()
    startup_prep_ms = (match_start - total_start) * 1000.0
    try:
        raw_matches, match_mode = match_scene(
            detector,
            scene_image,
            float(args.threshold),
            args.tile_size,
            args.tile_overlap,
        )
    except (ValueError, cv2.error) as exc:
        print(exc, file=sys.stderr)
        return 1

    match_end = time.perf_counter()
    matching_ms = (match_end - match_start) * 1000.0
    limited_matches = list(raw_matches[: max(1, args.top_k)])
    detections = resolve_detections(limited_matches, variants)
    detections.sort(key=lambda item: item.similarity, reverse=True)
    kept_detections = non_max_suppression(
        detections,
        iou_threshold=args.nms_iou,
        max_detections=args.max_detections,
    )

    overlay = draw_overlay(scene_image, kept_detections)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)

    if args.template_preview_output:
        preview = render_template_preview(detector, variants, args.preview_variants)
        preview_path = Path(args.template_preview_output)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), preview)

    print(f"template size after crop: {cropped_template.shape[1]}x{cropped_template.shape[0]}")
    print(f"mask foreground pixels: {int(np.count_nonzero(cropped_mask))}")
    print(f"generated templates: {len(variants)}")
    print(f"matching mode: {match_mode}")
    print(f"raw matches above threshold: {len(raw_matches)}")
    print(f"matches inspected before NMS: {len(limited_matches)}")
    print(f"detections kept after NMS: {len(kept_detections)}")
    print(f"annotated scene saved to: {output_path}")

    for index, detection in enumerate(kept_detections, start=1):
        x, y, width, height = detection.box
        print(
            f"[{index}] score={detection.similarity:.1f} class={detection.class_id} "
            f"template_id={detection.template_id} angle={detection.angle:g} "
            f"scale={detection.scale:g} box=({x},{y},{width},{height})"
        )

    if args.template_preview_output:
        print(f"template preview saved to: {Path(args.template_preview_output)}")

    total_end = time.perf_counter()
    postprocess_ms = (total_end - match_end) * 1000.0
    total_ms = (total_end - total_start) * 1000.0
    print(f"timing startup_prep_ms: {startup_prep_ms:.3f}")
    print(f"timing matching_ms: {matching_ms:.3f}")
    print(f"timing postprocess_ms: {postprocess_ms:.3f}")
    print(f"timing total_ms: {total_ms:.3f}")

    return 0 if kept_detections else 2


if __name__ == "__main__":
    raise SystemExit(main())
