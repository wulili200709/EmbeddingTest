from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from iv4matchmethod.annotate import polygon_to_image
from iv4matchmethod.image_ops import draw_prediction_overlay, load_rgb, parse_bbox


def load_template_annotation(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "roi_image_polygon" not in payload:
        payload["roi_image_polygon"] = polygon_to_image(
            payload["roi_ref_polygon"],
            payload["template_bbox"],
        )
    return payload


def resize_to_max_dim(image: Image.Image, max_dim: int) -> tuple[Image.Image, float]:
    if max_dim <= 0:
        return image.copy(), 1.0
    scale = min(float(max_dim) / max(image.size), 1.0)
    if scale == 1.0:
        return image.copy(), scale
    size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(size, resample=Image.Resampling.BILINEAR), scale


def scale_bbox(bbox: Iterable[float], scale: float) -> tuple[float, float, float, float]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    x, y, w, h = parse_bbox(bbox)
    return x * scale, y * scale, w * scale, h * scale


def expand_bbox(bbox: Iterable[float], margin_ratio: float) -> tuple[float, float, float, float]:
    x, y, w, h = parse_bbox(bbox)
    margin_x = w * max(0.0, margin_ratio)
    margin_y = h * max(0.0, margin_ratio)
    return x - margin_x, y - margin_y, w + margin_x * 2.0, h + margin_y * 2.0


def clamp_bbox_to_image(
    bbox: Iterable[float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = parse_bbox(bbox)
    image_width, image_height = image_size
    left = max(0, int(math.floor(x)))
    top = max(0, int(math.floor(y)))
    right = min(image_width, int(math.ceil(x + w)))
    bottom = min(image_height, int(math.ceil(y + h)))
    if right <= left or bottom <= top:
        raise ValueError("bbox does not overlap the image")
    return left, top, right, bottom


def pil_to_gray_array(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def build_edge_feature(gray_u8: np.ndarray, canny_low: int, canny_high: int) -> tuple[np.ndarray, np.ndarray]:
    edges = cv2.Canny(gray_u8, canny_low, canny_high)
    feature = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (3, 3), 0)
    return feature, edges


def build_template_feature(gray_u8: np.ndarray, canny_low: int, canny_high: int) -> tuple[np.ndarray, np.ndarray]:
    feature, edges = build_edge_feature(gray_u8, canny_low, canny_high)
    mask = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    if int(mask.sum()) == 0:
        raise ValueError("template crop does not contain enough edges")
    return feature, mask


def warp_feature_and_mask(
    feature: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if scale <= 0:
        return None

    height, width = feature.shape[:2]
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle_deg, scale)
    cos_v = abs(matrix[0, 0])
    sin_v = abs(matrix[0, 1])
    out_width = max(1, int(math.ceil(width * cos_v + height * sin_v)))
    out_height = max(1, int(math.ceil(width * sin_v + height * cos_v)))
    matrix[0, 2] += out_width / 2.0 - center[0]
    matrix[1, 2] += out_height / 2.0 - center[1]

    warped_feature = cv2.warpAffine(
        feature,
        matrix,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_mask = cv2.warpAffine(
        mask,
        matrix,
        (out_width, out_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ys, xs = np.where(warped_mask > 0)
    if len(xs) == 0:
        return None

    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    center_point = cv2.transform(center.reshape(1, 1, 2), matrix).reshape(2) - np.array([left, top], dtype=np.float32)
    return warped_feature[top:bottom, left:right], warped_mask[top:bottom, left:right], center_point


def iter_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be >= start")
    values = np.arange(start, stop + step * 0.5, step, dtype=np.float64)
    decimals = max(0, min(6, int(math.ceil(-math.log10(step))) + 1)) if step < 1 else 3
    return [round(float(v), decimals) for v in values.tolist()]


def iter_pose_grid(
    angle_min: float,
    angle_max: float,
    angle_step: float,
    scale_min: float,
    scale_max: float,
    scale_step: float,
) -> list[tuple[float, float]]:
    poses: list[tuple[float, float]] = []
    for angle in iter_values(angle_min, angle_max, angle_step):
        for scale in iter_values(scale_min, scale_max, scale_step):
            poses.append((angle, scale))
    return poses


def transform_relative_points(
    relative_points: Iterable[Iterable[float]],
    center_xy: Iterable[float],
    angle_deg: float,
    scale: float,
) -> np.ndarray:
    pts = np.asarray(list(relative_points), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("relative_points must be shaped [N, 2]")
    theta = math.radians(float(angle_deg))
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=np.float32,
    )
    center = np.asarray(list(center_xy), dtype=np.float32)
    return (pts @ rotation.T) * float(scale) + center


def bbox_to_relative_polygon(bbox: Iterable[float]) -> np.ndarray:
    _, _, width, height = parse_bbox(bbox)
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    return np.asarray(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float32,
    )


def draw_shape_match_visualization(
    template_crop: Image.Image,
    search_image: Image.Image,
    bbox_follow: list[list[float]] | None,
    roi_follow: list[list[float]] | None,
    score: float,
    angle_deg: float,
    scale: float,
) -> Image.Image:
    if template_crop.mode != "RGB":
        template_crop = template_crop.convert("RGB")
    overlay = draw_prediction_overlay(search_image, polygon=roi_follow)
    if bbox_follow is not None:
        draw = ImageDraw.Draw(overlay)
        pts = [(float(x), float(y)) for x, y in bbox_follow]
        draw.line(pts + [pts[0]], fill=(0, 200, 255), width=3)

    canvas = Image.new(
        "RGB",
        (template_crop.width + overlay.width, max(template_crop.height, overlay.height) + 40),
        (20, 20, 20),
    )
    canvas.paste(template_crop, (0, 40))
    canvas.paste(overlay, (template_crop.width, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"score={score:.3f} angle={angle_deg:.1f} scale={scale:.3f}", fill=(235, 235, 235))
    return canvas


@dataclass(slots=True)
class ShapeMatchConfig:
    max_dim: int = 640
    angle_min: float = -20.0
    angle_max: float = 20.0
    coarse_angle_step: float = 4.0
    fine_angle_step: float = 1.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    coarse_scale_step: float = 0.05
    fine_scale_step: float = 0.01
    canny_low: int = 40
    canny_high: int = 120
    template_margin_ratio: float = 0.0
    min_score: float = 0.2


@dataclass(slots=True)
class ShapeMatchResult:
    template_image: str
    template_annotation: str
    search_image: str
    method: str
    center: list[float] | None
    angle_deg: float | None
    scale: float | None
    score: float | None
    matched: bool
    roi_follow: list[list[float]] | None
    bbox_follow: list[list[float]] | None
    visualization_path: str | None
    timing_ms: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PreparedShapeTemplate:
    template_image: str
    template_annotation: str
    annotation: dict[str, object]
    original_image: Image.Image
    resized_image: Image.Image
    template_crop: Image.Image
    template_scale: float
    crop_box: tuple[int, int, int, int]
    feature: np.ndarray
    mask: np.ndarray


class ShapeTemplateMatcher:
    def __init__(self, config: ShapeMatchConfig | None = None) -> None:
        self.config = config or ShapeMatchConfig()
        self._template_cache: dict[tuple[str, str, int, int, int, float], PreparedShapeTemplate] = {}

    def prepare_template(
        self,
        template_image_path: str | Path,
        template_annotation_path: str | Path,
    ) -> PreparedShapeTemplate:
        template_image = str(Path(template_image_path).resolve())
        template_annotation = str(Path(template_annotation_path).resolve())
        cache_key = (
            template_image,
            template_annotation,
            self.config.max_dim,
            self.config.canny_low,
            self.config.canny_high,
            self.config.template_margin_ratio,
        )
        cached = self._template_cache.get(cache_key)
        if cached is not None:
            return cached

        annotation = load_template_annotation(template_annotation)
        template_original = load_rgb(template_image)
        template_resized, template_scale = resize_to_max_dim(template_original, self.config.max_dim)
        scaled_bbox = scale_bbox(annotation["template_bbox"], template_scale)
        crop_box = clamp_bbox_to_image(
            expand_bbox(scaled_bbox, self.config.template_margin_ratio),
            template_resized.size,
        )
        template_crop = template_resized.crop(crop_box)
        gray_crop = pil_to_gray_array(template_crop)
        feature, mask = build_template_feature(gray_crop, self.config.canny_low, self.config.canny_high)

        prepared = PreparedShapeTemplate(
            template_image=template_image,
            template_annotation=template_annotation,
            annotation=annotation,
            original_image=template_original,
            resized_image=template_resized,
            template_crop=template_crop,
            template_scale=template_scale,
            crop_box=crop_box,
            feature=feature,
            mask=mask,
        )
        self._template_cache[cache_key] = prepared
        return prepared

    def _match_pose(
        self,
        search_feature: np.ndarray,
        poses: Iterable[tuple[float, float]],
        prepared_template: PreparedShapeTemplate,
    ) -> tuple[float, float, float, tuple[int, int], np.ndarray] | None:
        best: tuple[float, float, float, tuple[int, int], np.ndarray] | None = None
        for angle_deg, scale in poses:
            warped = warp_feature_and_mask(prepared_template.feature, prepared_template.mask, angle_deg, scale)
            if warped is None:
                continue
            warped_feature, warped_mask, center_offset = warped
            if warped_feature.shape[0] >= search_feature.shape[0] or warped_feature.shape[1] >= search_feature.shape[1]:
                continue
            response = cv2.matchTemplate(
                search_feature,
                warped_feature,
                cv2.TM_CCORR_NORMED,
                mask=warped_mask,
            )
            _, max_score, _, max_loc = cv2.minMaxLoc(response)
            candidate = (
                float(max_score),
                float(angle_deg),
                float(scale),
                (int(max_loc[0]), int(max_loc[1])),
                center_offset,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best

    def match_search_image(
        self,
        prepared_template: PreparedShapeTemplate,
        search_image_path: str | Path,
        output_dir: str | Path | None = None,
        *,
        save_visualization: bool = True,
        save_result_json: bool = True,
        print_result: bool = True,
    ) -> ShapeMatchResult:
        if (save_visualization or save_result_json) and output_dir is None:
            raise ValueError("output_dir is required when saving outputs")

        search_image = str(Path(search_image_path).resolve())
        search_original = load_rgb(search_image)
        search_resized, search_scale = resize_to_max_dim(search_original, self.config.max_dim)

        t0 = time.perf_counter()
        search_feature, _ = build_edge_feature(
            pil_to_gray_array(search_resized),
            self.config.canny_low,
            self.config.canny_high,
        )
        t1 = time.perf_counter()

        coarse_poses = iter_pose_grid(
            self.config.angle_min,
            self.config.angle_max,
            self.config.coarse_angle_step,
            self.config.scale_min,
            self.config.scale_max,
            self.config.coarse_scale_step,
        )
        coarse_best = self._match_pose(search_feature, coarse_poses, prepared_template)
        t2 = time.perf_counter()

        best = coarse_best
        if coarse_best is not None and self.config.fine_angle_step > 0 and self.config.fine_scale_step > 0:
            fine_poses = iter_pose_grid(
                max(self.config.angle_min, coarse_best[1] - self.config.coarse_angle_step),
                min(self.config.angle_max, coarse_best[1] + self.config.coarse_angle_step),
                self.config.fine_angle_step,
                max(self.config.scale_min, coarse_best[2] - self.config.coarse_scale_step),
                min(self.config.scale_max, coarse_best[2] + self.config.coarse_scale_step),
                self.config.fine_scale_step,
            )
            fine_best = self._match_pose(search_feature, fine_poses, prepared_template)
            if fine_best is not None and (best is None or fine_best[0] >= best[0]):
                best = fine_best
        t3 = time.perf_counter()

        matched = best is not None and best[0] >= self.config.min_score
        center_original = None
        angle_deg = None
        actual_scale = None
        roi_follow = None
        bbox_follow = None

        if best is not None:
            center_resized = np.asarray(
                [best[3][0] + float(best[4][0]), best[3][1] + float(best[4][1])],
                dtype=np.float32,
            )
            center_original = (center_resized / max(search_scale, 1e-6)).tolist()
            angle_deg = float(best[1])
            actual_scale = float(best[2] * prepared_template.template_scale / max(search_scale, 1e-6))
            if matched:
                roi_follow = transform_relative_points(
                    prepared_template.annotation["roi_ref_polygon"],
                    center_original,
                    angle_deg,
                    actual_scale,
                ).tolist()
                bbox_follow = transform_relative_points(
                    bbox_to_relative_polygon(prepared_template.annotation["template_bbox"]),
                    center_original,
                    angle_deg,
                    actual_scale,
                ).tolist()

        timing_ms = {
            "search_feature_ms": round((t1 - t0) * 1000, 1),
            "coarse_match_ms": round((t2 - t1) * 1000, 1),
            "fine_match_ms": round((t3 - t2) * 1000, 1),
            "total_match_ms": round((t3 - t0) * 1000, 1),
        }

        visualization_path = None
        output_path = Path(output_dir) if output_dir is not None else None
        if output_path is not None and (save_visualization or save_result_json):
            output_path.mkdir(parents=True, exist_ok=True)

        if output_path is not None and save_visualization:
            visualization = draw_shape_match_visualization(
                prepared_template.template_crop,
                search_original,
                bbox_follow=bbox_follow,
                roi_follow=roi_follow,
                score=float(best[0]) if best is not None else 0.0,
                angle_deg=float(angle_deg or 0.0),
                scale=float(actual_scale or 0.0),
            )
            visualization_path = str((output_path / "shape_match_result.png").resolve())
            visualization.save(visualization_path)

        result = ShapeMatchResult(
            template_image=prepared_template.template_image,
            template_annotation=prepared_template.template_annotation,
            search_image=search_image,
            method="edge_template_match",
            center=[float(center_original[0]), float(center_original[1])] if matched and center_original is not None else None,
            angle_deg=angle_deg if matched else None,
            scale=actual_scale if matched else None,
            score=float(best[0]) if best is not None else None,
            matched=bool(matched),
            roi_follow=roi_follow,
            bbox_follow=bbox_follow,
            visualization_path=visualization_path,
            timing_ms=timing_ms,
        )

        if output_path is not None and save_result_json:
            result_path = output_path / "shape_match_result.json"
            result_path.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if print_result:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return result


def match_template_with_shape(
    template_image_path: str | Path,
    template_annotation_path: str | Path,
    search_image_path: str | Path,
    output_dir: str | Path | None,
    config: ShapeMatchConfig | None = None,
    *,
    matcher: ShapeTemplateMatcher | None = None,
    save_visualization: bool = True,
    save_result_json: bool = True,
    print_result: bool = True,
) -> ShapeMatchResult:
    config = config or ShapeMatchConfig()
    matcher = matcher or ShapeTemplateMatcher(config)
    prepared_template = matcher.prepare_template(template_image_path, template_annotation_path)
    return matcher.match_search_image(
        prepared_template,
        search_image_path,
        output_dir=output_dir,
        save_visualization=save_visualization,
        save_result_json=save_result_json,
        print_result=print_result,
    )


def run_shape_match(args) -> ShapeMatchResult:
    template_image = (
        Path(args.template_image)
        if args.template_image
        else Path(load_template_annotation(args.template_annotation)["template_image"])
    )
    save_visualization = not getattr(args, "no_write_visuals", False)
    save_result_json = not getattr(args, "no_write_json", False)
    if (save_visualization or save_result_json) and not args.output_dir:
        raise ValueError("--output-dir is required unless both --no-write-visuals and --no-write-json are set")
    return match_template_with_shape(
        template_image_path=template_image,
        template_annotation_path=args.template_annotation,
        search_image_path=args.search_image,
        output_dir=args.output_dir,
        config=ShapeMatchConfig(
            max_dim=args.max_dim,
            angle_min=args.angle_min,
            angle_max=args.angle_max,
            coarse_angle_step=args.coarse_angle_step,
            fine_angle_step=args.fine_angle_step,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            coarse_scale_step=args.coarse_scale_step,
            fine_scale_step=args.fine_scale_step,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            template_margin_ratio=args.template_margin_ratio,
            min_score=args.min_score,
        ),
        save_visualization=save_visualization,
        save_result_json=save_result_json,
        print_result=not getattr(args, "quiet", False),
    )
