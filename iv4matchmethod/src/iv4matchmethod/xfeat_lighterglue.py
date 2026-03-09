from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from kornia.feature.lightglue import LightGlue
from PIL import Image, ImageDraw
from torch import nn

from iv4matchmethod.annotate import polygon_to_image
from iv4matchmethod.image_ops import draw_prediction_overlay, load_rgb


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


def pil_to_batched_rgb_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def rescale_points(points: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return np.asarray(points, dtype=np.float32) / scale


def apply_homography(points: Iterable[Iterable[float]], homography: np.ndarray) -> np.ndarray:
    pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, homography.astype(np.float64))
    return transformed.reshape(-1, 2)


def bbox_to_polygon(bbox: Iterable[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    return np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )


def draw_match_visualization(
    template_image: Image.Image,
    search_image: Image.Image,
    template_points: np.ndarray,
    search_points: np.ndarray,
    inlier_mask: np.ndarray | None,
    max_matches: int,
) -> Image.Image:
    if template_image.mode != "RGB":
        template_image = template_image.convert("RGB")
    if search_image.mode != "RGB":
        search_image = search_image.convert("RGB")

    canvas = Image.new(
        "RGB",
        (template_image.width + search_image.width, max(template_image.height, search_image.height)),
        (18, 18, 18),
    )
    canvas.paste(template_image, (0, 0))
    canvas.paste(search_image, (template_image.width, 0))
    draw = ImageDraw.Draw(canvas)

    count = min(len(template_points), max_matches)
    if count == 0:
        return canvas

    indices = np.linspace(0, len(template_points) - 1, num=count, dtype=int)
    for idx in indices:
        x0, y0 = template_points[idx]
        x1, y1 = search_points[idx]
        color = (80, 220, 80)
        if inlier_mask is not None and not bool(inlier_mask[idx]):
            color = (255, 90, 90)
        draw.line((float(x0), float(y0), float(x1 + template_image.width), float(y1)), fill=color, width=2)
        draw.ellipse((x0 - 3, y0 - 3, x0 + 3, y0 + 3), fill=color)
        draw.ellipse((x1 + template_image.width - 3, y1 - 3, x1 + template_image.width + 3, y1 + 3), fill=color)
    return canvas


@dataclass(slots=True)
class XFeatMatchConfig:
    top_k: int = 4096
    detection_threshold: float = 0.05
    min_confidence: float = 0.1
    max_dim: int = 1024
    ransac_reproj_threshold: float = 4.0
    max_draw_matches: int = 80


@dataclass(slots=True)
class XFeatMatchResult:
    template_image: str
    template_annotation: str
    search_image: str
    keypoints_template: int
    keypoints_search: int
    matches: int
    inliers: int
    inlier_ratio: float
    homography: list[list[float]] | None
    roi_follow: list[list[float]] | None
    bbox_follow: list[list[float]] | None
    match_vis_path: str | None
    roi_vis_path: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PreparedXFeatTemplate:
    template_image: str
    template_annotation: str
    annotation: dict[str, object]
    original_image: Image.Image
    resized_image: Image.Image
    scale: float
    features: dict[str, torch.Tensor]


class LighterGlueWrapper(nn.Module):
    # Derived from the official XFeat repository wrapper for Kornia LightGlue.
    default_conf_xfeat = {
        "name": "xfeat",
        "input_dim": 64,
        "descriptor_dim": 96,
        "add_scale_ori": False,
        "add_laf": False,
        "scale_coef": 1.0,
        "n_layers": 6,
        "num_heads": 1,
        "flash": True,
        "mp": False,
        "depth_confidence": -1,
        "width_confidence": 0.95,
        "filter_threshold": 0.1,
        "weights": None,
    }

    def __init__(self) -> None:
        super().__init__()
        LightGlue.default_conf = self.default_conf_xfeat
        self.net = LightGlue(None)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt",
            map_location=self.dev,
        )

        for index in range(self.net.conf.n_layers):
            state_dict = {
                key.replace(f"self_attn.{index}", f"transformers.{index}.self_attn"): value
                for key, value in state_dict.items()
            }
            state_dict = {
                key.replace(f"cross_attn.{index}", f"transformers.{index}.cross_attn"): value
                for key, value in state_dict.items()
            }
            state_dict = {key.replace("matcher.", ""): value for key, value in state_dict.items()}

        self.net.load_state_dict(state_dict, strict=False)
        self.net.to(self.dev).eval()

    @torch.inference_mode()
    def forward(self, feature0: dict[str, torch.Tensor], feature1: dict[str, torch.Tensor], min_confidence: float) -> np.ndarray:
        self.net.conf.filter_threshold = min_confidence
        result = self.net(
            {
                "image0": {
                    "keypoints": feature0["keypoints"][None, ...],
                    "descriptors": feature0["descriptors"][None, ...],
                    "image_size": feature0["image_size"][None, ...],
                },
                "image1": {
                    "keypoints": feature1["keypoints"][None, ...],
                    "descriptors": feature1["descriptors"][None, ...],
                    "image_size": feature1["image_size"][None, ...],
                },
            }
        )
        return result["matches"][0].detach().cpu().numpy()


def load_xfeat(top_k: int, detection_threshold: float):
    return torch.hub.load(
        "verlab/accelerated_features",
        "XFeat",
        pretrained=True,
        top_k=top_k,
        detection_threshold=detection_threshold,
        trust_repo=True,
        skip_validation=True,
    )


def detect_xfeat_features(model, image: Image.Image, top_k: int) -> dict[str, torch.Tensor]:
    tensor = pil_to_batched_rgb_tensor(image)
    output = model.detectAndCompute(tensor, top_k=top_k)[0]
    output["image_size"] = torch.tensor((image.width, image.height), dtype=torch.float32, device=output["keypoints"].device)
    return output


def estimate_homography(
    template_points: np.ndarray,
    search_points: np.ndarray,
    ransac_reproj_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(template_points) < 4 or len(search_points) < 4:
        return None, None
    homography, mask = cv2.findHomography(
        template_points.astype(np.float32),
        search_points.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=5000,
        confidence=0.995,
    )
    if homography is None or mask is None:
        return None, None
    return homography, mask.reshape(-1).astype(bool)


class XFeatLighterGlueMatcher:
    def __init__(self, config: XFeatMatchConfig | None = None) -> None:
        self.config = config or XFeatMatchConfig()
        self.xfeat = load_xfeat(self.config.top_k, self.config.detection_threshold)
        self.lighterglue = LighterGlueWrapper()
        self._template_cache: dict[tuple[str, str, int, int, float], PreparedXFeatTemplate] = {}

    def prepare_template(
        self,
        template_image_path: str | Path,
        template_annotation_path: str | Path,
    ) -> PreparedXFeatTemplate:
        template_image = str(Path(template_image_path).resolve())
        template_annotation = str(Path(template_annotation_path).resolve())
        cache_key = (
            template_image,
            template_annotation,
            self.config.max_dim,
            self.config.top_k,
            self.config.detection_threshold,
        )
        cached = self._template_cache.get(cache_key)
        if cached is not None:
            return cached

        annotation = load_template_annotation(template_annotation)
        template_original = load_rgb(template_image)
        template_resized, template_scale = resize_to_max_dim(template_original, self.config.max_dim)
        template_features = detect_xfeat_features(self.xfeat, template_resized, self.config.top_k)

        prepared = PreparedXFeatTemplate(
            template_image=template_image,
            template_annotation=template_annotation,
            annotation=annotation,
            original_image=template_original,
            resized_image=template_resized,
            scale=template_scale,
            features=template_features,
        )
        self._template_cache[cache_key] = prepared
        return prepared

    def match_search_image(
        self,
        prepared_template: PreparedXFeatTemplate,
        search_image_path: str | Path,
        output_dir: str | Path | None = None,
        *,
        save_visuals: bool = True,
        save_result_json: bool = True,
        print_result: bool = True,
    ) -> XFeatMatchResult:
        if (save_visuals or save_result_json) and output_dir is None:
            raise ValueError("output_dir is required when saving outputs")

        search_image = str(Path(search_image_path).resolve())
        search_original = load_rgb(search_image)
        search_resized, search_scale = resize_to_max_dim(search_original, self.config.max_dim)

        feature1 = detect_xfeat_features(self.xfeat, search_resized, self.config.top_k)
        match_indices = self.lighterglue(
            prepared_template.features,
            feature1,
            min_confidence=self.config.min_confidence,
        )

        mkpts0 = prepared_template.features["keypoints"][match_indices[:, 0]].detach().cpu().numpy()
        mkpts1 = feature1["keypoints"][match_indices[:, 1]].detach().cpu().numpy()
        mkpts0_original = rescale_points(mkpts0, prepared_template.scale)
        mkpts1_original = rescale_points(mkpts1, search_scale)

        homography, inlier_mask = estimate_homography(
            mkpts0_original,
            mkpts1_original,
            self.config.ransac_reproj_threshold,
        )

        roi_follow = None
        bbox_follow = None
        if homography is not None:
            roi_follow = apply_homography(prepared_template.annotation["roi_image_polygon"], homography).tolist()
            bbox_follow = apply_homography(
                bbox_to_polygon(prepared_template.annotation["template_bbox"]),
                homography,
            ).tolist()

        match_vis_path = None
        roi_vis_path = None
        result_path = None
        output_path = Path(output_dir) if output_dir is not None else None
        if output_path is not None and (save_visuals or save_result_json):
            output_path.mkdir(parents=True, exist_ok=True)

        if output_path is not None and save_visuals:
            match_vis = draw_match_visualization(
                prepared_template.resized_image,
                search_resized,
                mkpts0,
                mkpts1,
                inlier_mask,
                self.config.max_draw_matches,
            )
            match_vis_path = str((output_path / "xfeat_lighterglue_matches.png").resolve())
            match_vis.save(match_vis_path)

            roi_overlay = draw_prediction_overlay(search_original, polygon=roi_follow)
            if bbox_follow is not None:
                draw = ImageDraw.Draw(roi_overlay)
                pts = [(float(x), float(y)) for x, y in bbox_follow]
                draw.line(pts + [pts[0]], fill=(0, 200, 255), width=3)
            roi_vis_path = str((output_path / "xfeat_lighterglue_roi.png").resolve())
            roi_overlay.save(roi_vis_path)

        result = XFeatMatchResult(
            template_image=prepared_template.template_image,
            template_annotation=prepared_template.template_annotation,
            search_image=search_image,
            keypoints_template=int(prepared_template.features["keypoints"].shape[0]),
            keypoints_search=int(feature1["keypoints"].shape[0]),
            matches=int(len(match_indices)),
            inliers=int(inlier_mask.sum()) if inlier_mask is not None else 0,
            inlier_ratio=float(inlier_mask.mean()) if inlier_mask is not None and len(inlier_mask) > 0 else 0.0,
            homography=homography.tolist() if homography is not None else None,
            roi_follow=roi_follow,
            bbox_follow=bbox_follow,
            match_vis_path=match_vis_path,
            roi_vis_path=roi_vis_path,
        )

        if output_path is not None and save_result_json:
            result_path = output_path / "xfeat_lighterglue_result.json"
            result_path.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if print_result:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return result


def match_template_with_xfeat_lighterglue(
    template_image_path: str | Path,
    template_annotation_path: str | Path,
    search_image_path: str | Path,
    output_dir: str | Path | None,
    config: XFeatMatchConfig | None = None,
    *,
    matcher: XFeatLighterGlueMatcher | None = None,
    save_visuals: bool = True,
    save_result_json: bool = True,
    print_result: bool = True,
) -> XFeatMatchResult:
    config = config or XFeatMatchConfig()
    matcher = matcher or XFeatLighterGlueMatcher(config)
    prepared_template = matcher.prepare_template(template_image_path, template_annotation_path)
    return matcher.match_search_image(
        prepared_template,
        search_image_path,
        output_dir=output_dir,
        save_visuals=save_visuals,
        save_result_json=save_result_json,
        print_result=print_result,
    )


def run_xfeat_match(args) -> XFeatMatchResult:
    template_image = Path(args.template_image) if args.template_image else Path(load_template_annotation(args.template_annotation)["template_image"])
    save_visuals = not getattr(args, "no_write_visuals", False)
    save_result_json = not getattr(args, "no_write_json", False)
    if (save_visuals or save_result_json) and not args.output_dir:
        raise ValueError("--output-dir is required unless both --no-write-visuals and --no-write-json are set")
    return match_template_with_xfeat_lighterglue(
        template_image_path=template_image,
        template_annotation_path=args.template_annotation,
        search_image_path=args.search_image,
        output_dir=args.output_dir,
        config=XFeatMatchConfig(
            top_k=args.top_k,
            detection_threshold=args.detection_threshold,
            min_confidence=args.min_confidence,
            max_dim=args.max_dim,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
            max_draw_matches=args.max_draw_matches,
        ),
        save_visuals=save_visuals,
        save_result_json=save_result_json,
        print_result=not getattr(args, "quiet", False),
    )
