from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from iv4matchmethod.config import ModelConfig
from iv4matchmethod.geometry import decode_pose, extract_aligned_roi, transform_polygon
from iv4matchmethod.image_ops import (
    crop_bbox,
    draw_prediction_overlay,
    letterbox_image,
    load_rgb,
    normalize_tensor,
    parse_json_arg,
    pil_to_tensor,
    points_to_original,
)
from iv4matchmethod.models.network import TemplateConditionedLocator
from iv4matchmethod.prototype import embed_patch, load_bank, prototype_scores
from iv4matchmethod.train import choose_device


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[TemplateConditionedLocator, ModelConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ModelConfig(**checkpoint["config"])
    model = TemplateConditionedLocator(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, config


def run_inference(args) -> dict[str, object]:
    device = choose_device(args.device)
    model, config = load_model(args.checkpoint, device)

    template_bbox = parse_json_arg(args.template_bbox)
    roi_ref_polygon = parse_json_arg(args.roi_ref_polygon) if args.roi_ref_polygon else None

    template_image = load_rgb(args.template_image)
    search_image = load_rgb(args.search_image)
    template_patch = crop_bbox(template_image, template_bbox).resize(
        (config.template_size, config.template_size),
        resample=Image.Resampling.BILINEAR,
    )
    search_patch, scale_factor, pad = letterbox_image(search_image, config.search_size)

    with torch.no_grad():
        outputs = model(
            normalize_tensor(
                pil_to_tensor(template_patch),
                config.input_mean,
                config.input_std,
            ).unsqueeze(0).to(device),
            normalize_tensor(
                pil_to_tensor(search_patch),
                config.input_mean,
                config.input_std,
            ).unsqueeze(0).to(device),
        )
        pose = decode_pose(outputs)[0]

    center_original = points_to_original(
        np.asarray([[pose.cx, pose.cy]], dtype=np.float32),
        scale_factor,
        pad,
    )[0]

    roi_polygon_original: list[list[float]] | None = None
    roi_patch = None
    if roi_ref_polygon is not None:
        roi_polygon_model = transform_polygon(
            roi_ref_polygon,
            pose.cx,
            pose.cy,
            pose.theta,
            pose.sx,
            pose.sy,
        )
        roi_polygon = points_to_original(roi_polygon_model, scale_factor, pad)
        roi_polygon_original = roi_polygon.tolist()
        roi_patch = extract_aligned_roi(search_image, roi_polygon_original, (args.roi_size, args.roi_size))

    judgement = None
    if args.prototype_bank:
        if roi_patch is None:
            raise ValueError("--prototype-bank requires --roi-ref-polygon")
        bank = load_bank(args.prototype_bank)
        judgement = prototype_scores(embed_patch(roi_patch), bank)

    if args.debug_image:
        overlay = draw_prediction_overlay(
            search_image,
            center=(float(center_original[0]), float(center_original[1])),
            polygon=roi_polygon_original,
        )
        Path(args.debug_image).parent.mkdir(parents=True, exist_ok=True)
        overlay.save(args.debug_image)

    if args.roi_patch_output and roi_patch is not None:
        Path(args.roi_patch_output).parent.mkdir(parents=True, exist_ok=True)
        roi_patch.save(args.roi_patch_output)

    result = {
        "pose": {
            **pose.to_dict(),
            "cx_original": float(center_original[0]),
            "cy_original": float(center_original[1]),
        },
        "roi_follow": roi_polygon_original,
        "judgement": judgement,
    }
    print(json.dumps(result, indent=2))
    return result
