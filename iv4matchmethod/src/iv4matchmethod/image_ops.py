from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def parse_bbox(value: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in value]
    return x, y, w, h


def crop_bbox(image: Image.Image, bbox: Iterable[float]) -> Image.Image:
    x, y, w, h = parse_bbox(bbox)
    left = max(0, min(image.width, x))
    top = max(0, min(image.height, y))
    right = max(left + 1.0, min(image.width, x + w))
    bottom = max(top + 1.0, min(image.height, y + h))
    return image.crop((left, top, right, bottom))


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), resample=Image.Resampling.BILINEAR)


def letterbox_image(
    image: Image.Image,
    size: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, float, tuple[float, float]]:
    scale = min(size / image.width, size / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
    return canvas, float(scale), (float(pad_x), float(pad_y))


def points_to_model(
    points: np.ndarray,
    scale: float,
    pad: tuple[float, float],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] = pts[:, 0] * scale + pad[0]
    pts[:, 1] = pts[:, 1] * scale + pad[1]
    return pts


def points_to_original(
    points: np.ndarray,
    scale: float,
    pad: tuple[float, float],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] = (pts[:, 0] - pad[0]) / max(scale, 1e-6)
    pts[:, 1] = (pts[:, 1] - pad[1]) / max(scale, 1e-6)
    return pts


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Iterable[float],
    std: Iterable[float],
) -> torch.Tensor:
    mean_tensor = torch.tensor(list(mean), dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(list(std), dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    data = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    data = (data.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(data)


def parse_json_arg(value: str) -> object:
    candidate = Path(value)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(value)


def draw_prediction_overlay(
    image: Image.Image,
    center: tuple[float, float] | None = None,
    polygon: Iterable[Iterable[float]] | None = None,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    if polygon is not None:
        pts = [(float(x), float(y)) for x, y in polygon]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=(255, 80, 0), width=3)
    if center is not None:
        cx, cy = center
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(0, 220, 0))
    return result
