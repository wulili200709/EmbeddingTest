from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from iv4matchmethod.config import ModelConfig
from iv4matchmethod.image_ops import crop_bbox, letterbox_image, load_rgb, normalize_tensor, pil_to_tensor


def load_manifest(path: str | Path) -> list[dict[str, object]]:
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "episodes" in payload:
            return list(payload["episodes"])
        if "items" in payload:
            return list(payload["items"])
    raise ValueError(f"unsupported manifest format: {manifest_path}")


def make_heatmap(height: int, width: int, center_y: float, center_x: float, sigma: float) -> np.ndarray:
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    exponent = ((x - center_x) ** 2 + (y - center_y) ** 2) / (2.0 * sigma * sigma)
    return np.exp(-exponent)


def resolve_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return root / candidate


class EpisodeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        config: ModelConfig,
        heatmap_sigma: float = 1.75,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.manifest_root = self.manifest_path.parent
        self.records = load_manifest(manifest_path)
        self.config = config
        self.heatmap_sigma = float(heatmap_sigma)
        self.config.validate()

    def __len__(self) -> int:
        return len(self.records)

    def _build_targets(
        self,
        center_xy: tuple[float, float],
        scale_xy: tuple[float, float],
        angle_deg: float,
    ) -> dict[str, torch.Tensor]:
        response_size = self.config.response_size
        kernel = self.config.template_kernel
        coord_x = center_xy[0] / self.config.feature_stride - kernel / 2.0
        coord_y = center_xy[1] / self.config.feature_stride - kernel / 2.0
        coord_x = float(np.clip(coord_x, 0.0, response_size - 1.0))
        coord_y = float(np.clip(coord_y, 0.0, response_size - 1.0))

        grid_x = int(np.clip(np.rint(coord_x), 0, response_size - 1))
        grid_y = int(np.clip(np.rint(coord_y), 0, response_size - 1))
        heatmap = make_heatmap(response_size, response_size, coord_y, coord_x, self.heatmap_sigma)

        offset = np.zeros((2, response_size, response_size), dtype=np.float32)
        offset[0, grid_y, grid_x] = coord_x - grid_x
        offset[1, grid_y, grid_x] = coord_y - grid_y

        scale = np.zeros((2, response_size, response_size), dtype=np.float32)
        scale[0, grid_y, grid_x] = math.log(max(scale_xy[0], 1e-6))
        scale[1, grid_y, grid_x] = math.log(max(scale_xy[1], 1e-6))

        angle = np.zeros((2, response_size, response_size), dtype=np.float32)
        radians = math.radians(angle_deg)
        angle[0, grid_y, grid_x] = math.sin(radians)
        angle[1, grid_y, grid_x] = math.cos(radians)

        mask = np.zeros((1, response_size, response_size), dtype=np.float32)
        mask[0, grid_y, grid_x] = 1.0

        quality = np.zeros((1, response_size, response_size), dtype=np.float32)
        quality[0, grid_y, grid_x] = 1.0

        return {
            "heatmap": torch.from_numpy(heatmap[None, ...]),
            "offset": torch.from_numpy(offset),
            "scale": torch.from_numpy(scale),
            "angle": torch.from_numpy(angle),
            "quality": torch.from_numpy(quality),
            "mask": torch.from_numpy(mask),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        template_image = load_rgb(resolve_path(self.manifest_root, str(record["template_image"])))
        search_image = load_rgb(resolve_path(self.manifest_root, str(record["search_image"])))

        template_patch = crop_bbox(template_image, record["template_bbox"]).resize(
            (self.config.template_size, self.config.template_size),
            resample=Image.Resampling.BILINEAR,
        )
        search_patch, scale_factor, pad = letterbox_image(search_image, self.config.search_size)

        center = np.asarray(record["center"], dtype=np.float32)
        center_model = np.array(
            [center[0] * scale_factor + pad[0], center[1] * scale_factor + pad[1]],
            dtype=np.float32,
        )
        scale_xy = tuple(float(v) for v in record.get("scale", [1.0, 1.0]))
        angle_deg = float(record.get("angle_deg", 0.0))

        return {
            "template": normalize_tensor(
                pil_to_tensor(template_patch),
                self.config.input_mean,
                self.config.input_std,
            ),
            "search": normalize_tensor(
                pil_to_tensor(search_patch),
                self.config.input_mean,
                self.config.input_std,
            ),
            "target": self._build_targets((float(center_model[0]), float(center_model[1])), scale_xy, angle_deg),
            "meta": {
                "template_image": str(record["template_image"]),
                "search_image": str(record["search_image"]),
                "center": center.tolist(),
                "center_model": center_model.tolist(),
                "scale": list(scale_xy),
                "angle_deg": angle_deg,
                "pad": [float(pad[0]), float(pad[1])],
                "scale_factor": float(scale_factor),
                "ok_ng": str(record.get("ok_ng", "OK")),
            },
        }
