from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_background(size: int, rng: np.random.Generator) -> Image.Image:
    background = Image.new("RGB", (size, size), (32, 38, 48))
    draw = ImageDraw.Draw(background)
    for _ in range(18):
        x0 = int(rng.integers(0, size - 20))
        y0 = int(rng.integers(0, size - 20))
        x1 = x0 + int(rng.integers(8, 36))
        y1 = y0 + int(rng.integers(8, 36))
        shade = int(rng.integers(40, 95))
        draw.rectangle((x0, y0, x1, y1), fill=(shade, shade, shade))
    return background


def make_object_patch(defect: bool = False) -> Image.Image:
    patch = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    draw.rounded_rectangle((32, 40, 128, 120), radius=18, fill=(214, 214, 214, 255), outline=(36, 36, 36, 255), width=4)
    draw.rectangle((48, 56, 112, 104), outline=(20, 70, 160, 255), width=4)
    draw.line((80, 40, 80, 120), fill=(36, 36, 36, 255), width=3)
    draw.line((32, 80, 128, 80), fill=(36, 36, 36, 255), width=3)
    draw.ellipse((68, 68, 92, 92), fill=(250, 180, 40, 255), outline=(36, 36, 36, 255), width=2)
    if defect:
        draw.rectangle((72, 72, 100, 88), fill=(210, 20, 40, 255))
    return patch


def paste_center(background: Image.Image, patch: Image.Image, center: tuple[float, float]) -> None:
    left = int(round(center[0] - patch.width / 2.0))
    top = int(round(center[1] - patch.height / 2.0))
    background.paste(patch, (left, top), patch)


def create_episode(index: int, image_dir: Path, rng: np.random.Generator) -> dict[str, object]:
    search_size = 384
    template_size = 256
    bbox = [80, 88, 96, 80]
    roi_ref_polygon = [[-20, -12], [20, -12], [20, 12], [-20, 12]]
    is_ng = bool(rng.random() < 0.25)

    template_image = make_background(template_size, rng)
    template_patch = make_object_patch(defect=False)
    paste_center(template_image, template_patch, (template_size / 2.0, template_size / 2.0))

    search_image = make_background(search_size, rng)
    sx = float(rng.uniform(0.85, 1.15))
    sy = float(rng.uniform(0.85, 1.15))
    angle_deg = float(rng.uniform(-30.0, 30.0))
    patch = make_object_patch(defect=is_ng)
    scaled = patch.resize((max(1, int(round(patch.width * sx))), max(1, int(round(patch.height * sy)))), resample=Image.Resampling.BILINEAR)
    rotated = scaled.rotate(angle_deg, resample=Image.Resampling.BILINEAR, expand=True)
    margin = 96
    center = (
        float(rng.uniform(margin, search_size - margin)),
        float(rng.uniform(margin, search_size - margin)),
    )
    paste_center(search_image, rotated, center)

    template_name = f"template_{index:04d}.png"
    search_name = f"search_{index:04d}.png"
    template_image.save(image_dir / template_name)
    search_image.save(image_dir / search_name)

    return {
        "template_image": f"images/{template_name}",
        "template_bbox": bbox,
        "search_image": f"images/{search_name}",
        "center": [center[0], center[1]],
        "angle_deg": angle_deg,
        "scale": [sx, sy],
        "roi_ref_polygon": roi_ref_polygon,
        "ok_ng": "NG" if is_ng else "OK",
    }


def synthesize_dataset(args) -> dict[str, object]:
    output_dir = Path(args.output)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    train_records = [create_episode(index, image_dir, rng) for index in range(args.train_samples)]
    val_records = [
        create_episode(args.train_samples + index, image_dir, rng)
        for index in range(args.val_samples)
    ]

    (output_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(record) for record in train_records),
        encoding="utf-8",
    )
    (output_dir / "val.jsonl").write_text(
        "\n".join(json.dumps(record) for record in val_records),
        encoding="utf-8",
    )
    meta = {
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "template_bbox": [80, 88, 96, 80],
        "roi_ref_polygon": [[-20, -12], [20, -12], [20, 12], [-20, 12]],
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_dir.resolve()), **meta}, indent=2))
    return meta

