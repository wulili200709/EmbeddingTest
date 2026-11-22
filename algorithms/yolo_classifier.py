from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from common.app_paths import packaged_embedding_test_root
from common.labelme_io import clamp_roi_xywh, labelme_json_of_image, read_shape_from_labelme
from common.safe_io import atomic_write_json, load_json_with_backup


YOLO_CLASSIFIER_CODE = "yolov8n_cls"
YOLO_STORAGE_CODE = "y8ncls"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 16
DEFAULT_NG_THRESHOLD = 0.50
LETTERBOX_COLOR = (114, 114, 114)


def _safe_model_key(value: object) -> str:
    text = str(value or "").strip()
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text).strip("._-")


@dataclass(frozen=True)
class YoloClassifierPaths:
    pt: Path
    onnx: Path
    metadata: Path


def classifier_paths(product_dir: str | Path, model_key: object = "") -> YoloClassifierPaths:
    key = _safe_model_key(model_key)
    stem = f"{key}_{YOLO_STORAGE_CODE}" if key else YOLO_STORAGE_CODE
    root = Path(product_dir)
    return YoloClassifierPaths(
        pt=root / f"{stem}.pt",
        onnx=root / f"{stem}.onnx",
        metadata=root / f"{stem}.json",
    )


def base_model_path() -> Path:
    return packaged_embedding_test_root(__file__) / "res" / "models" / "yolov8n-cls.pt"


def letterbox(image: Image.Image, size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    source = image.convert("RGB")
    if source.width <= 0 or source.height <= 0:
        raise ValueError("invalid ROI image size")
    scale = min(size / source.width, size / source.height)
    resized_size = (
        max(1, round(source.width * scale)),
        max(1, round(source.height * scale)),
    )
    resized = source.resize(resized_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), LETTERBOX_COLOR)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def _crop_shape(image_rgb: np.ndarray, shape: Mapping[str, object]) -> Image.Image:
    height, width = image_rgb.shape[:2]
    points = np.asarray(shape.get("points", []), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise RuntimeError("ROI points are empty or invalid")
    x_min, y_min = points[:, :2].min(axis=0)
    x_max, y_max = points[:, :2].max(axis=0)
    x, y, w, h = clamp_roi_xywh(
        int(round(float(x_min))),
        int(round(float(y_min))),
        int(round(float(x_max - x_min))),
        int(round(float(y_max - y_min))),
        W=width,
        H=height,
    )
    crop = np.ascontiguousarray(image_rgb[y : y + h, x : x + w].copy())
    if str(shape.get("shape_type", "rectangle") or "rectangle") == "polygon" and points.shape[0] >= 3:
        relative = np.round(points[:, :2] - np.array([[x, y]], dtype=np.float32)).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [relative], 255)
        crop[mask == 0] = 0
    return Image.fromarray(crop, mode="RGB")


def prepared_roi_from_path(
    image_path: str | Path,
    roi_label: str,
    *,
    size: int = DEFAULT_IMAGE_SIZE,
    roi_xywh: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    path = str(image_path)
    with Image.open(path) as source:
        image_rgb = np.asarray(source.convert("RGB"))
    if roi_xywh is not None:
        x, y, w, h = roi_xywh
        shape = {
            "shape_type": "rectangle",
            "points": [[x, y], [x + w, y + h]],
        }
    else:
        json_path = labelme_json_of_image(path)
        shape = read_shape_from_labelme(json_path, roi_label)
        if not shape:
            raise RuntimeError(f"Label '{roi_label}' not found in {json_path}")
    return letterbox(_crop_shape(image_rgb, shape), size=size)


def prepared_roi_from_bgr(
    image_bgr: np.ndarray,
    shape: Mapping[str, object],
    *,
    size: int = DEFAULT_IMAGE_SIZE,
) -> Image.Image:
    image = np.asarray(image_bgr)
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image_rgb = cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]), cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"unsupported runtime image shape: {image.shape!r}")
    return letterbox(_crop_shape(image_rgb, shape), size=size)


def _input_tensor(image: Image.Image, size: int) -> np.ndarray:
    prepared = letterbox(image, size=size) if image.size != (size, size) else image.convert("RGB")
    array = np.asarray(prepared, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponent = np.exp(shifted)
    return exponent / max(float(exponent.sum()), 1e-12)


class YoloOnnxClassifier:
    def __init__(self, onnx_path: str | Path, metadata_path: str | Path) -> None:
        self.onnx_path = str(Path(onnx_path))
        self.metadata_path = str(Path(metadata_path))
        metadata = load_json_with_backup(self.metadata_path, default={})
        if not isinstance(metadata, dict):
            raise RuntimeError(f"invalid deep classifier metadata: {self.metadata_path}")
        raw_names = metadata.get("names", {})
        if isinstance(raw_names, list):
            self.names = {index: str(name) for index, name in enumerate(raw_names)}
        elif isinstance(raw_names, dict):
            self.names = {int(index): str(name) for index, name in raw_names.items()}
        else:
            self.names = {}
        normalized_names = {str(name).strip().lower() for name in self.names.values()}
        if not {"ok", "ng"}.issubset(normalized_names):
            raise RuntimeError(f"classifier classes must contain ok and ng: {self.names}")
        self.image_size = max(1, int(metadata.get("image_size", DEFAULT_IMAGE_SIZE)))
        providers = ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError(f"invalid classifier ONNX graph: {self.onnx_path}")
        self.input_name = str(inputs[0].name)
        self.output_name = str(outputs[0].name)

    def predict(self, image: Image.Image, *, ng_threshold: float = DEFAULT_NG_THRESHOLD) -> dict[str, float | str]:
        output = np.asarray(
            self.session.run([self.output_name], {self.input_name: _input_tensor(image, self.image_size)})[0],
            dtype=np.float32,
        ).reshape(-1)
        if output.size != len(self.names):
            raise RuntimeError(f"classifier output size {output.size} does not match names {self.names}")
        if np.any(output < 0.0) or not math.isclose(float(output.sum()), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            output = _softmax(output)
        scores = {self.names[index].strip().lower(): float(output[index]) for index in range(output.size)}
        ok_score = scores["ok"]
        ng_score = scores["ng"]
        pred = "NG" if ng_score >= float(ng_threshold) else "OK"
        return {
            "pred": pred,
            "diff": ok_score - ng_score,
            "sim_ok": ok_score,
            "sim_ng": ng_score,
        }


def _write_dataset_group(
    samples: Sequence[tuple[str, str]],
    class_name: str,
    dataset_dir: Path,
    *,
    size: int,
    seed: int,
    progress_callback: Callable[[str], None] | None,
) -> None:
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    val_count = max(1, round(len(ordered) * 0.20))
    groups = (("val", ordered[:val_count]), ("train", ordered[val_count:]))
    completed = 0
    for split, entries in groups:
        output_dir = dataset_dir / split / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, (image_path, roi_label) in enumerate(entries, start=1):
            prepared = prepared_roi_from_path(image_path, roi_label, size=size)
            destination = output_dir / f"{index:05d}_{Path(image_path).stem}_{_safe_model_key(roi_label)}.png"
            prepared.save(destination, format="PNG")
            completed += 1
            if callable(progress_callback):
                progress_callback(f"preparing {class_name.upper()} ROI ({completed}/{len(ordered)})")


def _normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    return {}


def train_classifier(
    ok_samples: Sequence[tuple[str, str]],
    ng_samples: Sequence[tuple[str, str]],
    *,
    product_dir: str | Path,
    model_key: object,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[YoloOnnxClassifier, YoloClassifierPaths, Path]:
    if len(ok_samples) < 2 or len(ng_samples) < 2:
        raise RuntimeError("Deep classification training needs at least 2 OK and 2 NG ROI samples")
    base_path = base_model_path()
    if not base_path.is_file():
        raise FileNotFoundError(f"Classification base model not found: {base_path}")

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("The classification training dependency is not installed") from exc

    product_root = Path(product_dir)
    tag = time.strftime("%Y%m%d_%H%M%S")
    key = _safe_model_key(model_key) or "shared"
    dataset_dir = product_root / "classification_training_datasets" / key / tag
    _write_dataset_group(
        ok_samples, "ok", dataset_dir, size=image_size, seed=20260718,
        progress_callback=progress_callback,
    )
    _write_dataset_group(
        ng_samples, "ng", dataset_dir, size=image_size, seed=20260719,
        progress_callback=progress_callback,
    )

    device: str | int = 0 if torch.cuda.is_available() else "cpu"
    if callable(progress_callback):
        progress_callback(f"loading classification base model on {device}")
    model = YOLO(str(base_path))

    def _on_epoch_end(trainer: object) -> None:
        current = int(getattr(trainer, "epoch", 0)) + 1
        total = max(1, int(getattr(trainer, "epochs", epochs)))
        if callable(progress_callback):
            progress_callback(f"classification epoch ({current}/{total})")

    model.add_callback("on_train_epoch_end", _on_epoch_end)
    run_name = f"{key}_{YOLO_STORAGE_CODE}_{tag}"
    model.train(
        data=str(dataset_dir),
        epochs=max(1, int(epochs)),
        imgsz=max(32, int(image_size)),
        batch=max(1, int(batch)),
        patience=20,
        optimizer="auto",
        pretrained=True,
        device=device,
        project=str(product_root / "classification_runs"),
        name=run_name,
        exist_ok=False,
        seed=20260718,
        # Training and inference must work without downloading plot fonts.
        plots=False,
        workers=0,
        scale=0.0,
        erasing=0.0,
        auto_augment=None,
        fliplr=0.0,
        hsv_h=0.0,
        hsv_s=0.15,
        hsv_v=0.15,
        verbose=False,
    )
    best_path = Path(model.trainer.best)
    if not best_path.is_file():
        raise FileNotFoundError(f"Classification training finished but best.pt was not found: {best_path}")

    names = _normalize_names(getattr(model, "names", {}))
    if not {"ok", "ng"}.issubset({name.strip().lower() for name in names.values()}):
        raise RuntimeError(f"trained classifier classes must contain ok and ng: {names}")

    paths = classifier_paths(product_root, model_key)
    paths.pt.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged_pt = paths.pt.with_name(f".{paths.pt.stem}.{token}.pt")
    staged_onnx = staged_pt.with_suffix(".onnx")
    staged_metadata = paths.metadata.with_name(f".{paths.metadata.stem}.{token}.json")
    try:
        shutil.copy2(best_path, staged_pt)
        if callable(progress_callback):
            progress_callback("exporting classifier ONNX")
        exported = Path(
            YOLO(str(staged_pt)).export(
                format="onnx",
                opset=12,
                imgsz=max(32, int(image_size)),
                device="cpu",
                dynamic=False,
                simplify=False,
                verbose=False,
            )
        )
        if exported.resolve() != staged_onnx.resolve():
            shutil.copy2(exported, staged_onnx)
        onnx.checker.check_model(onnx.load(str(staged_onnx)))
        atomic_write_json(
            staged_metadata,
            {
                "algorithm": "deep_classifier",
                "names": {str(index): name for index, name in names.items()},
                "image_size": int(image_size),
                "resize_mode": "letterbox",
                "base_model": base_path.name,
            },
            create_backup=False,
        )
        staged_runner = YoloOnnxClassifier(staged_onnx, staged_metadata)
        staged_runner.predict(prepared_roi_from_path(ok_samples[0][0], ok_samples[0][1], size=image_size))
        if callable(progress_callback):
            progress_callback("activating verified classifier ONNX")
        os.replace(staged_pt, paths.pt)
        os.replace(staged_metadata, paths.metadata)
        os.replace(staged_onnx, paths.onnx)
    finally:
        for candidate in (staged_pt, staged_onnx, staged_metadata):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass

    return YoloOnnxClassifier(paths.onnx, paths.metadata), paths, dataset_dir


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_NG_THRESHOLD",
    "YOLO_CLASSIFIER_CODE",
    "YOLO_STORAGE_CODE",
    "YoloClassifierPaths",
    "YoloOnnxClassifier",
    "base_model_path",
    "classifier_paths",
    "prepared_roi_from_bgr",
    "prepared_roi_from_path",
    "train_classifier",
]
