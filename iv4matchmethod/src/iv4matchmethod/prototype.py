from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from iv4matchmethod.geometry import extract_aligned_roi, transform_polygon
from iv4matchmethod.image_ops import load_rgb


def embed_patch(image: Image.Image, size: int = 64) -> np.ndarray:
    patch = image.convert("L").resize((size, size), resample=Image.Resampling.BILINEAR)
    vector = np.asarray(patch, dtype=np.float32).reshape(-1) / 255.0
    vector = vector - vector.mean()
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return vector
    return vector / norm


def prototype_scores(embedding: np.ndarray, bank: dict[str, np.ndarray]) -> dict[str, float | str]:
    ok = bank.get("ok", np.empty((0, embedding.shape[0]), dtype=np.float32))
    ng = bank.get("ng", np.empty((0, embedding.shape[0]), dtype=np.float32))

    if not ok.size and not ng.size:
        raise ValueError("prototype bank is empty")
    ok_score = float((ok @ embedding).mean()) if ok.size else float("-inf")
    ng_score = float((ng @ embedding).mean()) if ng.size else float("-inf")
    margin = ok_score - ng_score
    label = "OK" if margin >= 0.0 else "NG"
    return {
        "ok_score": ok_score,
        "ng_score": ng_score,
        "threshold_margin": margin,
        "ok_ng_label": label,
    }


def save_bank(path: str | Path, ok_embeddings: list[np.ndarray], ng_embeddings: list[np.ndarray]) -> None:
    embedding_dim = 4096
    if ok_embeddings:
        embedding_dim = ok_embeddings[0].shape[0]
    elif ng_embeddings:
        embedding_dim = ng_embeddings[0].shape[0]
    ok = np.stack(ok_embeddings, axis=0) if ok_embeddings else np.empty((0, embedding_dim), dtype=np.float32)
    ng = np.stack(ng_embeddings, axis=0) if ng_embeddings else np.empty((0, embedding_dim), dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ok=ok.astype(np.float32), ng=ng.astype(np.float32))


def load_bank(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {"ok": data["ok"], "ng": data["ng"]}


def build_bank_from_manifest(
    manifest_records: list[dict[str, object]],
    output_path: str | Path,
    manifest_root: str | Path,
    roi_size: int = 128,
) -> dict[str, int]:
    root = Path(manifest_root)
    ok_embeddings: list[np.ndarray] = []
    ng_embeddings: list[np.ndarray] = []

    for record in manifest_records:
        polygon = record.get("roi_ref_polygon")
        if not polygon:
            continue
        image = load_rgb(root / str(record["search_image"]))
        center = record["center"]
        scale = record.get("scale", [1.0, 1.0])
        angle_deg = float(record.get("angle_deg", 0.0))
        run_polygon = transform_polygon(
            polygon,
            float(center[0]),
            float(center[1]),
            np.deg2rad(angle_deg),
            float(scale[0]),
            float(scale[1]),
        )
        patch = extract_aligned_roi(image, run_polygon, (roi_size, roi_size))
        embedding = embed_patch(patch)
        label = str(record.get("ok_ng", "OK")).upper()
        if label == "NG":
            ng_embeddings.append(embedding)
        else:
            ok_embeddings.append(embedding)

    save_bank(output_path, ok_embeddings, ng_embeddings)
    return {"ok": len(ok_embeddings), "ng": len(ng_embeddings)}
