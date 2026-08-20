from __future__ import annotations

import filecmp
import os
import shutil
import tempfile
from pathlib import Path


TRAINING_SAMPLE_SUBDIR = Path("samples") / "train"


def is_path_in_product(path: object, *, product_dir: object) -> bool:
    product_text = str(product_dir or "").strip()
    path_text = str(path or "").strip()
    if not product_text or not path_text:
        return False
    product_root = os.path.normcase(os.path.realpath(os.path.abspath(product_text)))
    candidate = path_text if os.path.isabs(path_text) else os.path.join(product_root, path_text)
    candidate = os.path.normcase(os.path.realpath(os.path.abspath(candidate)))
    try:
        return os.path.commonpath([product_root, candidate]) == product_root
    except ValueError:
        return False


def ensure_training_image_local(path: object, *, product_dir: object) -> str:
    """Return a product-local training image path, copying external data when needed."""
    product_text = str(product_dir or "").strip()
    if not product_text:
        raise ValueError("current product directory is unavailable")
    product_root = Path(os.path.abspath(product_text))

    source_text = str(path or "").strip()
    if not source_text:
        raise ValueError("training image path is empty")
    source = Path(source_text)
    if not source.is_absolute():
        source = product_root / source
    source = Path(os.path.abspath(source))
    if not source.is_file():
        raise FileNotFoundError(source)
    if is_path_in_product(source, product_dir=product_root):
        return str(source)

    target_dir = product_root / TRAINING_SAMPLE_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    source_sidecar = source.with_suffix(".json")

    candidate = target_dir / source.name
    suffix_index = 1
    while candidate.exists() or candidate.with_suffix(".json").exists():
        if _training_bundle_matches(source, candidate, source_sidecar=source_sidecar):
            return str(candidate)
        candidate = target_dir / f"{source.stem}_{suffix_index}{source.suffix}"
        suffix_index += 1

    copied_image = False
    try:
        _copy_file_atomically(source, candidate)
        copied_image = True
        if source_sidecar.is_file():
            _copy_file_atomically(source_sidecar, candidate.with_suffix(".json"))
    except Exception:
        if copied_image:
            try:
                candidate.unlink()
            except OSError:
                pass
        raise
    return str(candidate)


def _training_bundle_matches(
    source: Path,
    candidate: Path,
    *,
    source_sidecar: Path,
) -> bool:
    try:
        if not candidate.is_file() or not filecmp.cmp(source, candidate, shallow=False):
            return False
        if not source_sidecar.is_file():
            return True
        candidate_sidecar = candidate.with_suffix(".json")
        return candidate_sidecar.is_file() and filecmp.cmp(
            source_sidecar,
            candidate_sidecar,
            shallow=False,
        )
    except OSError:
        return False


def _copy_file_atomically(source: Path, target: Path) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".training-import-",
        suffix=target.suffix,
        dir=str(target.parent),
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, target)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "TRAINING_SAMPLE_SUBDIR",
    "ensure_training_image_local",
    "is_path_in_product",
]
