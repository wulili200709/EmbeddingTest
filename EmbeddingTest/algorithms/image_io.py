from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


def _path_text(path: object) -> str:
    return str(path or "").strip()


def _use_encoded_file_io(path_text: str) -> bool:
    return os.name == "nt" and not path_text.isascii()


def imread(path: object, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    path_text = _path_text(path)
    if not path_text:
        return None

    if not _use_encoded_file_io(path_text):
        image = cv2.imread(path_text, flags)
        if image is not None:
            return image

    try:
        data = np.fromfile(path_text, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: object, image: np.ndarray, params: Optional[Iterable[int]] = None) -> bool:
    path_text = _path_text(path)
    if not path_text:
        return False

    if not _use_encoded_file_io(path_text):
        if cv2.imwrite(path_text, image, list(params or [])):
            return True

    suffix = Path(path_text).suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image, list(params or []))
    if not ok:
        return False
    try:
        encoded.tofile(path_text)
    except OSError:
        return False
    return True


__all__ = ["imread", "imwrite"]
