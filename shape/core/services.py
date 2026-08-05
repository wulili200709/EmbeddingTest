from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .recipe import ShapeRecipe
from .roi_follow import FollowResult, locate_and_follow
from ..like_matcher import ShapeLikeDetector, load_detector_model


@dataclass(frozen=True)
class RuntimeDetectedShape:
    label_name: str
    shape_type: str
    points: tuple[Tuple[float, float], ...]
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class ShapeLocateResult:
    result: FollowResult
    locate_ms: float


@dataclass(frozen=True)
class ShapeRuntimeResult:
    result: FollowResult
    roi_shapes: tuple[RuntimeDetectedShape, ...]
    locate_ms: float
    total_ms: float


def runtime_shapes_from_follow_result(result: FollowResult) -> tuple[RuntimeDetectedShape, ...]:
    shapes: list[RuntimeDetectedShape] = []
    for region in result.regions:
        points = tuple((float(x), float(y)) for x, y in region.points)
        if not points:
            continue
        shape_type = str(region.source_shape_type or "rectangle")
        if shape_type == "rectangle" and len(points) == 4:
            shape_points = (points[0], points[2])
        else:
            shape_points = points
        shapes.append(
            RuntimeDetectedShape(
                label_name=str(region.label_name or "").strip() or "roi",
                shape_type=shape_type,
                points=shape_points,
                bbox=tuple(int(value) for value in region.bbox),
            )
        )
    return tuple(shapes)


class ShapeRunner:
    """Cached model runner for repeated locate calls with one model."""

    def __init__(self, model_path: str, detector: Optional[ShapeLikeDetector] = None) -> None:
        self.model_path = str(model_path)
        self.detector = detector or load_detector_model(self.model_path)

    def locate(
        self,
        scene_bgr: np.ndarray,
        recipe: ShapeRecipe,
        *,
        ref_img_path: str = "",
        scene_mask: Optional[np.ndarray] = None,
    ) -> ShapeLocateResult:
        if scene_bgr is None:
            raise ValueError("scene_bgr is required")
        if not recipe.model_path:
            recipe.model_path = self.model_path
        if (not ref_img_path) and recipe.reference_image:
            ref_img_path = recipe.reference_image
        locate_t0 = time.perf_counter()
        result = locate_and_follow(
            scene_bgr,
            ref_img_path,
            recipe,
            detector=self.detector,
            scene_mask=scene_mask,
        )
        locate_ms = (time.perf_counter() - locate_t0) * 1000.0
        return ShapeLocateResult(result=result, locate_ms=locate_ms)

    def locate_runtime_shapes(
        self,
        scene_bgr: np.ndarray,
        recipe: ShapeRecipe,
        *,
        ref_img_path: str = "",
        scene_mask: Optional[np.ndarray] = None,
    ) -> ShapeRuntimeResult:
        total_t0 = time.perf_counter()
        locate = self.locate(
            scene_bgr,
            recipe,
            ref_img_path=ref_img_path,
            scene_mask=scene_mask,
        )
        total_ms = (time.perf_counter() - total_t0) * 1000.0
        return ShapeRuntimeResult(
            result=locate.result,
            roi_shapes=runtime_shapes_from_follow_result(locate.result),
            locate_ms=locate.locate_ms,
            total_ms=total_ms,
        )


class ShapeLocateService:
    """Application-facing service that owns detector caching and locate orchestration."""

    def __init__(self) -> None:
        self._runners: dict[str, ShapeRunner] = {}
        self._model_fingerprints: dict[str, tuple[int, int, int]] = {}
        self._cache_lock = threading.RLock()

    @staticmethod
    def _cache_key(model_path: str) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(model_path)))

    @staticmethod
    def _model_fingerprint(model_path: str) -> tuple[int, int, int]:
        stat = Path(model_path).stat()
        return int(stat.st_mtime_ns), int(stat.st_ctime_ns), int(stat.st_size)

    def runner_for_model(self, model_path: str) -> ShapeRunner:
        key = self._cache_key(model_path)
        fingerprint = self._model_fingerprint(key)
        with self._cache_lock:
            runner = self._runners.get(key)
            if runner is not None and self._model_fingerprints.get(key) == fingerprint:
                return runner

        # Model files are saved atomically. Still verify the fingerprint around
        # loading so an external replacement cannot associate an old detector
        # with the new file's cache entry.
        candidate: ShapeRunner | None = None
        loaded_fingerprint = fingerprint
        for _attempt in range(3):
            fingerprint_before = self._model_fingerprint(key)
            candidate = ShapeRunner(key)
            fingerprint_after = self._model_fingerprint(key)
            loaded_fingerprint = fingerprint_after
            if fingerprint_before == fingerprint_after:
                break
        else:
            raise RuntimeError(f"shape model changed repeatedly while loading: {key}")

        assert candidate is not None
        with self._cache_lock:
            current = self._runners.get(key)
            if current is not None and self._model_fingerprints.get(key) == loaded_fingerprint:
                return current
            self._runners[key] = candidate
            self._model_fingerprints[key] = loaded_fingerprint
            return candidate

    def invalidate_model(self, model_path: str) -> bool:
        key = self._cache_key(model_path)
        with self._cache_lock:
            removed = self._runners.pop(key, None) is not None
            self._model_fingerprints.pop(key, None)
        return removed

    def locate(
        self,
        scene_bgr: np.ndarray,
        recipe: ShapeRecipe,
        *,
        ref_img_path: str = "",
        scene_mask: Optional[np.ndarray] = None,
    ) -> ShapeLocateResult:
        if not recipe.model_path:
            raise ValueError("recipe.model_path is required")
        return self.runner_for_model(recipe.model_path).locate(
            scene_bgr,
            recipe,
            ref_img_path=ref_img_path,
            scene_mask=scene_mask,
        )

    def locate_runtime_shapes(
        self,
        scene_bgr: np.ndarray,
        recipe: ShapeRecipe,
        *,
        ref_img_path: str = "",
        scene_mask: Optional[np.ndarray] = None,
    ) -> ShapeRuntimeResult:
        if not recipe.model_path:
            raise ValueError("recipe.model_path is required")
        return self.runner_for_model(recipe.model_path).locate_runtime_shapes(
            scene_bgr,
            recipe,
            ref_img_path=ref_img_path,
            scene_mask=scene_mask,
        )

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._runners.clear()
            self._model_fingerprints.clear()


__all__ = [
    "RuntimeDetectedShape",
    "ShapeLocateResult",
    "ShapeLocateService",
    "ShapeRunner",
    "ShapeRuntimeResult",
    "runtime_shapes_from_follow_result",
]


