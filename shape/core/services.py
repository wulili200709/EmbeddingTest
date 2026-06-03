from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .recipe import Line2DupRecipe
from .roi_follow import FollowResult, locate_and_follow
from ..like_matcher import Line2DupLikeDetector, load_detector_model


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

    def __init__(self, model_path: str, detector: Optional[Line2DupLikeDetector] = None) -> None:
        self.model_path = str(model_path)
        self.detector = detector or load_detector_model(self.model_path)

    def locate(
        self,
        scene_bgr: np.ndarray,
        recipe: Line2DupRecipe,
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
        recipe: Line2DupRecipe,
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

    def runner_for_model(self, model_path: str) -> ShapeRunner:
        key = str(model_path)
        runner = self._runners.get(key)
        if runner is None:
            runner = ShapeRunner(key)
            self._runners[key] = runner
        return runner

    def locate(
        self,
        scene_bgr: np.ndarray,
        recipe: Line2DupRecipe,
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
        recipe: Line2DupRecipe,
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
        self._runners.clear()


__all__ = [
    "RuntimeDetectedShape",
    "ShapeLocateResult",
    "ShapeLocateService",
    "ShapeRunner",
    "ShapeRuntimeResult",
    "runtime_shapes_from_follow_result",
]
