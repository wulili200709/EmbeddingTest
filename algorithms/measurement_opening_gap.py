"""Measure a dark opening between a bright upper and lower edge.

Coordinates are in the original image.  Each opening supplies narrow, separate
search bands for the two edges so unrelated gradients cannot win the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class OpeningGapRegion:
    name: str
    center_x: int
    upper_y: tuple[int, int]
    lower_y: tuple[int, int]
    width: int = 49


@dataclass(frozen=True)
class OpeningGapResult:
    name: str
    center_x: int
    measurement_x: int
    distance_px: float
    upper_y: float
    lower_y: float
    valid_columns: int
    total_columns: int
    mad_px: float
    upper_strength: float
    lower_strength: float
    column_edges: tuple[tuple[int, float, float], ...] = ()

    def to_dict(self) -> dict:
        return {
            "type": "opening_gap_median",
            "name": self.name,
            "center_x": self.center_x,
            "measurement_x": self.measurement_x,
            "distance_px": self.distance_px,
            "upper_y": self.upper_y,
            "lower_y": self.lower_y,
            "valid_columns": self.valid_columns,
            "total_columns": self.total_columns,
            "mad_px": self.mad_px,
            "upper_strength": self.upper_strength,
            "lower_strength": self.lower_strength,
            "dimension_segment": [[float(self.measurement_x), self.upper_y], [float(self.measurement_x), self.lower_y]],
            "edge_segments": [
                [[float(self.measurement_x - 12), self.upper_y], [float(self.measurement_x + 12), self.upper_y]],
                [[float(self.measurement_x - 12), self.lower_y], [float(self.measurement_x + 12), self.lower_y]],
            ],
        }


def _peak_subpixel(response: np.ndarray, lo: int, hi: int, valid: np.ndarray | None = None) -> tuple[float, float]:
    candidates = response[lo:hi]
    if valid is not None:
        candidates = np.where(valid[lo:hi], candidates, -np.inf)
    index = int(lo + np.argmax(candidates))
    center = float(response[index])
    if valid is not None and not np.isfinite(candidates[index - lo]):
        return float("nan"), float("-inf")
    offset = 0.0
    if 0 < index < len(response) - 1:
        left, right = float(response[index - 1]), float(response[index + 1])
        denominator = left - 2.0 * center + right
        if denominator < -1e-6:
            offset = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    return float(index + 0.5 + offset), center


def _dark_plateau_edge(
    profile: np.ndarray,
    derivative: np.ndarray,
    lo: int,
    hi: int,
    *,
    descending: bool,
    fraction: float,
) -> tuple[float, float]:
    """Locate the dark end of a transition, using its local light/dark levels."""
    response = -derivative if descending else derivative
    peak = int(lo + np.argmax(response[lo:hi]))
    strength = float(response[peak])
    if descending:
        dark = profile[peak + 1:hi + 1]
        light = profile[lo:peak + 1]
    else:
        dark = profile[lo:peak + 1]
        light = profile[peak + 1:hi + 1]
    if dark.size == 0 or light.size == 0:
        return float("nan"), float("-inf")
    dark_level = float(np.min(dark))
    light_level = float(np.max(light))
    if light_level - dark_level < 10.0:
        return float("nan"), float("-inf")
    threshold = dark_level + fraction * (light_level - dark_level)
    if descending:
        indices = np.flatnonzero(profile[peak + 1:hi + 1] <= threshold)
        if not indices.size:
            return float("nan"), float("-inf")
        right = int(peak + 1 + indices[0])
        left = right - 1
    else:
        indices = np.flatnonzero(profile[lo:peak + 1] <= threshold)
        if not indices.size:
            return float("nan"), float("-inf")
        left = int(lo + indices[-1])
        right = left + 1
    denominator = float(profile[right] - profile[left])
    if abs(denominator) < 1e-6:
        return float(right if descending else left), strength
    position = float(left + np.clip((threshold - float(profile[left])) / denominator, 0.0, 1.0))
    return position, strength


def measure_opening_gap(
    image_bgr: np.ndarray,
    region: OpeningGapRegion,
    *,
    blur_ksize: int = 5,
    min_strength: float = 5.0,
    min_valid_ratio: float = 0.6,
    edge_dark_max_gray: float | None = None,
    edge_mode: str = "peak",
    upper_dark_fraction: float = 0.03,
    lower_dark_fraction: float = 0.03,
    include_column_edges: bool = False,
) -> OpeningGapResult:
    """Return the median of paired vertical edge distances in the center columns."""
    image = np.asarray(image_bgr)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("image must be grayscale or BGR")
    height, image_width = gray.shape
    width = int(region.width)
    if width < 3 or width % 2 != 1:
        raise ValueError("opening width must be an odd integer >= 3")
    x0 = int(region.center_x) - width // 2
    x1 = x0 + width
    upper_lo, upper_hi = map(int, region.upper_y)
    lower_lo, lower_hi = map(int, region.lower_y)
    if not (0 <= x0 < x1 <= image_width and 0 <= upper_lo < upper_hi < lower_lo < lower_hi <= height - 1):
        raise ValueError(f"invalid opening region: {region}")
    ksize = int(blur_ksize)
    if ksize < 1 or ksize % 2 != 1:
        raise ValueError("blur_ksize must be a positive odd integer")
    y0 = max(0, upper_lo - ksize)
    y1 = min(height, lower_hi + ksize + 2)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    if ksize >= 3:
        patch = cv2.GaussianBlur(patch, (1, ksize), 0)
    derivative = np.diff(patch, axis=0)
    if edge_dark_max_gray is not None and not 0 <= edge_dark_max_gray <= 255:
        raise ValueError("edge_dark_max_gray must be between 0 and 255")
    if edge_mode not in {"peak", "dark_plateau"}:
        raise ValueError("edge_mode must be peak or dark_plateau")
    if not 0.0 < upper_dark_fraction < 1.0 or not 0.0 < lower_dark_fraction < 1.0:
        raise ValueError("dark fractions must be between 0 and 1")
    distances, uppers, lowers, columns, top_scores, bottom_scores = [], [], [], [], [], []
    for column in range(width):
        delta = derivative[:, column]
        if edge_mode == "dark_plateau":
            upper, top_score = _dark_plateau_edge(
                patch[:, column], delta, upper_lo - y0, upper_hi - y0,
                descending=True, fraction=upper_dark_fraction,
            )
            lower, bottom_score = _dark_plateau_edge(
                patch[:, column], delta, lower_lo - y0, lower_hi - y0,
                descending=False, fraction=lower_dark_fraction,
            )
        else:
            upper_valid = lower_valid = None
            if edge_dark_max_gray is not None:
                upper_valid = patch[1:, column] <= edge_dark_max_gray
                lower_valid = patch[:-1, column] <= edge_dark_max_gray
            upper, top_score = _peak_subpixel(-delta, upper_lo - y0, upper_hi - y0, upper_valid)
            lower, bottom_score = _peak_subpixel(delta, lower_lo - y0, lower_hi - y0, lower_valid)
        if top_score < min_strength or bottom_score < min_strength:
            continue
        uppers.append(upper + y0)
        lowers.append(lower + y0)
        distances.append(lower - upper)
        columns.append(column)
        top_scores.append(top_score)
        bottom_scores.append(bottom_score)
    minimum = max(3, int(np.ceil(width * min_valid_ratio)))
    if len(distances) < minimum:
        raise RuntimeError(f"{region.name}: valid columns {len(distances)}/{width}, need {minimum}")
    values = np.asarray(distances, dtype=np.float32)
    distance = float(np.median(values))
    order = np.argsort(values)
    median_indices = order[(len(order) - 1) // 2:len(order) // 2 + 1]
    measurement_x = int(region.center_x + round(float(np.mean([columns[i] for i in median_indices])) - width // 2))
    upper_y = float(np.mean([uppers[i] for i in median_indices]))
    lower_y = float(np.mean([lowers[i] for i in median_indices]))
    return OpeningGapResult(
        name=region.name,
        center_x=region.center_x,
        measurement_x=measurement_x,
        distance_px=distance,
        upper_y=upper_y,
        lower_y=lower_y,
        valid_columns=len(distances),
        total_columns=width,
        mad_px=float(np.median(np.abs(values - distance))),
        upper_strength=float(np.median(top_scores)),
        lower_strength=float(np.median(bottom_scores)),
        column_edges=(
            tuple((x0 + column, float(upper), float(lower)) for column, upper, lower in zip(columns, uppers, lowers))
            if include_column_edges else ()
        ),
    )


def regions_from_params(params: Mapping) -> list[OpeningGapRegion]:
    regions = []
    for index, raw in enumerate(params.get("regions", ())):
        regions.append(OpeningGapRegion(
            name=str(raw.get("name", f"opening_{index + 1}")),
            center_x=int(raw["center_x"]),
            upper_y=tuple(raw["upper_y"]),
            lower_y=tuple(raw["lower_y"]),
            width=int(raw.get("width", 49)),
        ))
    if not regions:
        raise ValueError("opening gap measurement requires at least one region")
    return regions


def measure_opening_gaps(image_bgr: np.ndarray, regions: Sequence[OpeningGapRegion], **kwargs) -> list[OpeningGapResult]:
    return [measure_opening_gap(image_bgr, region, **kwargs) for region in regions]


def measure_opening_gap_from_params(image_bgr: np.ndarray, params: Mapping) -> OpeningGapResult:
    raw = params.get("region")
    if not isinstance(raw, Mapping):
        raise ValueError("opening_gap_median requires params.region")
    region = OpeningGapRegion(
        name=str(raw.get("name", "opening")),
        center_x=int(raw["center_x"]),
        upper_y=tuple(raw["upper_y"]),
        lower_y=tuple(raw["lower_y"]),
        width=int(raw.get("width", 49)),
    )
    return measure_opening_gap(
        image_bgr,
        region,
        blur_ksize=int(params.get("blur_ksize", 5)),
        min_strength=float(params.get("min_strength", 5.0)),
        min_valid_ratio=float(params.get("min_valid_ratio", 0.6)),
        edge_dark_max_gray=(float(params["edge_dark_max_gray"]) if params.get("edge_dark_max_gray") is not None else None),
        edge_mode=str(params.get("edge_mode", "peak")),
        upper_dark_fraction=float(params.get("upper_dark_fraction", 0.03)),
        lower_dark_fraction=float(params.get("lower_dark_fraction", 0.03)),
    )


def measure_opening_gap_file(image_path: str, params: Mapping) -> OpeningGapResult:
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read opening gap image: {image_path}")
    return measure_opening_gap_from_params(image, params)


def judge_opening_gap(result: OpeningGapResult, params: Mapping) -> tuple[str, float, float | None, float | None, str]:
    unit = str(params.get("limit_unit", "px") or "px").lower()
    if unit not in {"px", "mm"}:
        raise ValueError("limit_unit must be px or mm")
    scale = float(params.get("pixel_size_mm", 0.0) or 0.0)
    if unit == "mm" and scale <= 0:
        raise ValueError("pixel_size_mm is required for mm limits")
    value = result.distance_px * scale if unit == "mm" else result.distance_px
    lower_raw = params.get("lower_limit", params.get(f"lower_limit_{unit}"))
    upper_raw = params.get("upper_limit", params.get(f"upper_limit_{unit}"))
    lower = float(lower_raw) if lower_raw is not None else None
    upper = float(upper_raw) if upper_raw is not None else None
    ok = (lower is None or value >= lower) and (upper is None or value <= upper)
    return "OK" if ok else "NG", float(value), lower, upper, unit
