from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass
class NccMatchRect:
    x: int = 0
    y: int = 0
    width: int = 1
    height: int = 1

    def normalized(self) -> "NccMatchRect":
        return NccMatchRect(
            x=max(0, int(self.x)),
            y=max(0, int(self.y)),
            width=max(1, int(self.width)),
            height=max(1, int(self.height)),
        )

    def to_xywh(self) -> Tuple[int, int, int, int]:
        rect = self.normalized()
        return rect.x, rect.y, rect.width, rect.height


@dataclass
class NccReferenceRegion:
    label_name: str = ""
    display_name: str = ""
    shape_type: str = "rectangle"
    points: List[Tuple[float, float]] = field(default_factory=list)

    def normalized(self) -> "NccReferenceRegion":
        shape_type = str(self.shape_type or "rectangle").strip().lower()
        if shape_type not in {"rectangle", "polygon"}:
            shape_type = "rectangle"
        points: List[Tuple[float, float]] = []
        for item in list(self.points or []):
            if not isinstance(item, Sequence) or len(item) < 2:
                continue
            points.append((float(item[0]), float(item[1])))
        if shape_type == "polygon":
            points = points[: max(3, len(points))]
        else:
            points = points[:2]
        label_name = str(self.label_name or "").strip()
        display_name = str(self.display_name or "").strip() or label_name
        return NccReferenceRegion(
            label_name=label_name,
            display_name=display_name,
            shape_type=shape_type,
            points=points,
        )


@dataclass
class NccAngleRange:
    start: float = -180.0
    end: float = 180.0


@dataclass
class NccAngleSearch:
    mode: str = "ranges"
    tolerance_angle: float = 0.0
    ranges: List[NccAngleRange] = field(default_factory=lambda: [NccAngleRange(-180.0, 180.0)])


@dataclass
class NccMatchOptions:
    target_num: int = 1
    max_overlap: float = 0.3
    score_threshold: float = 0.75
    angle_search: NccAngleSearch = field(default_factory=NccAngleSearch)
    min_reduced_area: int = 256
    use_simd: bool = True
    use_subpixel: bool = False
    bitwise_not: bool = False
    stop_layer1: bool = False

    def normalized(self) -> "NccMatchOptions":
        ranges = self.angle_search.ranges or [NccAngleRange(-180.0, 180.0)]
        normalized_ranges = []
        for item in ranges:
            start = float(item.start)
            end = float(item.end)
            if end < start:
                start, end = end, start
            normalized_ranges.append(NccAngleRange(start=start, end=end))
        return NccMatchOptions(
            target_num=max(1, int(self.target_num)),
            max_overlap=max(0.0, min(1.0, float(self.max_overlap))),
            score_threshold=max(0.0, min(1.0, float(self.score_threshold))),
            angle_search=NccAngleSearch(
                mode=str(self.angle_search.mode or "ranges"),
                tolerance_angle=max(0.0, float(self.angle_search.tolerance_angle)),
                ranges=normalized_ranges,
            ),
            min_reduced_area=max(1, int(self.min_reduced_area)),
            use_simd=bool(self.use_simd),
            use_subpixel=bool(self.use_subpixel),
            bitwise_not=bool(self.bitwise_not),
            stop_layer1=bool(self.stop_layer1),
        )

    def to_native_payload(self) -> Dict[str, Any]:
        options = self.normalized()
        return {
            "target_num": options.target_num,
            "max_overlap": options.max_overlap,
            "score_threshold": options.score_threshold,
            "angle_search": {
                "mode": options.angle_search.mode,
                "tolerance_angle": options.angle_search.tolerance_angle,
                "ranges": [
                    {"start": angle.start, "end": angle.end}
                    for angle in options.angle_search.ranges
                ],
            },
            "min_reduced_area": options.min_reduced_area,
            "use_simd": options.use_simd,
            "use_subpixel": options.use_subpixel,
            "bitwise_not": options.bitwise_not,
            "stop_layer1": options.stop_layer1,
        }


@dataclass
class NccMatchModel:
    schema: str = "ncc_match_model/3"
    display_name: str = "NCC Position Correction"
    source_image_path: str = "Source/source.png"
    template_image_path: str = "Template/template.png"
    preview_image_path: str = "Preview/template_preview.png"
    mask_image_path: str = "Mask/template_mask.png"
    template_mask_enabled: bool = False
    template_roi: NccMatchRect = field(default_factory=NccMatchRect)
    template_mask: NccReferenceRegion | None = None
    search_roi: NccMatchRect | None = None
    reference_regions: List[NccReferenceRegion] = field(default_factory=list)
    options: NccMatchOptions = field(default_factory=NccMatchOptions)

    def normalized(self) -> "NccMatchModel":
        return NccMatchModel(
            schema=str(self.schema or "ncc_match_model/3"),
            display_name=str(self.display_name or "NCC Position Correction"),
            source_image_path=str(self.source_image_path or "Source/source.png"),
            template_image_path=str(self.template_image_path or "Template/template.png"),
            preview_image_path=str(self.preview_image_path or "Preview/template_preview.png"),
            mask_image_path=str(self.mask_image_path or "Mask/template_mask.png"),
            template_mask_enabled=bool(self.template_mask_enabled),
            template_roi=self.template_roi.normalized(),
            template_mask=self.template_mask.normalized() if isinstance(self.template_mask, NccReferenceRegion) else None,
            search_roi=self.search_roi.normalized() if isinstance(self.search_roi, NccMatchRect) else None,
            reference_regions=[region.normalized() for region in list(self.reference_regions or [])],
            options=self.options.normalized(),
        )


@dataclass(frozen=True)
class NccMatchBoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class NccMatchResult:
    score: float
    angle: float
    center: Tuple[float, float]
    quad: Tuple[Tuple[float, float], ...]
    bbox: NccMatchBoundingBox


def create_default_model() -> NccMatchModel:
    return NccMatchModel().normalized()


def _get_value(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def _rect_from_any(value: Any) -> NccMatchRect:
    if isinstance(value, NccMatchRect):
        return value.normalized()
    if isinstance(value, dict):
        return NccMatchRect(
            x=int(_get_value(value, "x", default=0)),
            y=int(_get_value(value, "y", default=0)),
            width=int(_get_value(value, "width", "w", default=1)),
            height=int(_get_value(value, "height", "h", default=1)),
        ).normalized()
    if isinstance(value, Sequence) and len(value) >= 4:
        return NccMatchRect(
            x=int(value[0]),
            y=int(value[1]),
            width=int(value[2]),
            height=int(value[3]),
        ).normalized()
    return NccMatchRect()


def _optional_rect_from_any(value: Any) -> NccMatchRect | None:
    if value in (None, "", [], {}):
        return None
    rect = _rect_from_any(value)
    return rect.normalized()


def _bool_from_any(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return bool(default)


def _reference_region_from_any(value: Any) -> NccReferenceRegion | None:
    if isinstance(value, NccReferenceRegion):
        normalized = value.normalized()
        return normalized if normalized.label_name or normalized.points else None
    if not isinstance(value, dict):
        return None
    normalized = NccReferenceRegion(
        label_name=str(_get_value(value, "label_name", "labelName", "reference_label", "output_label", default="")),
        display_name=str(_get_value(value, "display_name", "displayName", "name", default="")),
        shape_type=str(_get_value(value, "shape_type", "shapeType", default="rectangle")),
        points=[
            (float(item[0]), float(item[1]))
            for item in list(_get_value(value, "points", default=[]))
            if isinstance(item, Sequence) and len(item) >= 2
        ],
    ).normalized()
    return normalized if normalized.label_name or normalized.points else None


def _reference_regions_from_any(value: Any) -> List[NccReferenceRegion]:
    if not isinstance(value, Iterable):
        return []
    regions: List[NccReferenceRegion] = []
    for item in value:
        region = _reference_region_from_any(item)
        if region is not None:
            regions.append(region)
    return regions


def _angle_ranges_from_any(value: Any) -> List[NccAngleRange]:
    if not isinstance(value, Iterable):
        return [NccAngleRange(-180.0, 180.0)]
    ranges: List[NccAngleRange] = []
    for item in value:
        if isinstance(item, NccAngleRange):
            ranges.append(item)
            continue
        if isinstance(item, dict):
            ranges.append(
                NccAngleRange(
                    start=float(_get_value(item, "start", default=-180.0)),
                    end=float(_get_value(item, "end", default=180.0)),
                )
            )
            continue
        if isinstance(item, Sequence) and len(item) >= 2:
            ranges.append(NccAngleRange(start=float(item[0]), end=float(item[1])))
    return ranges or [NccAngleRange(-180.0, 180.0)]


def _angle_search_from_any(value: Any) -> NccAngleSearch:
    if isinstance(value, NccAngleSearch):
        return value
    if not isinstance(value, dict):
        return NccAngleSearch()
    return NccAngleSearch(
        mode=str(_get_value(value, "mode", default="ranges")),
        tolerance_angle=float(_get_value(value, "tolerance_angle", "toleranceAngle", default=0.0)),
        ranges=_angle_ranges_from_any(_get_value(value, "ranges", default=[])),
    )


def _options_from_any(value: Any) -> NccMatchOptions:
    if isinstance(value, NccMatchOptions):
        return value.normalized()
    if not isinstance(value, dict):
        return NccMatchOptions().normalized()
    return NccMatchOptions(
        target_num=int(_get_value(value, "target_num", "targetNum", default=1)),
        max_overlap=float(_get_value(value, "max_overlap", "maxOverlap", default=0.3)),
        score_threshold=float(_get_value(value, "score_threshold", "scoreThreshold", default=0.75)),
        angle_search=_angle_search_from_any(_get_value(value, "angle_search", "angleSearch", default={})),
        min_reduced_area=int(_get_value(value, "min_reduced_area", "minReducedArea", default=256)),
        use_simd=bool(_get_value(value, "use_simd", "useSimd", default=True)),
        use_subpixel=bool(_get_value(value, "use_subpixel", "useSubpixel", default=False)),
        bitwise_not=bool(_get_value(value, "bitwise_not", "bitwiseNot", default=False)),
        stop_layer1=bool(_get_value(value, "stop_layer1", "stopLayer1", default=False)),
    ).normalized()


def normalize_model(model: NccMatchModel) -> NccMatchModel:
    return model.normalized()


def load_model(model_path: str | Path) -> NccMatchModel:
    path = Path(model_path)
    if not path.exists():
        return create_default_model()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    template_mask = _reference_region_from_any(_get_value(data, "template_mask", "templateMask", default=None))
    raw_template_mask_enabled = _get_value(
        data,
        "template_mask_enabled",
        "templateMaskEnabled",
        default=None,
    )
    model = NccMatchModel(
        schema=str(_get_value(data, "schema", default="ncc_match_model/3")),
        display_name=str(_get_value(data, "display_name", "displayName", default="NCC Position Correction")),
        source_image_path=str(_get_value(data, "source_image_path", "sourceImagePath", default="Source/source.png")),
        template_image_path=str(_get_value(data, "template_image_path", "templateImagePath", default="Template/template.png")),
        preview_image_path=str(_get_value(data, "preview_image_path", "previewImagePath", default="Preview/template_preview.png")),
        mask_image_path=str(_get_value(data, "mask_image_path", "maskImagePath", default="Mask/template_mask.png")),
        template_mask_enabled=_bool_from_any(
            raw_template_mask_enabled,
            default=template_mask is not None,
        ),
        template_roi=_rect_from_any(_get_value(data, "template_roi", "templateRoi", default={})),
        template_mask=template_mask,
        search_roi=_optional_rect_from_any(_get_value(data, "search_roi", "searchRoi", default=None)),
        reference_regions=_reference_regions_from_any(_get_value(data, "reference_regions", "referenceRegions", default=[])),
        options=_options_from_any(_get_value(data, "options", default={})),
    )
    return model.normalized()


def save_model(model_path: str | Path, model: NccMatchModel) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = model.normalized()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(normalized), handle, ensure_ascii=False, indent=2)


def resolve_asset_path(model_path: str | Path, raw_path: str) -> Path:
    base = Path(model_path).resolve().parent
    asset = Path(str(raw_path or "").strip())
    if asset.is_absolute():
        return asset
    return (base / asset).resolve()


def model_summary(model: NccMatchModel) -> str:
    normalized = model.normalized()
    rect = normalized.template_roi
    options = normalized.options
    ranges = ", ".join(
        f"[{angle.start:.1f}, {angle.end:.1f}]"
        for angle in options.angle_search.ranges
    )
    return "\n".join(
        [
            f"Name: {normalized.display_name}",
            f"Template ROI: x={rect.x}, y={rect.y}, w={rect.width}, h={rect.height}",
            (
                f"Template Mask: enabled ({normalized.template_mask.shape_type})"
            )
            if normalized.template_mask_enabled and isinstance(normalized.template_mask, NccReferenceRegion)
            else ("Template Mask: enabled" if normalized.template_mask_enabled else "Template Mask: disabled"),
            (
                f"Search ROI: x={normalized.search_roi.x}, y={normalized.search_roi.y}, "
                f"w={normalized.search_roi.width}, h={normalized.search_roi.height}"
            )
            if normalized.search_roi is not None
            else "Search ROI: full image",
            f"Reference ROI Count: {len(normalized.reference_regions)}",
            f"Threshold: {options.score_threshold:.3f}",
            f"Targets: {options.target_num}",
            f"Max overlap: {options.max_overlap:.2f}",
            f"Angles: {ranges}",
        ]
    )


__all__ = [
    "NccAngleRange",
    "NccAngleSearch",
    "NccMatchBoundingBox",
    "NccMatchModel",
    "NccMatchOptions",
    "NccMatchRect",
    "NccMatchResult",
    "NccReferenceRegion",
    "create_default_model",
    "load_model",
    "model_summary",
    "normalize_model",
    "resolve_asset_path",
    "save_model",
]
