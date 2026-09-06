from __future__ import annotations

import os
from typing import Any, Mapping

import cv2
import numpy as np

from algorithms.measurement_types import (
    BRIGHT_BLOCK_CENTER_ALGORITHM,
    BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM,
    CENTER_DISTANCE_ALGORITHM,
    CENTER_DISTANCE_ALGORITHMS,
    FIND_LINE_ALGORITHM,
    FIND_LINE_ALGORITHMS,
    FIND_LINE_SUBPIX_ALGORITHM,
    LINE_DISTANCE_ALGORITHM,
    LINE_DISTANCE_ALGORITHMS,
    LINE_DISTANCE_REF_NORMAL_ALGORITHM,
    MEASUREMENT_ALGORITHMS,
    MULTI_PIN_TIP_HEIGHT_ALGORITHM,
    PIN_CENTER_DISTANCE_ALGORITHM,
    PIN_TIP_POINT_ALGORITHM,
    POINT_LINE_DISTANCE_ALGORITHM,
    BrightBlockCenterResult,
    EdgeDistanceConfig,
    EdgeDistanceResult,
    FindLineConfig,
    FindLineMeasurementConfig,
    FindLineMeasurementResult,
    FittedLine,
    PinCenterCandidate,
    PinCenterDistanceConfig,
    PinCenterDistanceResult,
    PinTipPointConfig,
    PinTipPointResult,
    MultiPinTipHeightConfig,
    MultiPinTipHeightResult,
    is_measurement_algorithm,
)
from algorithms.measurement_lines import (
    _angle_delta_deg,
    _line_angle_deg,
    _line_distance_px,
    _line_position_px,
    _line_segment_in_crop,
    filter_line_points,
    find_edge_points,
    fit_line,
    fit_line_filtered,
)
from algorithms.measurement_pin_center import (
    _bright_block_center_has_adjacent_body,
    _find_bright_vertical_block_candidates,
    _find_pin_inner_strip_candidates,
    _normalized_block_orientation,
    _pin_center_binary,
    _pin_center_distance_px,
    _pin_pair_min_separation,
    _select_bright_block_center_candidate,
    _select_bright_block_y_pair,
    _select_pin_center_pair,
)
from algorithms.measurement_pin_tip import locate_pin_tip_in_crop
from algorithms.measurement_multi_pin_tip import locate_multi_pin_tips_in_crop


def _shape_from_labels(shape_by_label: Mapping[str, dict], preferred_label: str) -> tuple[str, dict]:
    label = str(preferred_label or "").strip() or "roi1"
    shape = dict(shape_by_label or {}).get(label)
    if shape is None:
        label = "roi"
        shape = dict(shape_by_label or {}).get(label)
    if shape is None:
        raise RuntimeError(f"measurement ROI missing: {preferred_label or 'roi1'}")
    return label, shape


def _crop_from_shape(image_bgr: np.ndarray, shape: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image = np.ascontiguousarray(np.asarray(image_bgr))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"unsupported image shape: {image.shape!r}")
    image = image[:, :, :3]
    h_img, w_img = image.shape[:2]
    pts = np.asarray(shape.get("points", []), dtype=np.float32)
    if pts.size == 0:
        raise RuntimeError("measurement ROI points empty")
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    x = max(0, int(np.floor(float(x_min))))
    y = max(0, int(np.floor(float(y_min))))
    x2 = min(w_img, int(np.ceil(float(x_max))))
    y2 = min(h_img, int(np.ceil(float(y_max))))
    if x2 <= x or y2 <= y:
        raise RuntimeError("measurement ROI bbox invalid")

    crop = image[y:y2, x:x2].copy()
    mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
    rel_pts = pts - np.array([[x, y]], dtype=np.float32)
    if str(shape.get("shape_type", "rectangle")) == "polygon" and len(rel_pts) >= 3:
        cv2.fillPoly(mask, [np.round(rel_pts).astype(np.int32)], 255)
    else:
        p0 = rel_pts.min(axis=0)
        p1 = rel_pts.max(axis=0)
        rx = max(0, int(np.floor(float(p0[0]))))
        ry = max(0, int(np.floor(float(p0[1]))))
        rx2 = min(mask.shape[1], int(np.ceil(float(p1[0]))))
        ry2 = min(mask.shape[0], int(np.ceil(float(p1[1]))))
        mask[ry:ry2, rx:rx2] = 255
    return crop, mask, (x, y)


def measure_bright_block_center_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> BrightBlockCenterResult:
    payload = dict(params or {})
    orientation = _normalized_block_orientation(
        payload.get("block_orientation", payload.get("target_orientation", "auto"))
    )
    config = PinCenterDistanceConfig.from_params(
        {
            **payload,
            "target_orientation": "any",
            "distance_mode": "euclidean",
        },
        roi_label=preferred_label,
    )
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, origin = _crop_from_shape(image_bgr, shape)
    _binary, threshold = _pin_center_binary(crop, mask, config)
    horizontal_candidates = (
        _find_pin_inner_strip_candidates(
            crop,
            mask,
            origin=origin,
            config=config,
            rough_threshold=threshold,
        )
        if orientation in {"auto", "horizontal"}
        else ()
    )
    vertical_candidates = (
        _find_bright_vertical_block_candidates(
            crop,
            mask,
            origin=origin,
            config=config,
            rough_threshold=threshold,
        )
        if orientation in {"auto", "vertical"}
        else ()
    )
    candidate, candidates = _select_bright_block_center_candidate(
        horizontal_candidates,
        vertical_candidates,
        orientation=orientation,
        origin=origin,
        roi_size=(int(crop.shape[1]), int(crop.shape[0])),
    )
    if not _bright_block_center_has_adjacent_body(
        crop,
        mask,
        candidate,
        origin=origin,
        params=payload,
    ):
        raise RuntimeError("bright block adjacent metal body not found")
    ox, oy = origin
    return BrightBlockCenterResult(
        roi_label=roi_label,
        center_xy=candidate.center_xy,
        candidate=candidate,
        candidates=candidates,
        threshold=float(threshold),
        orientation=orientation,
        roi_xywh=(int(ox), int(oy), int(crop.shape[1]), int(crop.shape[0])),
    )


def measure_pin_tip_point_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> PinTipPointResult:
    config = PinTipPointConfig.from_params(params, roi_label=preferred_label)
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, origin = _crop_from_shape(image_bgr, shape)
    located = locate_pin_tip_in_crop(crop, mask, config=config)
    ox, oy = origin
    local_point = located["point_xy"]
    local_bbox = located["component_bbox_xywh"]
    local_edges = located["edge_points"]
    point_xy = (float(local_point[0]) + ox, float(local_point[1]) + oy)
    bbox_xywh = (
        int(local_bbox[0]) + ox,
        int(local_bbox[1]) + oy,
        int(local_bbox[2]),
        int(local_bbox[3]),
    )
    edge_points = tuple(
        (float(point[0]) + ox, float(point[1]) + oy)
        for point in local_edges
    )
    return PinTipPointResult(
        roi_label=roi_label,
        point_xy=point_xy,
        axis_direction=(
            float(located["axis_direction"][0]),
            float(located["axis_direction"][1]),
        ),
        threshold=float(located["threshold"]),
        confidence=float(located["confidence"]),
        fit_residual=float(located["fit_residual"]),
        component_area_px=float(located["component_area_px"]),
        component_bbox_xywh=bbox_xywh,
        edge_points=edge_points,
        roi_xywh=(int(ox), int(oy), int(crop.shape[1]), int(crop.shape[0])),
    )


def measure_multi_pin_tip_height_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> MultiPinTipHeightResult:
    config = MultiPinTipHeightConfig.from_params(params, roi_label=preferred_label)
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, origin = _crop_from_shape(image_bgr, shape)
    payload = dict(params or {})
    reference_segment = payload.get("_reference_line_segment")
    if str(payload.get("reference_line_item_id", "") or "").strip() and reference_segment is None:
        raise RuntimeError("selected reference line is missing, disabled or failed")
    if reference_segment is not None:
        endpoints = np.asarray(reference_segment, dtype=np.float64)
        if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
            raise RuntimeError("invalid reference line result")
        reference_segment = tuple(
            (float(x) - origin[0], float(y) - origin[1]) for x, y in endpoints
        )
    located = locate_multi_pin_tips_in_crop(
        crop, mask, config=config, reference_segment=reference_segment,
    )
    ox, oy = origin
    local_line = located["reference_line"]
    local_segment = located["reference_line_segment"]
    candidates = list(located["candidates"])
    return MultiPinTipHeightResult(
        roi_label=roi_label,
        expected_pin_count=config.expected_pin_count,
        tip_points=tuple(
            (float(candidate["point_xy"][0]) + ox, float(candidate["point_xy"][1]) + oy)
            for candidate in candidates
        ),
        distances_px=tuple(float(candidate["distance_px"]) for candidate in candidates),
        spacings_px=tuple(
            float(candidates[index + 1]["order_position"])
            - float(candidates[index]["order_position"])
            for index in range(max(0, len(candidates) - 1))
        ),
        reference_line=FittedLine(
            vx=float(local_line.vx),
            vy=float(local_line.vy),
            x0=float(local_line.x0) + ox,
            y0=float(local_line.y0) + oy,
            residual=float(local_line.residual),
            point_count=int(local_line.point_count),
        ),
        reference_line_segment=(
            (float(local_segment[0][0]) + ox, float(local_segment[0][1]) + oy),
            (float(local_segment[1][0]) + ox, float(local_segment[1][1]) + oy),
        ),
        threshold=float(located["threshold"]),
        fit_residuals=tuple(float(candidate["fit_residual"]) for candidate in candidates),
        component_boxes_xywh=tuple(
            (
                int(candidate["bbox_xywh"][0]) + ox,
                int(candidate["bbox_xywh"][1]) + oy,
                int(candidate["bbox_xywh"][2]),
                int(candidate["bbox_xywh"][3]),
            )
            for candidate in candidates
        ),
        roi_xywh=(int(ox), int(oy), int(crop.shape[1]), int(crop.shape[0])),
    )


def measure_pin_center_distance_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> PinCenterDistanceResult:
    config = PinCenterDistanceConfig.from_params(params, roi_label=preferred_label)
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, origin = _crop_from_shape(image_bgr, shape)
    binary, threshold = _pin_center_binary(crop, mask, config)
    direct_candidates = (
        _find_pin_inner_strip_candidates(
            crop,
            mask,
            origin=origin,
            config=config,
            rough_threshold=threshold,
        )
        if config.center_target == "inner_bright_strip"
        else ()
    )
    if len(direct_candidates) >= 2:
        candidates = direct_candidates
        pin_a, pin_b = _select_pin_center_pair(
            candidates,
            config=config,
            roi_size=(int(crop.shape[1]), int(crop.shape[0])),
        )
    else:
        candidates = _find_pin_center_candidates(binary, origin=origin, config=config)
        pin_a, pin_b = _select_pin_center_pair(
            candidates,
            config=config,
            roi_size=(int(crop.shape[1]), int(crop.shape[0])),
        )
        refine = (
            _refine_pin_candidate_from_local_edges
            if config.center_target == "metal_body"
            else _refine_pin_candidate_to_inner_bright_strip
        )
        pin_a = refine(crop, mask, pin_a, origin=origin, config=config, rough_threshold=threshold)
        pin_b = refine(crop, mask, pin_b, origin=origin, config=config, rough_threshold=threshold)
    axis_index = 0 if config.sort_axis == "x" else 1
    axis_span = float(max(1, int(crop.shape[1]) if axis_index == 0 else int(crop.shape[0])))
    final_separation = abs(float(pin_b.center_xy[axis_index]) - float(pin_a.center_xy[axis_index]))
    min_separation = _pin_pair_min_separation(
        pin_a,
        pin_b,
        axis_span=axis_span,
        axis_index=axis_index,
        config=config,
    )
    if final_separation < min_separation:
        raise RuntimeError(
            f"pin center pair separation too small after refinement: need >= {min_separation:.1f}px"
        )
    pin_a, pin_b = tuple(sorted((pin_a, pin_b), key=lambda item: item.center_xy[axis_index]))
    distance_px = _pin_center_distance_px(pin_a.center_xy, pin_b.center_xy, config.distance_mode)
    distance_mm = distance_px * config.pixel_size_mm if config.pixel_size_mm > 0.0 else None
    ox, oy = origin
    return PinCenterDistanceResult(
        roi_label=roi_label,
        distance_px=float(distance_px),
        distance_mm=float(distance_mm) if distance_mm is not None else None,
        center_a=pin_a.center_xy,
        center_b=pin_b.center_xy,
        candidates=(pin_a, pin_b),
        threshold=float(threshold),
        distance_mode=config.distance_mode,
        roi_xywh=(int(ox), int(oy), int(crop.shape[1]), int(crop.shape[0])),
    )


def measure_bright_block_y_distance_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> PinCenterDistanceResult:
    config = PinCenterDistanceConfig.from_params(
        {
            **dict(params or {}),
            "distance_mode": "vertical",
            "sort_axis": "y",
            "target_orientation": "any",
        },
        roi_label=preferred_label,
    )
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, origin = _crop_from_shape(image_bgr, shape)
    _binary, threshold = _pin_center_binary(crop, mask, config)
    horizontal_candidates = _find_pin_inner_strip_candidates(
        crop,
        mask,
        origin=origin,
        config=config,
        rough_threshold=threshold,
    )
    vertical_candidates = _find_bright_vertical_block_candidates(
        crop,
        mask,
        origin=origin,
        config=config,
        rough_threshold=threshold,
    )
    vertical, horizontal = _select_bright_block_y_pair(
        vertical_candidates,
        horizontal_candidates,
        roi_size=(int(crop.shape[1]), int(crop.shape[0])),
    )
    distance_px = abs(float(horizontal.center_xy[1]) - float(vertical.center_xy[1]))
    distance_mm = distance_px * config.pixel_size_mm if config.pixel_size_mm > 0.0 else None
    dimension_x = float(horizontal.center_xy[0])
    dimension_segment = (
        (dimension_x, float(vertical.center_xy[1])),
        (dimension_x, float(horizontal.center_xy[1])),
    )
    ox, oy = origin
    return PinCenterDistanceResult(
        roi_label=roi_label,
        distance_px=float(distance_px),
        distance_mm=float(distance_mm) if distance_mm is not None else None,
        center_a=vertical.center_xy,
        center_b=horizontal.center_xy,
        candidates=(vertical, horizontal),
        threshold=float(threshold),
        distance_mode="vertical",
        roi_xywh=(int(ox), int(oy), int(crop.shape[1]), int(crop.shape[0])),
        measurement_type=BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM,
        dimension_segment=dimension_segment,
    )


def measure_find_line_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
    algorithm: object = FIND_LINE_ALGORITHM,
) -> FindLineMeasurementResult:
    config = FindLineMeasurementConfig.from_params(params, roi_label=preferred_label, algorithm=algorithm)
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, origin = _crop_from_shape(image_bgr, shape)
    points = find_edge_points(crop, mask, config.line)
    context = (
        f"roi={roi_label} line direction={config.line.direction} "
        f"threshold={config.line.edge_threshold:.3f}"
    )
    line, fit_points = fit_line_filtered(points, min_points=config.line.min_points, context=context)
    position_px = _line_position_px(line, config.line.direction)
    position_mm = position_px * config.pixel_size_mm if config.pixel_size_mm > 0.0 else None
    ox, oy = origin
    absolute_points = tuple(
        (float(x + ox), float(y + oy))
        for x, y in np.asarray(fit_points, dtype=np.float32).reshape(-1, 2)
    )
    return FindLineMeasurementResult(
        roi_label=roi_label,
        line=line,
        position_px=float(position_px),
        position_mm=float(position_mm) if position_mm is not None else None,
        angle_deg=float(_line_angle_deg(line)),
        value_mode=config.value_mode,
        roi_xywh=(int(ox), int(oy), int(crop.shape[1]), int(crop.shape[0])),
        edge_points=absolute_points,
        line_segment=_line_segment_in_crop(
            line,
            crop_width=int(crop.shape[1]),
            crop_height=int(crop.shape[0]),
            origin=origin,
        ),
    )


def measure_edge_distance_from_array(
    image_bgr: np.ndarray,
    *,
    shape_by_label: Mapping[str, dict],
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> EdgeDistanceResult:
    config = EdgeDistanceConfig.from_params(params, roi_label=preferred_label)
    roi_label, shape = _shape_from_labels(shape_by_label, config.roi_label or preferred_label)
    crop, mask, _origin = _crop_from_shape(image_bgr, shape)
    points_a = find_edge_points(crop, mask, config.line_a)
    points_b = find_edge_points(crop, mask, config.line_b)
    context_a = (
        f"roi={roi_label} line_a direction={config.line_a.direction} "
        f"threshold={config.line_a.edge_threshold:.3f}"
    )
    context_b = (
        f"roi={roi_label} line_b direction={config.line_b.direction} "
        f"threshold={config.line_b.edge_threshold:.3f}"
    )
    line_a, _fit_points_a = fit_line_filtered(points_a, min_points=config.line_a.min_points, context=context_a)
    line_b, _fit_points_b = fit_line_filtered(points_b, min_points=config.line_b.min_points, context=context_b)
    distance_px = _line_distance_px(line_a, line_b)
    distance_mm = distance_px * config.pixel_size_mm if config.pixel_size_mm > 0.0 else None
    return EdgeDistanceResult(
        roi_label=roi_label,
        distance_px=float(distance_px),
        distance_mm=float(distance_mm) if distance_mm is not None else None,
        line_a=line_a,
        line_b=line_b,
        angle_delta_deg=float(_angle_delta_deg(line_a, line_b)),
    )


def measure_edge_distance(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> EdgeDistanceResult:
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    from common.labelme_io import labelme_json_of_image, read_shape_from_labelme

    jpath = labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")
    label = str(preferred_label or "").strip() or "roi1"
    shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        label = "roi"
        shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(img_path)
    return measure_edge_distance_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
    )


def measure_pin_center_distance(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> PinCenterDistanceResult:
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    from common.labelme_io import labelme_json_of_image, read_shape_from_labelme

    jpath = labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")
    label = str(preferred_label or "").strip() or "roi1"
    shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        label = "roi"
        shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(img_path)
    return measure_pin_center_distance_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
    )


def measure_bright_block_center(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> BrightBlockCenterResult:
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    from common.labelme_io import labelme_json_of_image, read_shape_from_labelme

    jpath = labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")
    label = str(preferred_label or "").strip() or "roi1"
    shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        label = "roi"
        shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(img_path)
    return measure_bright_block_center_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
    )


def measure_pin_tip_point(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> PinTipPointResult:
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    from common.labelme_io import labelme_json_of_image, read_shape_from_labelme

    jpath = labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")
    label = str(preferred_label or "").strip() or "roi1"
    shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        label = "roi"
        shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(img_path)
    return measure_pin_tip_point_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
    )


def measure_multi_pin_tip_height(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> MultiPinTipHeightResult:
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    from common.labelme_io import labelme_json_of_image, read_shape_from_labelme

    jpath = labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")
    label = str(preferred_label or "").strip() or "roi1"
    shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        label = "roi"
        shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(img_path)
    return measure_multi_pin_tip_height_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
    )


def measure_bright_block_y_distance(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
) -> PinCenterDistanceResult:
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    from common.labelme_io import labelme_json_of_image, read_shape_from_labelme

    jpath = labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")
    label = str(preferred_label or "").strip() or "roi1"
    shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        label = "roi"
        shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(img_path)
    return measure_bright_block_y_distance_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
    )


def measure_find_line(
    img_path: str,
    *,
    preferred_label: str = "roi1",
    params: Mapping[str, Any] | None = None,
    algorithm: object = FIND_LINE_ALGORITHM,
) -> FindLineMeasurementResult:
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    from common.labelme_io import labelme_json_of_image, read_shape_from_labelme

    jpath = labelme_json_of_image(img_path)
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"Missing labelme json: {jpath}")
    label = str(preferred_label or "").strip() or "roi1"
    shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        label = "roi"
        shape = read_shape_from_labelme(jpath, label)
    if shape is None:
        raise RuntimeError(f"{os.path.basename(img_path)} missing {preferred_label}/roi")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(img_path)
    return measure_find_line_from_array(
        image,
        shape_by_label={label: shape},
        preferred_label=label,
        params=params,
        algorithm=algorithm,
    )


def measurement_value(
    result: EdgeDistanceResult | FindLineMeasurementResult | PinCenterDistanceResult | BrightBlockCenterResult | PinTipPointResult | MultiPinTipHeightResult,
    algorithm: object,
) -> float:
    key = str(algorithm or "").strip().lower()
    if key == "edge_distance" and isinstance(result, EdgeDistanceResult):
        return float(result.distance_px)
    if key in FIND_LINE_ALGORITHMS and isinstance(result, FindLineMeasurementResult):
        return float(result.position_px)
    if key == PIN_CENTER_DISTANCE_ALGORITHM and isinstance(result, PinCenterDistanceResult):
        return float(result.distance_px)
    if key == BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM and isinstance(result, PinCenterDistanceResult):
        return float(result.distance_px)
    if key == BRIGHT_BLOCK_CENTER_ALGORITHM and isinstance(result, BrightBlockCenterResult):
        return float(result.center_xy[1])
    if key == PIN_TIP_POINT_ALGORITHM and isinstance(result, PinTipPointResult):
        return float(result.point_xy[1])
    if key == MULTI_PIN_TIP_HEIGHT_ALGORITHM and isinstance(result, MultiPinTipHeightResult):
        return max((float(value) for value in result.distances_px), default=0.0)
    raise ValueError(f"Unsupported measurement algorithm: {algorithm}")


def judge_find_line(
    result: FindLineMeasurementResult,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, float, float | None, float | None, str]:
    config = FindLineMeasurementConfig.from_params(params, roi_label=result.roi_label)
    unit = "deg" if config.value_mode == "angle" else "px"
    if config.value_mode == "angle":
        value = float(result.angle_deg)
    elif config.value_mode == "residual":
        value = float(result.line.residual)
    elif config.limit_unit == "mm":
        distance_mm = result.position_mm
        if distance_mm is None and config.pixel_size_mm > 0.0:
            distance_mm = float(result.position_px) * float(config.pixel_size_mm)
        if distance_mm is None:
            raise RuntimeError("pixel_size_mm is required when find-line limits use mm")
        value = float(distance_mm)
        unit = "mm"
    else:
        value = float(result.position_px)
    lower = config.lower_limit
    upper = config.upper_limit
    ok = True
    if lower is not None and value < lower:
        ok = False
    if upper is not None and value > upper:
        ok = False
    return ("OK" if ok else "NG"), value, lower, upper, unit


def judge_edge_distance(
    result: EdgeDistanceResult,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, float, float | None, float | None, str]:
    config = EdgeDistanceConfig.from_params(params, roi_label=result.roi_label)
    unit = config.limit_unit
    if unit == "mm":
        distance_mm = result.distance_mm
        if distance_mm is None and config.pixel_size_mm > 0.0:
            distance_mm = float(result.distance_px) * float(config.pixel_size_mm)
        if distance_mm is None:
            raise RuntimeError("pixel_size_mm is required when measurement limits use mm")
        value = float(distance_mm)
    else:
        unit = "px"
        value = float(result.distance_px)
    lower = config.lower_limit
    upper = config.upper_limit
    ok = True
    if lower is not None and value < lower:
        ok = False
    if upper is not None and value > upper:
        ok = False
    return ("OK" if ok else "NG"), value, lower, upper, unit


def judge_pin_center_distance(
    result: PinCenterDistanceResult,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, float, float | None, float | None, str]:
    config = PinCenterDistanceConfig.from_params(params, roi_label=result.roi_label)
    unit = config.limit_unit
    if unit == "mm":
        distance_mm = result.distance_mm
        if distance_mm is None and config.pixel_size_mm > 0.0:
            distance_mm = float(result.distance_px) * float(config.pixel_size_mm)
        if distance_mm is None:
            raise RuntimeError("pixel_size_mm is required when pin-center limits use mm")
        value = float(distance_mm)
    else:
        unit = "px"
        value = float(result.distance_px)
    lower = config.lower_limit
    upper = config.upper_limit
    ok = True
    if lower is not None and value < lower:
        ok = False
    if upper is not None and value > upper:
        ok = False
    return ("OK" if ok else "NG"), value, lower, upper, unit


def judge_multi_pin_tip_height(
    result: MultiPinTipHeightResult,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, tuple[float, ...], float | None, float | None, str, tuple[bool, ...]]:
    evaluation = evaluate_multi_pin_tip_height(result, params)
    return (
        str(evaluation["pred"]),
        tuple(evaluation["height_values"]),
        evaluation["height_lower_limit"],
        evaluation["height_upper_limit"],
        str(evaluation["unit"]),
        tuple(evaluation["height_in_spec"]),
    )


def evaluate_multi_pin_tip_height(
    result: MultiPinTipHeightResult,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = MultiPinTipHeightConfig.from_params(params, roi_label=result.roi_label)
    unit = config.limit_unit
    if unit == "mm":
        if config.pixel_size_mm <= 0.0:
            raise RuntimeError("pixel_size_mm is required when multi-pin limits use mm")
        height_values = tuple(float(value) * config.pixel_size_mm for value in result.distances_px)
        spacing_values = tuple(float(value) * config.pixel_size_mm for value in result.spacings_px)
    else:
        unit = "px"
        height_values = tuple(float(value) for value in result.distances_px)
        spacing_values = tuple(float(value) for value in result.spacings_px)
    height_in_spec = tuple(
        not config.height_check_enabled
        or (
            (config.lower_limit is None or value >= config.lower_limit)
            and (config.upper_limit is None or value <= config.upper_limit)
        )
        for value in height_values
    )
    spacing_results: list[dict[str, Any]] = []
    for index, value in enumerate(spacing_values):
        spec = config.spacing_specs[index] if index < len(config.spacing_specs) else None
        nominal = float(spec[0]) if spec is not None else None
        lower_tolerance = float(spec[1]) if spec is not None else None
        upper_tolerance = float(spec[2]) if spec is not None else None
        lower_limit = nominal - lower_tolerance if spec is not None else None
        upper_limit = nominal + upper_tolerance if spec is not None else None
        configured = spec is not None
        ok = (
            not config.spacing_check_enabled
            or (
                configured
                and lower_limit is not None
                and upper_limit is not None
                and value >= lower_limit
                and value <= upper_limit
            )
        )
        spacing_results.append(
            {
                "index": index + 1,
                "from_pin": index + 1,
                "to_pin": index + 2,
                "distance": float(value),
                "nominal": nominal,
                "lower_tolerance": lower_tolerance,
                "upper_tolerance": upper_tolerance,
                "lower_limit": lower_limit,
                "upper_limit": upper_limit,
                "unit": unit,
                "configured": configured,
                "pred": "OK" if ok else "NG",
            }
        )
    count_ok = len(height_values) == int(config.expected_pin_count)
    expected_spacing_count = max(0, int(config.expected_pin_count) - 1)
    spacing_config_ok = len(config.spacing_specs) >= expected_spacing_count
    spacing_count_ok = len(spacing_values) == expected_spacing_count
    spacing_ok = (
        not config.spacing_check_enabled
        or (
            spacing_config_ok
            and spacing_count_ok
            and all(item["pred"] == "OK" for item in spacing_results)
        )
    )
    height_ok = not config.height_check_enabled or all(height_in_spec)
    pred = "OK" if count_ok and height_ok and spacing_ok else "NG"
    return {
        "pred": pred,
        "unit": unit,
        "height_check_enabled": bool(config.height_check_enabled),
        "spacing_check_enabled": bool(config.spacing_check_enabled),
        "height_values": height_values,
        "height_in_spec": height_in_spec,
        "height_lower_limit": config.lower_limit,
        "height_upper_limit": config.upper_limit,
        "spacing_values": spacing_values,
        "spacing_results": tuple(spacing_results),
        "count_ok": count_ok,
        "spacing_count_ok": spacing_count_ok,
        "spacing_config_ok": spacing_config_ok,
    }


def multi_pin_tip_judgment_payload(
    result: MultiPinTipHeightResult,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation = evaluate_multi_pin_tip_height(result, params)
    unit = str(evaluation["unit"])
    height_values = tuple(float(value) for value in evaluation["height_values"])
    height_in_spec = tuple(bool(value) for value in evaluation["height_in_spec"])
    height_enabled = bool(evaluation["height_check_enabled"])
    spacing_enabled = bool(evaluation["spacing_check_enabled"])
    pin_results = [
        {
            "index": index + 1,
            "point": [float(point[0]), float(point[1])],
            "distance": float(height_values[index]),
            "unit": unit,
            "pred": "OK" if height_in_spec[index] else "NG",
        }
        for index, point in enumerate(result.tip_points)
    ]
    spacing_results: list[dict[str, Any]] = []
    for raw in evaluation["spacing_results"]:
        item = dict(raw)
        index = int(item["index"]) - 1
        if 0 <= index < len(result.tip_points) - 1:
            item["point_a"] = [
                float(result.tip_points[index][0]), float(result.tip_points[index][1])
            ]
            item["point_b"] = [
                float(result.tip_points[index + 1][0]), float(result.tip_points[index + 1][1])
            ]
        spacing_results.append(item)
    minimum = min(height_values) if height_values else 0.0
    maximum = max(height_values) if height_values else 0.0
    spacing_values = tuple(float(value) for value in evaluation["spacing_values"])
    payload = result.to_dict()
    payload.update(
        {
            "distances": list(height_values),
            "unit": unit,
            "lower_limit": evaluation["height_lower_limit"],
            "upper_limit": evaluation["height_upper_limit"],
            "height_check_enabled": height_enabled,
            "spacing_check_enabled": spacing_enabled,
            "pin_results": pin_results,
            "spacing_values": list(spacing_values),
            "spacing_results": spacing_results,
            "spacing_count_ok": bool(evaluation["spacing_count_ok"]),
            "spacing_config_ok": bool(evaluation["spacing_config_ok"]),
            "in_spec_points": [item["point"] for item in pin_results if item["pred"] == "OK"],
            "out_of_spec_points": [item["point"] for item in pin_results if item["pred"] == "NG"],
            "label": (
                f"{minimum:.3f}..{maximum:.3f}{unit}"
                if height_enabled
                else f"spacing {min(spacing_values, default=0.0):.3f}..{max(spacing_values, default=0.0):.3f}{unit}"
            ),
            "pred": str(evaluation["pred"]),
        }
    )
    return payload


def judge_bright_block_y_distance(
    result: PinCenterDistanceResult,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, float, float | None, float | None, str]:
    return judge_pin_center_distance(result, params)


__all__ = [
    "MEASUREMENT_ALGORITHMS",
    "BRIGHT_BLOCK_CENTER_ALGORITHM",
    "BRIGHT_BLOCK_Y_DISTANCE_ALGORITHM",
    "CENTER_DISTANCE_ALGORITHM",
    "CENTER_DISTANCE_ALGORITHMS",
    "FIND_LINE_ALGORITHM",
    "FIND_LINE_ALGORITHMS",
    "FIND_LINE_SUBPIX_ALGORITHM",
    "PIN_CENTER_DISTANCE_ALGORITHM",
    "PIN_TIP_POINT_ALGORITHM",
    "MULTI_PIN_TIP_HEIGHT_ALGORITHM",
    "POINT_LINE_DISTANCE_ALGORITHM",
    "LINE_DISTANCE_ALGORITHM",
    "LINE_DISTANCE_ALGORITHMS",
    "LINE_DISTANCE_REF_NORMAL_ALGORITHM",
    "EdgeDistanceConfig",
    "EdgeDistanceResult",
    "FindLineConfig",
    "FindLineMeasurementConfig",
    "FindLineMeasurementResult",
    "FittedLine",
    "BrightBlockCenterResult",
    "PinCenterCandidate",
    "PinCenterDistanceConfig",
    "PinCenterDistanceResult",
    "PinTipPointConfig",
    "PinTipPointResult",
    "MultiPinTipHeightConfig",
    "MultiPinTipHeightResult",
    "find_edge_points",
    "filter_line_points",
    "fit_line",
    "fit_line_filtered",
    "is_measurement_algorithm",
    "judge_edge_distance",
    "judge_bright_block_y_distance",
    "judge_find_line",
    "judge_pin_center_distance",
    "judge_multi_pin_tip_height",
    "evaluate_multi_pin_tip_height",
    "multi_pin_tip_judgment_payload",
    "measure_bright_block_center",
    "measure_bright_block_center_from_array",
    "measure_bright_block_y_distance",
    "measure_bright_block_y_distance_from_array",
    "measure_edge_distance",
    "measure_edge_distance_from_array",
    "measure_find_line",
    "measure_find_line_from_array",
    "measure_pin_center_distance",
    "measure_pin_center_distance_from_array",
    "measure_pin_tip_point",
    "measure_pin_tip_point_from_array",
    "measure_multi_pin_tip_height",
    "measure_multi_pin_tip_height_from_array",
    "measurement_value",
]
