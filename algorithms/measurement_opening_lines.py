"""Exploratory straight-edge opening measurement, with inspectable quality flags.

This is a separate candidate from ``opening_gap_median``.  It measures the
distance from the lower segment's midpoint to the upper infinite line.  It does
not calibrate millimetres, certify a physical boundary, or use laboratory values.
Every sampled column and every detected local gradient peak remain inspectable.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from algorithms.measurement_opening_gap import OpeningGapRegion


def _select_inner_branch(samples, descending):
    """Find supported line branches, then prefer the one facing the opening.

    This fixed image-space search is deliberately conservative: slopes outside
    +/-0.5, unsupported inner branches, and unresolved branches require review
    or failure. Brightness is not allowed to make an outer backing plate win.
    """
    width = len(samples["points"])
    x = np.asarray([p[0] for p in samples["points"]], dtype=float)
    center = float((x[0] + x[-1]) / 2)
    peaks = samples["candidate_peaks"]
    samples["strongest_points"] = [p[:] for p in samples["points"]]
    samples["strongest_strengths"] = list(samples["strengths"])
    samples["strongest_boundary_mask"] = list(samples["boundary_mask"])
    samples["strongest_ambiguous_mask"] = list(samples["ambiguous_mask"])
    samples["branch_flags"] = []
    samples["branch_selection_failed"] = False
    samples["branches"] = []
    samples["selected_peak_indices"] = [None] * width
    if not any(peaks):
        samples["branch_selection_failed"] = True
        samples["branch_flags"].append("no_supported_inner_branch")
        return samples
    max_peaks = max(map(len, peaks))
    ys = np.full((width, max_peaks), np.inf)
    strengths = np.zeros_like(ys)
    for i, column in enumerate(peaks):
        for j, (y, strength) in enumerate(column):
            ys[i, j], strengths[i, j] = y, strength
    hypotheses = []
    rows = np.arange(width)
    tolerance = 2.5
    for initial_slope in np.linspace(-.5, .5, 41):
        offsets = (ys - initial_slope * (x[:, None] - center))
        finite = offsets[np.isfinite(offsets)]
        bins, counts = np.unique(np.rint(finite / 2).astype(int), return_counts=True)
        for bin_index in np.argsort(counts)[-3:]:
            if counts[bin_index] < max(8, math.ceil(width * .18)):
                continue
            slope, center_y = float(initial_slope), float(bins[bin_index] * 2)
            for _ in range(3):
                delta = np.abs(ys - (slope * (x[:, None] - center) + center_y))
                indices = np.argmin(delta, axis=1)
                supported = delta[rows, indices] <= tolerance
                if supported.sum() < 5:
                    break
                design = np.column_stack((x[supported] - center, np.ones(int(supported.sum()))))
                slope, center_y = map(float, np.linalg.lstsq(
                    design, ys[rows[supported], indices[supported]], rcond=None)[0])
            delta = np.abs(ys - (slope * (x[:, None] - center) + center_y))
            indices = np.argmin(delta, axis=1)
            supported = delta[rows, indices] <= tolerance
            count = int(supported.sum())
            if count < max(8, math.ceil(width * .2)) or abs(slope) > .5:
                continue
            coverage = float(np.ptp(x[supported]) / (width - 1))
            if coverage < .2:
                continue
            rmse = float(np.sqrt(np.mean(delta[rows[supported], indices[supported]] ** 2)))
            hypotheses.append({"slope": slope, "center_y": center_y, "support": count,
                               "coverage": coverage, "rmse_px": rmse,
                               "indices": indices, "supported": supported})
    # Deduplicate fits to the same spatial branch; one column has one vote.
    branches = []
    for hypothesis in sorted(hypotheses, key=lambda h: (-h["support"], h["rmse_px"])):
        if any(abs(hypothesis["center_y"] - b["center_y"]) < 5 for b in branches):
            continue
        branches.append(hypothesis)
    for branch in branches:
        branch["eligible"] = branch["support"] >= math.ceil(width * .6) and branch["coverage"] >= .8
    eligible = [b for b in branches if b["eligible"]]
    samples["branches"] = [{k: v for k, v in b.items() if k not in {"indices", "supported"}}
                           for b in branches]
    if not eligible:
        samples["branch_selection_failed"] = True
        samples["branch_flags"].append("no_supported_inner_branch")
        return samples
    chosen = (max if descending else min)(eligible, key=lambda b: b["center_y"])
    samples["selected_branch_center_y"] = chosen["center_y"]
    samples["selected_branch_slope"] = chosen["slope"]
    inner_unsupported = [b for b in branches if not b["eligible"] and
                         ((b["center_y"] > chosen["center_y"] + 5) if descending
                          else (b["center_y"] < chosen["center_y"] - 5))]
    if inner_unsupported:
        samples["branch_selection_failed"] = True
        samples["branch_flags"].append("inner_branch_insufficient_support")
    if len(branches) > 1:
        samples["branch_flags"].append("multiple_spatial_edge_branches")
    # Keep points exactly on measured peaks, never substitute the fitted line.
    # Columns without a candidate on this branch remain recorded but invalid.
    selected_mask = chosen["supported"]
    nonselected_inner = []
    for i in range(width):
        index = int(chosen["indices"][i])
        predicted = chosen["slope"] * (x[i] - center) + chosen["center_y"]
        nonselected_inner.append(any((y > predicted + tolerance if descending else y < predicted - tolerance)
                                     for y, _ in peaks[i]))
        if selected_mask[i]:
            samples["points"][i] = [int(x[i]), float(ys[i, index])]
            samples["strengths"][i] = float(strengths[i, index])
            samples["selected_peak_indices"][i] = index
        samples["valid_mask"][i] = bool(selected_mask[i])
    samples["nonselected_inner_mask"] = nonselected_inner
    run = longest = 0
    for marked in nonselected_inner:
        run = run + 1 if marked else 0
        longest = max(longest, run)
    samples["nonselected_inner_max_run"] = longest
    if longest >= 5:
        samples["branch_flags"].append("unresolved_inner_structure")
    return samples


def _candidate_points(gray, x0, x1, band, descending, blur_ksize, min_strength):
    lo, hi = band
    margin = blur_ksize + 2
    y0, y1 = max(0, lo - margin), min(gray.shape[0], hi + margin + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    if blur_ksize > 1:
        patch = cv2.GaussianBlur(patch, (1, blur_ksize), 0)
    response = np.diff(patch, axis=0) * (-1 if descending else 1)
    points, strengths, valid, peaks, ambiguous, boundary = [], [], [], [], [], []
    for column in range(x1 - x0):
        values = response[lo - y0:hi - y0, column]
        best = int(np.argmax(values))
        strength = float(values[best])
        offset = 0.0
        if 0 < best < len(values) - 1:
            left, right = float(values[best - 1]), float(values[best + 1])
            denominator = left - 2 * strength + right
            if denominator < -1e-6:
                offset = float(np.clip(.5 * (left - right) / denominator, -.5, .5))
        y = float(lo + best + .5 + offset)
        local = [i for i, value in enumerate(values)
                 if value >= min_strength
                 and (i == 0 or value >= values[i - 1])
                 and (i == len(values) - 1 or value > values[i + 1])]
        competing = [i for i in local if abs(i - best) >= max(4, blur_ksize // 2)
                     and float(values[i]) >= .55 * strength]
        points.append([int(x0 + column), y])
        strengths.append(strength)
        valid.append(bool(strength >= min_strength))
        column_peaks = []
        for i in local:
            correction = 0.0
            if 0 < i < len(values) - 1:
                left, middle, right = map(float, values[i - 1:i + 2])
                denominator = left - 2 * middle + right
                if denominator < -1e-6:
                    correction = float(np.clip(.5 * (left - right) / denominator, -.5, .5))
            column_peaks.append([float(lo + i + .5 + correction), float(values[i])])
        peaks.append(column_peaks)
        ambiguous.append(bool(competing))
        boundary.append(bool(min(best, len(values) - 1 - best) < max(2, blur_ksize // 2)))
    samples = {"points": points, "strengths": strengths, "valid_mask": valid,
               "candidate_peaks": peaks, "ambiguous_mask": ambiguous,
               "boundary_mask": boundary}
    samples = _select_inner_branch(samples, descending)
    samples["boundary_mask"] = [bool(min(y - lo, hi - y) < max(2, blur_ksize // 2))
                                for _, y in samples["points"]]
    samples["ambiguous_mask"] = [
        any(abs(peak_y - y) >= max(4, blur_ksize // 2)
            and strength >= .55 * samples["strengths"][i]
            for peak_y, strength in samples["candidate_peaks"][i])
        for i, (_, y) in enumerate(samples["points"])
    ]
    return samples


def _fit_edge(samples, span, min_strength):
    """Huber fit plus diagnostics on *all* valid candidates, including outliers."""
    points = np.asarray(samples["points"], dtype=float)
    valid = np.asarray(samples["valid_mask"], dtype=bool)
    width, count = len(points), int(valid.sum())
    result = {**samples, "segment": None, "coefficients": None,
              "inlier_mask": [False] * width, "rmse_px": None,
              "all_points_rmse_px": None, "inlier_ratio": 0.0,
              "valid_ratio": count / width, "valid_columns": count,
              "quality_flags": list(samples.get("branch_flags", []))}
    if count < max(5, math.ceil(width * .6)):
        result["quality_flags"].append("insufficient_edge_support")
        return result
    selected = points[valid]
    center = float(np.mean(selected[:, 0]))
    design = np.column_stack((selected[:, 0] - center, np.ones(count)))
    coefficients = np.linalg.lstsq(design, selected[:, 1], rcond=None)[0]
    for _ in range(20):
        residual = selected[:, 1] - design @ coefficients
        scale = max(.35, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-9))
        root_weights = np.sqrt(weights)
        updated = np.linalg.lstsq(design * root_weights[:, None],
                                  selected[:, 1] * root_weights, rcond=None)[0]
        if np.max(np.abs(updated - coefficients)) < 1e-7:
            coefficients = updated
            break
        coefficients = updated
    slope, centered_intercept = map(float, coefficients)
    intercept = centered_intercept - slope * center
    residual = selected[:, 1] - (slope * selected[:, 0] + intercept)
    normal_residual = residual / math.hypot(slope, 1.0)
    # A fixed pixel tolerance prevents a real step from enlarging its own
    # acceptance band. Excluded points remain in all-point diagnostics below.
    inliers = np.abs(normal_residual) <= 1.5
    full_inliers = np.zeros(width, dtype=bool)
    full_inliers[np.flatnonzero(valid)] = inliers
    bins = np.array_split(np.arange(count), 7)
    bin_bias = max(abs(float(np.median(normal_residual[b]))) for b in bins if len(b))
    adjacent = np.diff(selected[:, 0]) == 1
    jumps = np.abs(np.diff(normal_residual))[adjacent]
    max_jump = float(jumps.max()) if jumps.size else 0.0
    inlier_count = int(inliers.sum())
    flags = result["quality_flags"]
    median_strength = float(np.median(np.asarray(samples["strengths"])[valid]))
    ambiguous_ratio = float(np.mean(np.asarray(samples["ambiguous_mask"])[valid]))
    boundary_ratio = float(np.mean(np.asarray(samples["boundary_mask"])[valid]))
    if count / width < .9:
        flags.append("partial_edge_support")
    if median_strength < 2 * min_strength:
        flags.append("low_edge_contrast")
    if ambiguous_ratio > .1:
        flags.append("multiple_competing_edges")
    if boundary_ratio > .05:
        flags.append("edge_near_search_boundary")
    if inlier_count / count < .9:
        flags.append("many_points_off_line")
    all_rmse = float(np.sqrt(np.mean(normal_residual ** 2)))
    if all_rmse > 1.5 or bin_bias > 1.5 or max_jump > 3.0:
        flags.append("nonstraight_or_discontinuous_edge")
    supported_x = selected[:, 0]
    coverage = float((supported_x.max() - supported_x.min()) / max(1, span[1] - span[0]))
    max_missing_run = int(max(np.diff(supported_x).max() - 1, 0)) if count > 1 else width
    if coverage < .9 or max_missing_run > max(5, width // 10):
        flags.append("uneven_horizontal_support")
    result.update({
        "segment": [[int(x), float(slope * x + intercept)] for x in span],
        "coefficients": [slope, intercept], "inlier_mask": full_inliers.tolist(),
        "rmse_px": float(np.sqrt(np.mean(normal_residual[inliers] ** 2))) if inlier_count else None,
        "all_points_rmse_px": all_rmse, "inlier_ratio": inlier_count / count,
        "median_strength": median_strength, "ambiguous_ratio": ambiguous_ratio,
        "boundary_ratio": boundary_ratio, "max_bin_residual_px": bin_bias,
        "max_adjacent_jump_px": max_jump, "horizontal_coverage": coverage,
        "max_missing_run": max_missing_run,
    })
    return result


def _project(point, coefficients):
    x, y = point
    slope, intercept = coefficients
    signed = y - slope * x - intercept
    fraction = signed / (1 + slope * slope)
    foot = [float(x + slope * fraction), float(y - fraction)]
    return abs(float(signed)) / math.hypot(slope, 1.0), foot


def measure_opening_lines(image_bgr, region: OpeningGapRegion, *, fit_width=121,
                          blur_ksize=9, min_strength=5.0) -> dict:
    """Return serializable line measurements; ``ok`` means diagnostic checks pass.

    Pixel coordinates refer to the unresampled input image. ``points`` contains
    one selected branch candidate per sampled column (or the diagnostic strongest
    candidate when no supported branch is available), even for weak columns;
    consult ``valid_mask`` and ``inlier_mask`` before using them. ``candidate_peaks``
    retains every above-threshold local peak as [y, strength] per column.
    ``rmse_px`` is the inlier perpendicular residual; ``all_points_rmse_px`` also
    includes rejected valid candidates. Flags are exploratory pixel thresholds,
    not dimensional tolerances. Neither distance is a minimum throat clearance.
    """
    image = np.asarray(image_bgr)
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("image must be grayscale or BGR")
    if not np.isfinite(gray).all():
        raise ValueError("image must contain finite values")
    if int(fit_width) != fit_width or fit_width < 5 or fit_width % 2 != 1:
        raise ValueError("fit_width must be an odd integer >= 5")
    if int(blur_ksize) != blur_ksize or blur_ksize < 1 or blur_ksize % 2 != 1:
        raise ValueError("blur_ksize must be a positive odd integer")
    if not math.isfinite(min_strength) or min_strength <= 0:
        raise ValueError("min_strength must be finite and positive")
    fit_width, blur_ksize = int(fit_width), int(blur_ksize)
    height, width = gray.shape
    x0 = int(region.center_x) - fit_width // 2
    x1 = x0 + fit_width
    ulo, uhi = map(int, region.upper_y)
    llo, lhi = map(int, region.lower_y)
    if not (0 <= x0 < x1 <= width and 0 <= ulo < uhi < llo < lhi <= height - 1):
        raise ValueError("line sampling region is outside image or search bands overlap")
    span = [x0, x1 - 1]
    upper = _fit_edge(_candidate_points(gray, x0, x1, (ulo, uhi), True,
                                        blur_ksize, min_strength), span, min_strength)
    lower = _fit_edge(_candidate_points(gray, x0, x1, (llo, lhi), False,
                                        blur_ksize, min_strength), span, min_strength)
    reasons = [f"{side}:{flag}" for side, edge in (("upper", upper), ("lower", lower))
               for flag in edge["quality_flags"]]
    result = {"name": region.name, "type": "opening_lines_candidate", "status": "failed",
              "reasons": reasons, "upper": upper, "lower": lower,
              "fit_width": fit_width, "span": span, "distance_px": None,
              "diagnostic_distance_px": None,
              "reverse_distance_px": None, "dimension_segment": None,
              "reverse_dimension_segment": None, "angle_difference_deg": None,
              "distance_definition": "lower_segment_midpoint_to_upper_infinite_line"}
    if upper["coefficients"] is None or lower["coefficients"] is None:
        return result
    au, bu = upper["coefficients"]
    al, bl = lower["coefficients"]
    if any(al * x + bl <= au * x + bu for x in span):
        reasons.append("lines_cross_or_gap_nonpositive_within_span")
        return result
    midpoint_x = (span[0] + span[1]) / 2
    upper_mid = [midpoint_x, au * midpoint_x + bu]
    lower_mid = [midpoint_x, al * midpoint_x + bl]
    distance, upper_foot = _project(lower_mid, upper["coefficients"])
    reverse_distance, lower_foot = _project(upper_mid, lower["coefficients"])
    angle = abs(math.degrees(math.atan(al) - math.atan(au)))
    if angle > .5:
        reasons.append("upper_lower_lines_not_parallel")
    if any(not span[0] <= p[0] <= span[1] for p in (upper_foot, lower_foot)):
        reasons.append("perpendicular_foot_outside_sampled_span")
    result.update({"distance_px": distance, "reverse_distance_px": reverse_distance,
                   "dimension_segment": [upper_foot, lower_mid],
                   "reverse_dimension_segment": [upper_mid, lower_foot],
                   "angle_difference_deg": angle,
                   "status": "review" if reasons else "ok"})
    if upper.get("branch_selection_failed") or lower.get("branch_selection_failed"):
        result.update({"status": "failed", "diagnostic_distance_px": distance,
                       "diagnostic_reverse_distance_px": reverse_distance,
                       "diagnostic_dimension_segment": result["dimension_segment"],
                       "distance_px": None, "reverse_distance_px": None,
                       "dimension_segment": None, "reverse_dimension_segment": None})
    return result
