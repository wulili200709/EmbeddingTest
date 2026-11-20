"""Measure a new image batch with the unchanged confirmed body-edge algorithm.

Raw images are read-only. Padded crops preserve source pixels and filter support;
diagnostics retain their crop coordinate origin. No reference dimensions are
used for finding edges. Transferred millimetre parameters are explicitly named.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from algorithms.measurement_opening_body_edge import measure_opening_body_edges
from algorithms.measurement_opening_gap import regions_from_params
from compare_lab_opening_lines import draw_measurements, physical_identity
from test_lab_openings import locate_rows, read_image

ORDER = [f'{r}{i}' for r in 'UL' for i in range(1, 8)]
MAPPING = {'top': [f'L{i}' for i in range(7, 0, -1)], 'bottom': [f'U{i}' for i in range(7, 0, -1)]}
LOCATOR = {'search_margin': [180, 120], 'min_score': .75,
           'rows': {'U': [1100, 830, 3850, 1140], 'L': [1180, 2470, 3880, 2800]}}
STATUS = {'ok': '有读数', 'review': '有读数，待复核', 'failed': '无有效值'}
REASONS = {
    'no_coherent_lip_seed': '边缘粗定位不足',
    'no_two_sided_coherent_body_transition': '未找到左右一致的边缘',
    'spatial_recovery_conflicts_with_observed_side_contour': '连续边恢复与实测侧边冲突',
    'spatial_recovery_behind_unresolved_outer_contour': '外侧边仍未确定，拒绝采用内侧边',
    'insufficient_two_sided_contour_anchors': '左右有效边缘点不足',
    'body_contour_behind_unsupported_inner_branch': '存在未解决的外侧边分支',
    'middle_contour_not_observable': '中间轮廓不清晰',
    'insufficient_two_sided_inlier_support': '左右符合拟合线的点不足',
    'partial_side_anchor_support': '侧边有效点覆盖不完整',
    'side_anchors_do_not_define_single_body_line': '左右侧边不能形成一致直线',
    'side_contour_not_straight_or_ambiguous': '侧边不直或存在歧义',
    'middle_contour_departs_from_body_line': '中间轮廓偏离拟合线',
    'side_contour_outliers': '侧边存在偏离点',
    'full_contour_variation': '整体轮廓起伏较大',
    'lines_cross_or_gap_nonpositive_within_span': '拟合线交叉或开口非正',
    'upper_lower_lines_not_parallel': '上下拟合线夹角超过0.5度',
    'perpendicular_foot_outside_sampled_span': '垂足超出采样区间',
    'localization_failed': '产品排定位失败',
    'measurement_exception': '测量运行异常',
}


def translate_reason(reason):
    parts = reason.split(':', 1)
    return (('上边：' if parts[0] == 'upper' else '下边：') + REASONS.get(parts[1], parts[1])
            if len(parts) == 2 and parts[0] in ('upper', 'lower') else REASONS.get(reason, reason))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def global_result(raw, origin):
    ox, oy = origin
    r = {k: raw.get(k) for k in ('name', 'type', 'status', 'reasons', 'distance_px', 'reverse_distance_px',
          'diagnostic_distance_px', 'angle_difference_deg', 'distance_definition', 'contour_definition',
          'fit_width', 'anchor_abs_dx_range_px')}
    r['span'] = [x + ox for x in raw.get('span', [])]
    for key in ('dimension_segment', 'reverse_dimension_segment', 'diagnostic_dimension_segment'):
        r[key] = [[x + ox, y + oy] for x, y in raw[key]] if raw.get(key) else None
    for side in ('upper', 'lower'):
        e = raw.get(side, {})
        r[side] = {k: e.get(k) for k in ('valid_mask', 'inlier_mask', 'anchor_mask', 'anchor_zone_mask',
            'valid_ratio', 'valid_columns', 'anchor_rmse_px', 'rmse_px', 'all_points_rmse_px',
            'left_anchor_support', 'right_anchor_support', 'middle_max_deviation_px', 'middle_deviation_max_run',
            'selection_mode', 'measurement_failed', 'quality_flags', 'anchor_abs_dx_range_px')}
        for key in ('points', 'segment'):
            r[side][key] = [[x + ox, y + oy] for x, y in e.get(key, [])] if e.get(key) is not None else None
        coef = e.get('coefficients')
        r[side]['coefficients'] = [coef[0], coef[1] + oy - coef[0] * ox] if coef else None
        r[side]['anchor_x_ranges'] = [[a + ox, b + ox] if a is not None else None for a, b in e.get('anchor_x_ranges', [])] if e.get('anchor_x_ranges') else []
    return r


def measure_padded(gray, region, config):
    # Preserve original y coordinates: the existing coarse branch search bins
    # absolute intercepts on a 2-pixel grid, so odd y shifts change bin phase.
    # Horizontal padding exceeds filter support.
    ox = max(0, region.center_x - 50 - 32)
    right = min(gray.shape[1], region.center_x + 50 + 33)
    oy = 0
    bottom = gray.shape[0]
    crop = gray[oy:bottom, ox:right]
    local = replace(region, center_x=region.center_x - ox,
                    upper_y=tuple(y - oy for y in region.upper_y), lower_y=tuple(y - oy for y in region.lower_y))
    raw = measure_opening_body_edges(crop, local, fit_width=101,
                                    blur_ksize=config['blur_ksize'], min_strength=config['min_strength'])
    return global_result(raw, (ox, oy)), {'coordinate_origin_xy': [ox, oy], 'crop_size': list(crop.shape[::-1]), 'measurement': raw}


def panels(gray, points, target):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    annotated = draw_measurements(bgr, points, labels=False)
    panel = Image.new('RGB', (7 * 290, 2 * 330), '#17242c')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    draw = ImageDraw.Draw(panel)
    for i, p in enumerate(points):
        region, auto = p['region'], p['auto']
        x = region['center_x']; y0 = region['upper_y'][0] - 15; y1 = region['lower_y'][1] + 20
        crop = Image.fromarray(cv2.cvtColor(annotated[y0:y1, x-135:x+135], cv2.COLOR_BGR2RGB))
        px, py = (i % 7) * 290 + 8, (i // 7) * 330 + 5
        panel.paste(crop, (px, py + 65))
        value = auto['distance_px']; mm = auto['distance_mm']
        draw.text((px, py), f"{p['name']}: {value:.3f} px" if value is not None else f"{p['name']}: NO RESULT", fill='#9cebc3', font=font)
        draw.text((px, py+23), f"{mm:.4f} mm (transferred)" if mm is not None else 'No numeric measurement', fill='#dde8ef', font=small)
        draw.text((px, py+44), auto['status'].upper(), fill='#ffce78', font=small)
    panel.save(target)
    overview = Image.fromarray(cv2.cvtColor(draw_measurements(bgr, points), cv2.COLOR_BGR2RGB))
    overview.thumbnail((2400, 1600))
    overview.save(target.parent.parent / 'overviews' / f'{target.stem}.jpg', quality=95)


def report(out, data):
    meta = data['meta']; cards = []
    summary_rows = []
    for name in ORDER:
        group = [p['auto'] for im in data['images'] for p in im['points'] if p['name'] == name]
        vals = [r['distance_px'] for r in group if r['distance_px'] is not None]
        mm = [r['distance_mm'] for r in group if r['distance_mm'] is not None]
        fmt = lambda x, prec: '—' if x is None else f'{x:.{prec}f}'
        summary_rows.append(f'<tr><th>{name}</th><td>{len(vals)}/50</td><td>{fmt(min(vals) if vals else None,3)}</td><td>{fmt(max(vals) if vals else None,3)}</td><td>{fmt(max(vals)-min(vals) if vals else None,3)}</td><td>{fmt(min(mm) if mm else None,4)}</td><td>{fmt(max(mm) if mm else None,4)}</td></tr>')
    wide = []
    for image in data['images']:
        values = {p['name']: p['auto'] for p in image['points']}
        cells = ''.join(f'<td class="{values[n]["status"]}">{values[n]["distance_mm"]:.4f}</td>' if values[n]['distance_mm'] is not None else '<td class="failed">—</td>' for n in ORDER)
        wide.append(f'<tr><th><a href="#image-{image["index"]}">{image["index"]:02d}</a></th><td>{html.escape(image["name"])}</td>{cells}</tr>')
        detail = []
        for n in ORDER:
            a = values[n]
            detail.append('<tr>'+''.join([f'<th>{n}</th>',f'<td>{fmt(a["distance_px"],3)}</td>',f'<td>{fmt(a["distance_mm"],4)}</td>',f'<td>{STATUS[a["status"]]}</td>',f'<td class="reason">{html.escape("；".join(a["reasons_cn"]))}</td>'])+'</tr>')
        cards.append(f'''<section id="image-{image['index']}"><h2>{image['index']:02d} · {html.escape(image['name'])}</h2>
<p><a href="{html.escape(image['source_uri'])}">原始图片</a> · <a href="overviews/{image['index']:02d}.jpg">整图标注预览</a> · 上排从左到右 L7→L1，下排 U7→U1</p>
<a href="details/{image['index']:02d}.png"><img loading="lazy" src="details/{image['index']:02d}.png" alt="14个开口的测量线"></a>
<details><summary>展开该图的14个数值和复核原因</summary><table><thead><tr><th>点位</th><th>像素 px</th><th>参考 mm</th><th>状态</th><th>原因</th></tr></thead><tbody>{''.join(detail)}</tbody></table></details></section>''')
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>50张开口检测结果</title>
<style>body{{font:15px/1.6 "Microsoft YaHei",sans-serif;background:#f3f7f9;color:#173743;margin:24px}}h1{{font-size:26px}}h2{{font-size:20px}}section{{background:white;padding:18px;margin:24px 0;border-radius:8px}}.note{{padding:14px;background:#fff1d7}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px 10px;border-bottom:1px solid #dce5e8;text-align:right;white-space:nowrap}}thead th{{background:#183f50;color:white}}th:first-child,td.reason{{text-align:left}}td.reason{{white-space:normal;min-width:280px}}img{{max-width:100%;height:auto}}a{{color:#007c83}}.failed{{background:#ffe1d9;color:#973d25}}.review{{background:#fff8e8}}details{{margin-top:12px}}.badge{{font-weight:bold;color:#007c83}}</style>
<h1>50张开口检测结果</h1><p>最新主体边缘算法 · 原图101列采样 · 2026-10-30</p>
<p class="badge">50张 × 14个位置＝700处；有数值 {meta['numeric_count']} 处，待复核 {meta['status_counts'].get('review',0)} 处，无有效值 {meta['status_counts'].get('failed',0)} 处。</p>
<p class="note">毫米结果按用户指定，沿用 Lab Test 前5张计算、用于第6张验证的参数：k={meta['calibration']['scale_mm_per_px']:.11f} mm/px，另加各位置补偿。此批成像倍率未核实，毫米值供参考。原始像素值一并保留。状态表示算法取边质量，不代表产品尺寸合格。</p>
<p>青色短线为边缘拟合线，绿色线为测量垂线，红色虚线为失败点的诊断线。空白值不填零。编号继续采用用户确认的上排L7→L1、下排U7→U1。旧0916配置中的编号另保存在明细中。</p>
<h2>逐图参考毫米值</h2><div class="scroll"><table><thead><tr><th>序号</th><th>图片</th>{''.join(f'<th>{n}</th>' for n in ORDER)}</tr></thead><tbody>{''.join(wide)}</tbody></table></div>
<h2>各位置数值范围</h2><table><thead><tr><th>位置</th><th>有数值</th><th>最小px</th><th>最大px</th><th>极差px</th><th>最小参考mm</th><th>最大参考mm</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
{''.join(cards)}</html>'''
    (out / '检测标注报告.html').write_text(document, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--verify-full-first', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    source, out = args.input_dir.resolve(), args.out.resolve()
    if source == out or source in out.parents:
        raise ValueError('Keep outputs separate from source image directory')
    for folder in ('details', 'overviews', 'diagnostics', 'qa'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'config/examples/0916_opening_gaps.json').read_text(encoding='utf-8'))
    previous = json.loads((ROOT / 'reports/lab_holdout6_20261030/holdout6_results.json').read_text(encoding='utf-8'))
    calibration = {**previous['methods']['new']['parameters'], 'training_images': previous['training_images'],
                   'training_pairs': previous['training_pairs'], 'source': 'Lab Test 前5张拟合，用于第6张测试',
                   'formula': 'distance_mm = distance_px * scale_mm_per_px + offsets_add_mm[point]',
                   'status': 'transferred_parameters_magnification_unverified'}
    paths = sorted(source.glob('*.png'), key=lambda p: p.name)
    if len(paths) != 50:
        raise ValueError(f'Expected 50 source PNG files; got {len(paths)}')
    if args.limit: paths = paths[:args.limit]
    reference = read_image(paths[0]); reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    base_regions = regions_from_params(config)
    algorithm = ROOT / 'algorithms/measurement_opening_body_edge.py'
    algorithm_hash = digest(algorithm)
    if algorithm_hash != 'ed040bfac071b01eaf6e6aa22825e7d139c3c8ea3434dc6605f49d9e4db6a7e0':
        raise ValueError('Algorithm differs from user-confirmed version')
    data = {'schema_version': 1, 'meta': {'created_at': '2026-10-30', 'input_dir': str(source),
            'reference_image': paths[0].name, 'image_size': config['image_size'], 'method': 'body_outline',
            'algorithm_sha256': algorithm_hash, 'fit_width': 101, 'mapping': MAPPING, 'report_order': ORDER,
            'calibration': calibration, 'localization': LOCATOR,
            'source_config': '0916_opening_gaps.json', 'geometry_policy': 'Preserve the established per-opening centers and translate each row.',
            'source_pixels_resampled': False, 'measurement_uses_padded_original_pixel_crops': True,
            'distance_rule': '下线段中点到上直线的垂距', 'numeric_count': 0, 'status_counts': {}}, 'images': []}
    full_checks = []
    for index, path in enumerate(paths, 1):
        started = time.perf_counter(); image_hash = digest(path)
        gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None or list(gray.shape[::-1]) != config['image_size']:
            raise ValueError(f'Unexpected image dimensions: {path.name}')
        error = None
        try:
            shifts = locate_rows(gray, reference, LOCATOR)
        except RuntimeError as exc:
            shifts = {}; error = str(exc)
        entry = {'index': index, 'name': path.name, 'source_uri': path.as_uri(), 'sha256': image_hash,
                 'width': gray.shape[1], 'height': gray.shape[0], 'localization': shifts, 'points': [],
                 'detail_image': f'details/{index:02d}.png'}
        diagnostics = {'image': path.name, 'points': [], 'algorithm_sha256': algorithm_hash}
        for base in base_regions:
            name, row, col = physical_identity(base.name)
            shift = shifts.get(base.name[0], {'dx': 0, 'dy': 0})
            region = replace(base, name=name, center_x=base.center_x+shift['dx'],
                             upper_y=tuple(y+shift['dy'] for y in base.upper_y), lower_y=tuple(y+shift['dy'] for y in base.lower_y))
            if error:
                auto = {'name': name, 'status': 'failed', 'distance_px': None, 'reverse_distance_px': None,
                        'angle_difference_deg': None, 'reasons': ['localization_failed'], 'upper': {}, 'lower': {}, 'error': error}
            else:
                auto, raw = measure_padded(gray, region, config)
                diagnostics['points'].append({'name': name, 'source_region': asdict(region), **raw})
                if args.verify_full_first and index == 1:
                    full = measure_opening_body_edges(gray, region, fit_width=101, blur_ksize=config['blur_ksize'], min_strength=config['min_strength'])
                    assert full['status'] == auto['status'] and full['reasons'] == auto['reasons'], name
                    a, b = full['distance_px'], auto['distance_px']
                    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b) < 1e-8), (name, a, b)
                    for side in ('upper', 'lower'):
                        if full[side].get('coefficients') is not None:
                            assert np.max(np.abs(np.array(full[side]['coefficients'])-auto[side]['coefficients'])) < 1e-8
                    full_checks.append({'name': name, 'status': auto['status'], 'full_px': a, 'crop_px': b,
                                        'absolute_difference_px': abs(a-b) if a is not None else None})
            px = auto['distance_px']
            auto['distance_mm'] = px*calibration['scale_mm_per_px']+calibration['offsets_add_mm'][name] if px is not None else None
            auto['status_cn'] = STATUS[auto['status']]
            auto['reasons_cn'] = [translate_reason(r) for r in auto['reasons']]
            entry['points'].append({'name': name, 'physical_row': row, 'physical_col': col,
                                    'source_roi_name': base.name, 'region': asdict(region),
                                    'legacy': {'distance_px': None}, 'auto': auto})
        with gzip.open(out / 'diagnostics' / f'{index:02d}.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(diagnostics, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        panels(gray, entry['points'], out / 'details' / f'{index:02d}.png')
        data['images'].append(entry)
        statuses = Counter(p['auto']['status'] for p in entry['points'])
        print(f'{index:02d}/{len(paths)} {path.name} {dict(statuses)} {time.perf_counter()-started:.1f}s', flush=True)
        dump(out / 'progress.json', {'processed': index, 'total': len(paths), 'last_image': path.name})
    for entry in data['images']:
        assert digest(source / entry['name']) == entry['sha256'], 'Source changed during processing'
    assert digest(algorithm) == algorithm_hash, 'Algorithm changed during processing'
    counts = Counter(p['auto']['status'] for im in data['images'] for p in im['points'])
    data['meta']['status_counts'] = dict(counts)
    data['meta']['numeric_count'] = sum(p['auto']['distance_px'] is not None for im in data['images'] for p in im['points'])
    data['meta']['image_count'] = len(paths)
    data['meta']['measurement_count'] = len(paths)*14
    data['meta']['source_hashes_verified'] = True
    data['meta']['full_vs_crop_first_image_verified'] = bool(full_checks)
    dump(out / 'qa/full_vs_crop.json', full_checks)
    dump(out / 'measurements.json', data)
    dump(out / 'source_config.json', config)
    report(out, data)
    print(json.dumps({'output': str(out), 'images': len(paths), 'numeric': data['meta']['numeric_count'], 'states': dict(counts),
                      'full_vs_crop_checks': len(full_checks)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
