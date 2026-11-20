"""Cross-check delivered labels, source identity, geometry and CSV missing data."""
from collections import Counter
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path

folder = Path(__file__).resolve().parents[1]
root = folder.parents[1]
data = json.loads((folder / 'measurements.json').read_text(encoding='utf-8'))
previous = json.loads(Path(data['meta']['previous_report']).read_text(encoding='utf-8'))
previous_points = {(im['name'], p['name']): p for im in previous['images'] for p in im['points']}
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
names = [f'{row}{i}' for row in 'UL' for i in range(1, 8)]
rows = list(csv.DictReader((folder / 'measurement_comparison.csv').open(encoding='utf-8-sig', newline='')))
assert len(rows) == 84 and len(data['images']) == 6
assert data['meta']['method'] == 'body_outline'
width = data['meta']['fit_width']
assert width == 101 and data['meta']['previous_fit_width'] == 101
assert digest(data['meta']['workbook']) == data['meta']['workbook_sha256']
assert digest(root / 'algorithms/measurement_opening_body_edge.py') == data['meta']['algorithm_sha256']
assert digest(root / 'tools/compare_lab_opening_lines.py') == data['meta']['runner_sha256']
assert data['meta']['manual_reviewed_initially'] == 0
assert [(r['image'], r['point']) for r in rows] == [(f'{i}.png', n) for i in range(1, 7) for n in names]
by_key = {(r['image'], r['point']): r for r in rows}
counts = Counter()
failed = []
recovered = []
for image in data['images']:
    assert digest(Path(data['meta']['input_dir']) / image['name']) == image['sha256']
    assert digest(folder / image['url']) == image['sha256']
    assert [p['name'] for p in image['points']] == [f'{row}{i}' for row in 'LU' for i in range(7, 0, -1)]
    for index, point in enumerate(image['points']):
        assert point['physical_row'] == ('top' if index < 7 else 'bottom')
        assert point['physical_col'] == index % 7 + 1
        auto, row = point['auto'], by_key[(image['name'], point['name'])]
        counts[auto['status']] += 1
        for side, edge in ((side, auto[side]) for side in ('upper', 'lower')):
            old_edge = previous_points[(image['name'], point['name'])]['auto'][side]
            recovery = edge['selection_mode'] == 'coherent_branch_recovery'
            if recovery:
                assert edge['spatial_recovery_reasons'] and edge['selected_spatial_body_branch']
                recovered.append([image['name'], point['name'], side])
            else:
                assert edge['selection_mode'] == 'local_transition'
                assert edge['points'] == old_edge['points'] and edge['valid_mask'] == old_edge['valid_mask']
            assert len(edge['points']) == len(edge['anchor_mask']) == len(edge['valid_mask']) == width
            low, high = edge['anchor_abs_dx_range_px']
            assert abs(low - 40*(width-1)/140) < 1e-9
            assert abs(high - 65*(width-1)/140) < 1e-9
            assert edge['span'] == [point['region']['center_x']-50, point['region']['center_x']+50]
            if edge['segment']:
                assert edge['segment'][1][0]-edge['segment'][0][0] == width-1
                assert edge['anchor_rmse_rejection_threshold_px'] == 2*edge['inlier_tolerance_px']
                if edge['anchor_rmse_px'] > edge['anchor_rmse_rejection_threshold_px']:
                    assert edge['measurement_failed']
                    assert 'side_anchors_do_not_define_single_body_line' in edge['quality_flags']
                    assert auto['status'] == 'failed' and auto['distance_px'] is None
            for xy, anchor, valid in zip(edge['points'], edge['anchor_mask'], edge['valid_mask']):
                assert all(math.isfinite(v) for v in xy)
                if anchor:
                    assert valid and low <= abs(xy[0] - point['region']['center_x']) <= high
            assert len(edge['onset_points']) == len(edge['background_diagnostics']) == width
            for index, (xy, valid, evidence) in enumerate(zip(edge['points'], edge['valid_mask'], edge['background_diagnostics'])):
                if valid:
                    if recovery:
                        event = evidence['selected_spatial_candidate']
                        assert abs(xy[1]-event['y']) < 1e-9
                        assert event in edge['local_transition_candidates'][index]
                        assert event['polarity'] == edge['selected_spatial_body_branch']['polarity']
                    else:
                        event = evidence['transition_localization']
                        assert event['status'] == 'localized'
                        assert abs(xy[1]-event['peak_y']) < 1e-9
                        assert evidence['onset_y'] is not None
        if auto['status'] == 'failed':
            assert auto['distance_px'] is None and auto['reverse_distance_px'] is None
            assert row['auto_px'] == '' and row['auto_provisional_mm'] == ''
            failed.append([image['name'], point['name']])
        else:
            a, b = auto['upper']['coefficients']
            c, d = auto['lower']['coefficients']
            x = point['region']['center_x']
            distance = abs((c-a)*x+d-b) / math.hypot(a, 1)
            assert abs(distance-auto['distance_px']) < 1e-9
            assert abs(float(row['auto_px'])-distance) < 1e-9

class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value and not value.startswith(('#', 'http', 'data:', 'blob:')):
                assert (folder / value).exists(), value

Links().feed((folder / 'report.html').read_text(encoding='utf-8'))
target = next(p for im in data['images'] if im['name']=='2.png' for p in im['points'] if p['name']=='U2')
target_y = {key: target[key]['upper']['coefficients'][0]*target['region']['center_x']+target[key]['upper']['coefficients'][1]
            for key in ('previous_auto','auto')}
target_u3 = next(p for im in data['images'] if im['name']=='2.png' for p in im['points'] if p['name']=='U3')
lower = target_u3['auto']['lower']
assert lower['selection_mode'] == 'coherent_branch_recovery' and not lower['measurement_failed']
audit_line = json.loads((folder/'probes/U3_lower_branch_evidence.json').read_text())['summary']['audit_line']
lower_y = lower['coefficients'][0]*target_u3['region']['center_x']+lower['coefficients'][1]
lower_angle = math.degrees(math.atan(lower['coefficients'][0]))
assert abs(lower_y-audit_line['center_y']) < .5
assert abs(lower_angle-audit_line['angle_deg']) < .5
ambiguous_u4 = next(p for im in data['images'] if im['name']=='5.png' for p in im['points'] if p['name']=='U4')
assert ambiguous_u4['auto']['status'] == 'failed'
assert ambiguous_u4['auto']['distance_px'] is None
assert ambiguous_u4['auto']['upper']['spatial_recovery_blocked_by_outer_contour']
assert ambiguous_u4['auto']['upper']['selection_mode'] == 'local_transition'
assert ambiguous_u4['auto']['upper']['selected_spatial_body_branch'] is None
result = {'points': 84, 'images': 6, 'rows_per_image': 2, 'status_counts': dict(counts),
          'actual_sampling_columns': width, 'line_horizontal_span_px': width-1,
          'all_valid_points_match_local_transition_peaks': True, 'U2_upper_center_y': target_y,
          'physical_order_checked': True, 'csv_order_and_nulls_checked': True,
          'source_image_and_workbook_hashes_checked': True,
          'algorithm_and_runner_hashes_checked': True, 'perpendicular_geometry_checked': True,
          'report_links_checked': True, 'U3_lower_center_y': lower_y, 'U3_lower_angle_deg': lower_angle,
          'U3_lower_matches_independent_pixel_observation': True,
          '5U4_unresolved_outer_lip_not_replaced_by_inner_face': True,
          'unrecovered_points_unchanged': True, 'recovered_edges': recovered, 'failed_positions': failed}
(folder / 'qa/measurement_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
