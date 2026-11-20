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
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
names = [f'{row}{i}' for row in 'UL' for i in range(1, 8)]
rows = list(csv.DictReader((folder / 'measurement_comparison.csv').open(encoding='utf-8-sig', newline='')))
assert len(rows) == 84 and len(data['images']) == 6
assert data['meta']['method'] == 'body_outline'
width = data['meta']['fit_width']
assert width == 101 and data['meta']['previous_fit_width'] == 141
assert digest(data['meta']['workbook']) == data['meta']['workbook_sha256']
assert digest(root / 'algorithms/measurement_opening_body_edge.py') == data['meta']['algorithm_sha256']
assert digest(root / 'tools/compare_lab_opening_lines.py') == data['meta']['runner_sha256']
assert data['meta']['manual_reviewed_initially'] == 0
assert [(r['image'], r['point']) for r in rows] == [(f'{i}.png', n) for i in range(1, 7) for n in names]
by_key = {(r['image'], r['point']): r for r in rows}
counts = Counter()
failed = []
for image in data['images']:
    assert digest(Path(data['meta']['input_dir']) / image['name']) == image['sha256']
    assert digest(folder / image['url']) == image['sha256']
    assert [p['name'] for p in image['points']] == [f'{row}{i}' for row in 'LU' for i in range(7, 0, -1)]
    for index, point in enumerate(image['points']):
        assert point['physical_row'] == ('top' if index < 7 else 'bottom')
        assert point['physical_col'] == index % 7 + 1
        auto, row = point['auto'], by_key[(image['name'], point['name'])]
        counts[auto['status']] += 1
        for edge in (auto['upper'], auto['lower']):
            assert len(edge['points']) == len(edge['anchor_mask']) == len(edge['valid_mask']) == width
            low, high = edge['anchor_abs_dx_range_px']
            assert abs(low - 40*(width-1)/140) < 1e-9
            assert abs(high - 65*(width-1)/140) < 1e-9
            assert edge['span'] == [point['region']['center_x']-50, point['region']['center_x']+50]
            if edge['segment']:
                assert edge['segment'][1][0]-edge['segment'][0][0] == width-1
            for xy, anchor, valid in zip(edge['points'], edge['anchor_mask'], edge['valid_mask']):
                assert all(math.isfinite(v) for v in xy)
                if anchor:
                    assert valid and low <= abs(xy[0] - point['region']['center_x']) <= high
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
result = {'points': 84, 'images': 6, 'rows_per_image': 2, 'status_counts': dict(counts),
          'actual_sampling_columns': width, 'line_horizontal_span_px': width-1,
          'physical_order_checked': True, 'csv_order_and_nulls_checked': True,
          'source_image_and_workbook_hashes_checked': True,
          'algorithm_and_runner_hashes_checked': True, 'perpendicular_geometry_checked': True,
          'report_links_checked': True, 'failed_positions': failed}
(folder / 'qa/measurement_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
