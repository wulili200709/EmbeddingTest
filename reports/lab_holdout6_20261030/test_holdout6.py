"""Read-only recalculation: train on images 1-5; evaluate image 6."""
from pathlib import Path
import collections
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile

SOURCE = Path('C:/Users/goney/Desktop/DC尺寸数据测试/Lab Test/新版取边与原数据对比_20261016.xlsx')
HERE = Path(__file__).resolve().parent
NAMES = [f'{side}{i}' for side in ['U', 'L'] for i in range(1, 8)]
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_sheet(z, part, strings):
    cells = {}
    for cell in ET.fromstring(z.read(part)).findall('.//s:c', NS):
        value = cell.find('s:v', NS)
        value = value.text if value is not None else None
        kind = cell.get('t')
        if kind == 's':
            value = strings[int(value)]
        elif kind == 'inlineStr':
            value = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
        elif value is not None and kind not in ('str', 'e'):
            value = float(value)
        cells[cell.get('r')] = value
    return cells


def fit(training, field):
    groups = {name: [r for r in training if r['name'] == name] for name in NAMES}
    ratios = {name: statistics.fmean(r['lab_mm'] for r in group) / statistics.fmean(r[field] for r in group)
              for name, group in groups.items()}
    scale = statistics.fmean(ratios.values())
    offsets = {name: statistics.fmean(r['lab_mm'] - scale * r[field] for r in group)
               for name, group in groups.items()}
    return {'scale_mm_per_px': scale, 'offsets_add_mm': offsets,
            'training_count_by_point': {name: len(group) for name, group in groups.items()}}


def evaluate(test, field, params):
    predictions = []
    for row in test:
        prediction = params['scale_mm_per_px'] * row[field] + params['offsets_add_mm'][row['name']] if row['paired'] else None
        error = prediction - row['lab_mm'] if prediction is not None else None
        predictions.append({'name': row['name'], 'lab_mm': row['lab_mm'], 'camera_px': row[field],
                            'predicted_mm': prediction, 'error_mm': error})
    valid = [r for r in predictions if r['error_mm'] is not None]
    worst = max(valid, key=lambda r: abs(r['error_mm']))
    return {'parameters': params, 'predictions': predictions, 'metrics': {
        'count': len(valid), 'mae_mm': statistics.fmean(abs(r['error_mm']) for r in valid),
        'rmse_mm': math.sqrt(statistics.fmean(r['error_mm'] ** 2 for r in valid)),
        'max_abs_mm': abs(worst['error_mm']), 'max_abs_point': worst['name'],
        'mean_error_mm': statistics.fmean(r['error_mm'] for r in valid)}}


def main():
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with zipfile.ZipFile(SOURCE) as z:
        strings = [''.join(t.text or '' for t in si.findall('.//s:t', NS))
                   for si in ET.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else []
        sheet = read_sheet(z, 'xl/worksheets/sheet2.xml', strings)
        calibration = read_sheet(z, 'xl/worksheets/sheet3.xml', strings)
    records = [{'image': sheet[f'A{r}'], 'name': sheet[f'B{r}'], 'old_px': sheet[f'D{r}'],
                'new_px': sheet.get(f'E{r}'), 'lab_mm': sheet[f'G{r}'],
                'paired': isinstance(sheet.get(f'E{r}'), (int, float)), 'source_row': r}
               for r in range(6, 90)]
    train = [r for r in records if r['image'] in [f'{i}.png' for i in range(1, 6)] and r['paired']]
    test = sorted([r for r in records if r['image'] == '6.png'], key=lambda r: NAMES.index(r['name']))
    assert len(train) == 66 and len(test) == 14
    assert all(r['image'] != '6.png' for r in train)
    methods = {kind: evaluate(test, f'{kind}_px', fit(train, f'{kind}_px')) for kind in ['old', 'new']}
    for kind, column, scale_col, offset_col in [('old', 'R', 'I', 'K'), ('new', 'S', 'J', 'L')]:
        for index, (row, pred) in enumerate(zip(test, methods[kind]['predictions'])):
            cached = sheet.get(f'{column}{row["source_row"]}')
            if row['paired']:
                assert abs(pred['error_mm'] - cached) < 1e-12, (row['name'], pred['error_mm'], cached)
            else:
                assert cached in (None, '')
            assert abs(methods[kind]['parameters']['scale_mm_per_px'] - calibration[f'{scale_col}{90+index}']) < 1e-12
            assert abs(methods[kind]['parameters']['offsets_add_mm'][row['name']] - calibration[f'{offset_col}{90+index}']) < 1e-12
    # Train-only fitting is invariant to any held-out values or labels.
    mutated = [dict(r, lab_mm=999, new_px=999, old_px=999) if r['image'] == '6.png' else dict(r) for r in records]
    train_after_mutation = [r for r in mutated if r['image'] in [f'{i}.png' for i in range(1, 6)] and r['paired']]
    assert fit(train_after_mutation, 'new_px') == methods['new']['parameters']
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    result = {'training_images': [f'{i}.png' for i in range(1, 6)], 'training_pairs': len(train),
              'training_missing': [{'image': r['image'], 'name': r['name']} for r in records if r['image'] != '6.png' and not r['paired']],
              'test_image': '6.png', 'test_pairs': sum(r['paired'] for r in test),
              'test_missing': [r['name'] for r in test if not r['paired']],
              'methods': methods, 'source_workbook': str(SOURCE), 'source_sha256': before,
              'existing_leave_one_out_values_match': True, 'held_out_value_mutation_does_not_affect_parameters': True,
              'source_unchanged': True,
              'scope': 'Fresh parameter fit on the first five images only; existing edge algorithm is unchanged and was developed on this six-image batch.'}
    (HERE / 'holdout6_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
