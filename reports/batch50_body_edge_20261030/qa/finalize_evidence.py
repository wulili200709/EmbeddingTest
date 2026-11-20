"""Attach already verified full-image evidence without changing measurements."""
from pathlib import Path
import json
import shutil
import html
from html.parser import HTMLParser

QA = Path(__file__).resolve().parent
OUT = QA.parent
ROOT = OUT.parents[1]
data_path = OUT / 'measurements.json'
data = json.loads(data_path.read_text(encoding='utf-8'))
pilot = json.loads((OUT / 'pilot/measurements.json').read_text(encoding='utf-8'))
checks = json.loads((OUT / 'pilot/qa/full_vs_crop.json').read_text(encoding='utf-8'))
assert len(data['images']) == 50 and data['meta']['source_hashes_verified']
assert len(checks) == 14 and pilot['meta']['full_vs_crop_first_image_verified']
assert data['meta']['algorithm_sha256'] == pilot['meta']['algorithm_sha256']
assert data['images'][0]['sha256'] == pilot['images'][0]['sha256']
for final, test, check in zip(data['images'][0]['points'], pilot['images'][0]['points'], checks):
    assert final['name'] == test['name'] == check['name']
    assert final['auto'] == test['auto']
    assert check['absolute_difference_px'] is None or check['absolute_difference_px'] < 1e-8
shutil.copy2(OUT / 'pilot/qa/full_vs_crop.json', QA / 'full_vs_crop.json')
data['meta']['full_vs_crop_first_image_verified'] = True
data['meta']['full_vs_crop_evidence'] = 'qa/full_vs_crop.json; same first-image SHA256, algorithm, parameters and measurements as the audited pilot'
temp = OUT / 'measurements.finalizing.json'
temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
temp.replace(data_path)

workbook = ROOT.parent / 'outputs/01a138dc-4174-7de1-bd9f-fd953646ca1e/batch50_20261030/50张图片_最新版开口检测结果_20261030.xlsx'
report = OUT / '检测标注报告.html'
page = report.read_text(encoding='utf-8')
link = f'<p><a href="{html.escape(workbook.as_uri())}">打开完整检测值 Excel（参考毫米、像素、700行明细）</a></p>'
if workbook.as_uri() not in page:
    page = page.replace('<h2>逐图参考毫米值</h2>', link + '<h2>逐图参考毫米值</h2>')
    report.write_text(page, encoding='utf-8')

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.images = []; self.sections = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'img': self.images.append(a['src'])
        if tag == 'section': self.sections.append(a['id'])

links = Links(); links.feed(page)
assert len(links.images) == 50 and len(links.sections) == 50
assert len(set(links.sections)) == 50
assert all((OUT / p).is_file() for p in links.images)
assert all((OUT / f'overviews/{i:02d}.jpg').is_file() for i in range(1, 51))
print(json.dumps({'images': 50, 'measurements': 700, 'status_counts': data['meta']['status_counts'],
                  'numeric_count': data['meta']['numeric_count'], 'full_image_evidence_attached': True,
                  'report_image_links_verified': 50, 'workbook_target': str(workbook)}, ensure_ascii=False))
