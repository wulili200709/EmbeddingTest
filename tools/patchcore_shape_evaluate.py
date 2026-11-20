"""Rebuild and evaluate two PatchCore variants on verified ShapeModel crops.

Run with the anomalib environment from the print-inspection workspace. The
original Golden alone trains each bank; two additional OK images remain held out.
"""
from __future__ import annotations
import argparse
import csv
import html
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patchcore-project', type=Path,
                        default=Path('C:/Users/goney/Desktop/2026-09-29-15-16-48'))
    parser.add_argument('--shape-root', type=Path, default=ROOT/'reports/shape_print_extraction_20261013')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/shape_patchcore_20261013')
    args = parser.parse_args()
    sys.path.insert(0, str(args.patchcore_project.resolve()))
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from reference_diff import file_sha256, read_image, write_image
    from housing_inspection import inspect_aligned_print
    from patchcore_housing_trial import run_variant

    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    shape = json.loads((args.shape_root/'report.json').read_text(encoding='utf-8'))
    # Recheck geometry in the same Python/OpenCV runtime that produced the
    # ShapeModel crops. 4.12 and 4.13 differ slightly in warp interpolation.
    verification_code = '''import sys,json,cv2,numpy as np
from pathlib import Path
p=Path(sys.argv[1]); r=json.loads((p/'report.json').read_text(encoding='utf-8'))
def read(path): return cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
for row in r['records']:
 a=read(p/row['crop']); b=cv2.warpPerspective(read(Path(row['source'])),np.array(row['source_to_crop']),(680,205))
 assert np.array_equal(a,b),row['source']
print(json.dumps({'all_saved_crops_reproduced_exactly':True,'opencv_version':cv2.__version__,'count':len(r['records'])}))
'''
    verified = subprocess.run(['py', '-3.12', '-c', verification_code, str(args.shape_root.resolve())],
                              check=True, capture_output=True, text=True, encoding='utf-8')
    crop_verification = json.loads(verified.stdout)
    config = json.loads((args.patchcore_project/'inspection_20261012.json').read_text(encoding='utf-8'))
    config.update(print_roi=[0., 0., 1., 1.], ink_min_gray=110, ink_min_contrast=15,
                  ink_min_component_area=8)
    reference_path = Path(shape['reference'])
    reference_hash = file_sha256(reference_path)
    ref_row = next(r for r in shape['records'] if r['sha256'] == reference_hash)
    reference = read_image(args.shape_root/ref_row['crop'])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples, preparation = [], []
    for i, row in enumerate(shape['records']):
        if not row['located']:
            raise RuntimeError('Shape localization failed: '+row['source'])
        if file_sha256(Path(row['source'])) != row['sha256']:
            raise RuntimeError('Source image changed after ShapeModel extraction: '+row['source'])
        crop_path = args.shape_root/row['crop']
        crop = read_image(crop_path)
        valid = np.full(crop.shape[:2], 255, np.uint8)
        checked = inspect_aligned_print(reference, crop, valid,
            dict(alignment_valid=True, alignment_method='native_shape_model_affine_roi',
                 shape_similarity=row['similarity']), config)
        write_image(output/'prepared'/f'{i:02d}_aligned.png', crop)
        write_image(output/'prepared'/f'{i:02d}_difference.png', checked.difference)
        samples.append(dict(source=row['source'], sha256=row['sha256'], rgb=crop,
                            residual=checked.difference, alignment=dict(alignment_valid=True)))
        preparation.append(dict(source=row['source'], sha256=row['sha256'],
            shape_similarity=row['similarity'], source_to_crop=row['source_to_crop'],
            shape_crop_sha256=file_sha256(crop_path), print_dx_px=checked.metrics['print_dx_px'],
            print_dy_px=checked.metrics['print_dy_px'], print_shift_correlation=checked.metrics['print_shift_correlation']))
    report = dict(reference=str(reference_path), reference_sha256=reference_hash,
                  localization='EmbeddingTest native ShapeModel Original + affine_roi',
                  shape_report=str((args.shape_root/'report.json').resolve()),
                  shape_model_sha256=file_sha256(args.shape_root/'model.json'),
                  shape_recipe_sha256=file_sha256(args.shape_root/'recipe.json'),
                  training_unique_good_count=1, independent_good_count=2,
                  augmentation_count=0, config=config, preparations=preparation,
                  crop_verification=crop_verification, variants={})
    for name in ['rgb', 'residual']:
        print('Starting ShapeModel crop / '+name, flush=True)
        variant = run_variant(name, samples, output, reference_hash)
        variant['independent_good_count'] = 2
        report['variants'][name] = variant
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    # User-provided labels are consulted only after both models have finished.
    evaluation = {}
    for name, variant in report['variants'].items():
        good = [r['score'] for r in variant['records'] if '正面合格' in Path(r['source']).name]
        bad = [r['score'] for r in variant['records'] if not r['reference_self_test'] and '正面合格' not in Path(r['source']).name]
        evaluation[name] = dict(held_out_good_scores=good, bad_scores=bad,
            max_held_out_good=max(good), min_bad=min(bad),
            separable_by_one_threshold=max(good)<min(bad),
            bad_count_above_observed_good_max=sum(s>max(good) for s in bad),
            bad_count=len(bad), gap=min(bad)-max(good),
            interpretation='post-hoc score comparison only, not a calibrated threshold or production detection rate')
    report['label_comparison_after_inference'] = evaluation
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for i, sample in enumerate(samples):
        record = dict(filename=Path(sample['source']).name,
                      shape_similarity=preparation[i]['shape_similarity'],
                      reference_self_test=sample['sha256']==reference_hash)
        for name, variant in report['variants'].items():
            score = variant['records'][i]['score']
            record[name+'_score'] = score
            record[name+'_above_observed_good_max'] = score > evaluation[name]['max_held_out_good']
        record['decision'] = 'UNCALIBRATED'
        rows.append(record)
    with (output/'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    w, h = 680, 255
    canvas = Image.new('RGB', (w*3, 145+h*len(samples)), '#eff3f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), 'ShapeModel 定位后重新评估 PatchCore：仅原参考图训练，两张新增 OK 保留为独立测试', font=font, fill='#172435')
    draw.text((16, 42), '蓝低、红高；每个模型使用统一色标。两列分数不能直接横向比较，未标定生产 OK/NG 阈值。', font=small, fill='#46556a')
    for col, title in enumerate(['ShapeModel 跟随裁切的印字区域', '字符区域原图 + PatchCore', '参考差分 + PatchCore']):
        draw.text((col*w+16, 73), title, font=font, fill='#172435')
        if col:
            name = 'rgb' if col==1 else 'residual'
            vmax = report['variants'][name]['heatmap_vmax']
            bar = np.tile(np.arange(256,dtype=np.uint8),(14,1))
            bar = cv2.cvtColor(cv2.applyColorMap(bar,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
            canvas.paste(Image.fromarray(bar).resize((260,14)),(col*w+16,109))
            draw.text((col*w+290,102),f'0 → {vmax:.3f}',font=small,fill='#46556a')
    table=[]
    for i,row in enumerate(rows):
        y=145+i*h
        label=row['filename'].removeprefix('Cam1_').removesuffix('.png')
        for col,title in enumerate([label,f"异常分数 {row['rgb_score']:.4f}",f"异常分数 {row['residual_score']:.4f}"]):
            draw.text((col*w+10,y+7),title,font=font,fill='#172435')
            path=output/'prepared'/f'{i:02d}_aligned.png' if col==0 else output/report['variants']['rgb' if col==1 else 'residual']['records'][i]['overlay']
            canvas.paste(Image.open(path).convert('RGB'),(col*w,y+40))
        table.append(f"<tr><td>{html.escape(label)}</td><td>{row['rgb_score']:.4f}</td><td>{row['residual_score']:.4f}</td><td>{'参考自检' if row['reference_self_test'] else '独立于训练图；未标定判定阈值'}</td></tr>")
    canvas.save(output/'overview.png')
    summaries=[]
    for name,title in [('rgb','字符区域原图模型'),('residual','参考差分模型')]:
        e=evaluation[name]
        summaries.append(f"<p><b>{title}：</b>两张独立 OK 分数为 {e['held_out_good_scores'][0]:.2f}、{e['held_out_good_scores'][1]:.2f}；7 张 NG 中，有 {e['bad_count_above_observed_good_max']} 张高于这两张 OK 的最高分。全部已知 OK/NG 是否可用单一分数阈值分开：{'是，仅限本批观察' if e['separable_by_one_threshold'] else '否，分数存在重叠'}。</p>")
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ShapeModel + PatchCore 检测实测</title>
<style>body{font-family:'Microsoft YaHei',sans-serif;max-width:1550px;margin:30px auto;padding:0 24px;background:#f5f7fa;color:#172435;line-height:1.8}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #d5deea}img{max-width:100%}a{color:#175ec2}.note{padding:16px;background:#e6edf5}</style>
<h1>ShapeModel 定位后的 PatchCore 检测</h1>
<p>使用已验证的 10 张 ShapeModel 跟随裁切结果，每张 680 × 205 像素；逐张重新计算源图变换并验证与保存的裁切一致。两套 PatchCore 都按新的输入区域重新建立特征库，仅使用原参考良品，新增两张 OK 未参与训练。输入补边至 704 × 224，保持原始像素尺度。ResNet18、layer1/layer2、10% coreset、5 邻居，与上一试验一致。</p>
<p>原图模型输入为紧凑的彩色印字区域，并未宣称所有壳体像素都被清除；差分模型使用灰度差、边缘差及笔画掩膜差。印字位置始终保留在壳体坐标中，没有按文字再次对齐。二值化沿用 ShapeModel 提取阶段的亮度/对比度参数。</p>
<h2>实测分数对照</h2>SUMMARY
<div class="note">“高于两张 OK 的最高分”仅为事后对照，不能当作经过独立校准的检出率。当前只有两张独立良品，不设置生产阈值。差分模型的唯一训练输入仍是参考图与自身的全零差分，没有学到真实良品的差分波动。模型高分不等于识别出了错误字符或精确定位到了断笔。</div>
<table><tr><th>图片</th><th>字符区域原图模型</th><th>参考差分模型</th><th>状态</th></tr>TABLE</table>
<h2>统一色标热力图</h2><a href="overview.png"><img src="overview.png"></a>
<p><a href="results.csv">结果 CSV</a> · <a href="report.json">参数及详细记录</a>。rgb/residual 目录各自保存模型、输入、原始浮点热力图、彩色热力图和叠加图。</p></html>'''
    page=page.replace('SUMMARY','\n'.join(summaries)).replace('TABLE','\n'.join(table))
    (output/'review.html').write_text(page,encoding='utf-8')
    print(json.dumps(evaluation,ensure_ascii=False),flush=True)
    print(f'Done: {output}',flush=True)


if __name__=='__main__':
    main()
