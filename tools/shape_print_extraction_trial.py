"""Exercise the application's native shape locator on external print samples.

Writes a separate model/recipe and evidence; never changes production recipes or
image-side annotations. This is ROI extraction, not an OK/NG classifier.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shape.core.template_core import RoiRect, MaskRect, build_multi_backend_detector, pose_infos_from_ui_values
from shape.core.recipe import ShapeRecipe, save_recipe
from shape.core.services import ShapeRunner
from shape.like_matcher import save_detector_model, load_detector_model


def read(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f'Cannot read {path}')
    return im


def write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError(f'Cannot encode {path}')
    encoded.tofile(str(path))


def extract_ink(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    bg = cv2.GaussianBlur(gray, (0, 0), 6)
    sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = ((smooth >= 110) & (smooth.astype(float)-bg.astype(float) >= 15) & (sat <= 100)).astype(np.uint8)*255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 8:
            cleaned[labels == label] = 255
    return cleaned


def render(report, output):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    rows = report['records']
    w, rh = 680, 255
    canvas = Image.new('RGB', (w*2, 85+rh*len(rows)), '#eff3f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), '实际调用 EmbeddingTest ShapeModel：壳体定位 → 印字 ROI 跟随 → 白色笔画提取', font=font, fill='#172435')
    draw.text((16, 42), '左：壳体坐标中的印字区域；右：提取的白色笔画。定位分数不是缺陷分数，不代表 OK/NG。', font=small, fill='#46556a')
    table = []
    for i, row in enumerate(rows):
        y = 85+i*rh
        name = Path(row['source']).stem.removeprefix('Cam1_')
        if row['located']:
            caption = f"{name} | 定位 {row['similarity']:.1f} | {row['locate_ms']:.0f} ms"
            for col, key in enumerate(['crop', 'ink_mask']):
                im = Image.open(output/row[key]).convert('RGB')
                canvas.paste(im, (col*w, y+38))
            score = f"{row['similarity']:.2f}"
        else:
            caption = name+' | 定位失败'
            score = '失败'
        draw.text((12,y+8),caption,font=font,fill='#172435')
        table.append(f"<tr><td>{html.escape(name)}</td><td>{score}</td><td>{row.get('ink_pixels','—')}</td><td><a href='{row.get('overlay','')}'>原图定位框</a></td></tr>")
    canvas.save(output/'overview.png')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ShapeModel 印字提取验证</title>
<style>body{font-family:'Microsoft YaHei',sans-serif;max-width:1450px;margin:30px auto;padding:0 24px;background:#f5f7fa;color:#172435;line-height:1.8}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #d5deea}img{max-width:100%}a{color:#175ec2}</style>
<h1>ShapeModel 定位与印字提取验证</h1>
<p>实际调用本工程 shape.core.template_core、ShapeRunner 和 affine_roi 跟随功能，使用原生 Original 后端。只建立一个下部壳体轮廓模板，屏蔽圆孔内部变化；印字、橙色拨片不参加模板建立。模板搜索平移、旋转与缩放，随后将紧凑印字 ROI 映射到各张图片并矫正为相同尺寸。</p>
<p>本次仅验证定位和笔画提取。没有重新训练 PatchCore、没有 OCR 识别，也没有根据定位相似度判 OK/NG。白色笔画图包含字符与分隔线；其二值化仍受浅印、模糊和反光影响，应保留原始灰度图用于后续检查。位置偏移仍保留在壳体坐标中。</p>
<table><tr><th>文件</th><th>壳体定位分数</th><th>白色像素</th><th>定位复核</th></tr>TABLE</table>
<p><a href="template_selection.png">定位模板及印字区域设置</a> · <a href="report.json">详细记录</a> · <a href="model.json">ShapeModel</a> · <a href="recipe.json">ROI 跟随配方</a></p>
<a href="overview.png"><img src="overview.png"></a></html>'''.replace('TABLE','\n'.join(table))
    (output/'review.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=ROOT / 'artifacts/reports/shape_print_extraction_20261013')
    parser.add_argument('--reuse-model',action='store_true')
    parser.add_argument('--search-mode', choices=('full-image', 'legacy-roi'), default='full-image',
                        help='Full-image search keeps a displaced housing template from being clipped; legacy-roi reproduces the earlier trial.')
    args=parser.parse_args()
    cv2.setNumThreads(4)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    reference=read(args.reference)
    # Full-resolution Golden coordinates. The reference contains no printing in this shell ROI.
    shell=RoiRect(870,1075,850,380)
    excluded=[MaskRect(60,0,730,230)]
    text_roi=[940,890,680,205]
    poses=dict(angle_start=-3.,angle_end=3.,angle_step=1.,scale_start=.86,scale_end=1.04,scale_step=.02)
    if args.reuse_model:
        detector=load_detector_model(str(output/'model.json'))
    else:
        print('Building native shell shape template...',flush=True)
        roi=reference[shell.y:shell.y+shell.h,shell.x:shell.x+shell.w].copy()
        detector,kept,skipped=build_multi_backend_detector(class_id='shell',roi_img=roi,roi_rect=shell,
             mask_rects=excluded,pose_infos=pose_infos_from_ui_values(**poses),pose_ui=poses,
             levels=[4,8],num_features=128,weak_threshold=10,strong_threshold=20,
             original_mode='auto',source_image_path=str(args.reference.resolve()))
        save_detector_model(detector,str(output/'model.json'))
        print(f'Templates: {kept}, skipped: {skipped}',flush=True)
    x,y,w,h=text_roi
    recipe=ShapeRecipe(model_path=str(output/'model.json'),reference_image=str(args.reference.resolve()),
        class_id='shell',backend='original',threshold=55.,follow_mode='affine_roi',
        reference_regions=[dict(reference_label='print_roi',output_label='print_roi',shape_type='polygon',
                                points=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]])],
        search_shape_type='rectangle' if args.search_mode == 'legacy-roi' else '',
        search_points=[[650,850],[1900,1630]] if args.search_mode == 'legacy-roi' else [])
    save_recipe(recipe,str(output/'recipe.json'))
    runner=ShapeRunner(str(output/'model.json'),detector=detector)
    template_vis=reference.copy()
    cv2.rectangle(template_vis,(shell.x,shell.y),(shell.x+shell.w,shell.y+shell.h),(0,255,0),4)
    for mask in excluded:
        cv2.rectangle(template_vis,(shell.x+mask.x,shell.y+mask.y),(shell.x+mask.x+mask.w,shell.y+mask.y+mask.h),(0,0,255),3)
    cv2.rectangle(template_vis,(x,y),(x+w,y+h),(255,255,0),4)
    write(output/'template_selection.png',template_vis)
    records=[]
    for i,path in enumerate(sorted(args.input.glob('*.png'))):
        scene=read(path)
        row=dict(source=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        try:
            located=runner.locate(scene,recipe)
            result=located.result
            points=np.float32(result.regions[0].points)
            dst=np.float32([[0,0],[w,0],[w,h],[0,h]])
            matrix=cv2.getPerspectiveTransform(points,dst)
            crop=cv2.warpPerspective(scene,matrix,(w,h))
            ink=extract_ink(crop)
            overlay=scene.copy()
            cv2.polylines(overlay,[np.int32(points)],True,(255,255,0),4,cv2.LINE_AA)
            folder=f'samples/{i:02d}'
            write(output/folder/'crop.png',crop)
            write(output/folder/'ink_mask.png',ink)
            write(output/folder/'overlay.png',overlay)
            isolated=cv2.bitwise_and(crop,crop,mask=ink)
            write(output/folder/'isolated_ink.png',isolated)
            meta=detector.get_template_meta(result.match.class_id,result.match.template_id)
            row.update(located=True,similarity=float(result.match.similarity),locate_ms=located.locate_ms,
                template_id=int(result.match.template_id),pose=meta,roi_points=points.tolist(),
                source_to_crop=matrix.tolist(),crop=f'{folder}/crop.png',ink_mask=f'{folder}/ink_mask.png',
                overlay=f'{folder}/overlay.png',ink_pixels=int(np.count_nonzero(ink)))
        except Exception as exc:
            row.update(located=False,error=str(exc))
        records.append(row)
        print(json.dumps({k:row[k] for k in ['source','located','similarity','ink_pixels','error'] if k in row},ensure_ascii=False),flush=True)
    report=dict(reference=str(args.reference.resolve()),backend='original',method='native_shape_model_affine_roi',
                template_roi=[shell.x,shell.y,shell.w,shell.h],print_roi=text_roi,pose_search=poses,
                threshold=recipe.threshold, search_mode=args.search_mode,
                search_rectangle_xyxy=[650,850,1900,1630] if args.search_mode == 'legacy-roi' else None,
                records=records,scope='localization_and_ink_extraction_only')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    render(report,output)
    print(f'Done: {output}',flush=True)


if __name__=='__main__':
    main()
