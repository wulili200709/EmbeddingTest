"""Read-only cover-gap investigation. Writes diagnostics only beside this script."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFont

from algorithms.embedding import (
    OrtBackboneRunner, TF, _build_image_roi_context, _resolve_roi_image,
    embed_batch, load_register_model_npz, predict_one_with_model,
)
from algorithms.embedding_core import compute_prototypes
from shape.core.locator import load_recipe_for_product
from shape.core.services import ShapeLocateService

OUT = Path(__file__).resolve().parent
PRODUCT = next(Path('C:/Users/goney/Desktop').glob('*/14867982/14867982'))
TARGETS = ('cam1_debug_cam_20261119_134622_249531',
           'cam1_debug_cam_20261119_163833_081283')
LABELS = ('roi33', 'roi34')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def bbox(shape):
    pts = np.asarray(shape['points'])
    lo, hi = pts.min(0), pts.max(0)
    return [int(round(x)) for x in (*lo, *(hi-lo))]


def classify(e, model):
    pred, diff, ok, ng = predict_one_with_model(e, model)
    ok_scores, ng_scores = model.ok_bank @ e, model.ng_bank @ e
    return dict(pred=pred, diff=diff, sim_ok=ok, sim_ng=ng,
                ok_bank_max=float(ok_scores.max()), ng_bank_max=float(ng_scores.max()),
                nearest_ng_index=int(ng_scores.argmax()))


def make_runner():
    cache = ROOT / '.cache/ort_backbones/b0_cpu_v1.ort'
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(cache), sess_options=options, providers=['CPUExecutionProvider'])
    return OrtBackboneRunner(session=session, input_name=session.get_inputs()[0].name,
                             output_name=session.get_outputs()[0].name,
                             input_shape=tuple(session.get_inputs()[0].shape),
                             providers=('CPUExecutionProvider',), device='cpu', model_path=str(cache),
                             cpu_inference_chunk_size=8)


def infer_crops(runner, crops):
    batch = torch.stack([TF(crop) for crop in crops])
    return runner.run(batch)


def visual_overview():
    canvas = Image.new('RGB', (1600, 1260), '#f5f5f5')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    for col, stem in enumerate(TARGETS):
        path = PRODUCT / 'debug_capture' / f'{stem}.png'
        im = Image.open(path).convert('RGB')
        shapes = {s['label']: s for s in read_json(path.with_suffix('.json'))['shapes']}
        annotated = im.copy()
        ad = ImageDraw.Draw(annotated)
        for label, color in [('roi33', '#ff4545'), ('roi34', '#00ff77')]:
            x,y,w,h = bbox(shapes[label])
            ad.rectangle((x,y,x+w,y+h), outline=color, width=5)
            ad.text((x,y-30),label,fill=color,font=font)
        annotated.thumbnail((560, 770))
        xoff = col*800
        draw.text((xoff+20,15),stem.rsplit('_',2)[-2]+' / saved ROIs',fill='black',font=font)
        canvas.paste(annotated,(xoff+110,55))
        for row,(name,cy) in enumerate([('TOP',260),('BOTTOM',1625)]):
            crop = im.crop((380,cy-75,1110,cy+75))
            crop.thumbnail((760,160))
            yy=865+row*185
            canvas.paste(crop,(xoff+20,yy))
            draw.text((xoff+20,yy-25),name+' - original pixels',fill='black',font=small)
    canvas.save(OUT/'roi_and_gap_comparison.png')


def main():
    torch.set_num_threads(4)
    models = {label:load_register_model_npz(str(PRODUCT/f'cam1__{label}_register_model_b0.npz')) for label in LABELS}
    runner = make_runner()
    annotations = read_json(PRODUCT/'sample_annotations.json')['images']
    session = read_json(PRODUCT/'session.json')
    result = {'product':str(PRODUCT),'targets':{},'samples':[], 'scope':'No production files modified.'}
    visual_overview()
    for stem in TARGETS:
        path = PRODUCT/'debug_capture'/f'{stem}.png'
        context = _build_image_roi_context(str(path),require_shapes=True)
        target = {'saved_rois':{l:bbox(context.shape_by_label[l]) for l in LABELS},'scores':{}}
        crops=[]
        variants=[]
        for label in LABELS:
            original=_resolve_roi_image(context,label_name=label)
            for angle in (0,180):
                crops.append(original if angle==0 else original.transpose(Image.Transpose.ROTATE_180))
                variants.append((label,angle))
        emb=infer_crops(runner,crops)
        for (label,angle),e in zip(variants,emb):
            target['scores'][f'{label}_rotation{angle}']=classify(e,models[label])
        result['targets'][stem]=target
        print('TARGET',stem,json.dumps(target),flush=True)

    # Every locally available, explicitly labeled sample; targets are included in training.
    train_names={Path(x).name for x in session['train_files']}
    for relative, annotation in annotations.items():
        path=PRODUCT/relative
        labels=[l for l in LABELS if annotation.get('roi_status',{}).get(f'cam1::{l}') in ('OK','NG')]
        if not path.exists() or not path.with_suffix('.json').exists() or not labels:
            continue
        context=_build_image_roi_context(str(path),require_shapes=True)
        labels=[l for l in labels if l in context.shape_by_label]
        if not labels:
            continue
        embeddings=embed_batch(str(path),runner,labels,device='cpu')
        for label,e in zip(labels,embeddings):
            model=models[label]
            scores=classify(e,model)
            bank=model.ok_bank if annotation['roi_status'][f'cam1::{label}']=='OK' else model.ng_bank
            distances=np.linalg.norm(bank-e,axis=1)
            row=dict(file=path.name,label=label,gt=annotation['roi_status'][f'cam1::{label}'],
                     in_session_train=path.name in train_names,bbox=bbox(context.shape_by_label[label]),
                     nearest_same_class_l2=float(distances.min()),**scores)
            if row['gt']=='NG' and distances.min()<0.002 and len(bank)>1:
                reduced=np.delete(bank,distances.argmin(),axis=0)
                _,proto=compute_prototypes(model.ok_bank,reduced)
                ng=float(proto.reshape(-1)@e)
                row['leave_self_out']=dict(sim_ng=ng,diff=scores['sim_ok']-ng,
                                          pred='OK' if scores['sim_ok']-ng>=model.margin else 'NG')
            result['samples'].append(row)
            print('SAMPLE',json.dumps(row),flush=True)
    result['model_counts']={l:{'ok':len(m.ok_bank),'ng':len(m.ng_bank)} for l,m in models.items()}
    result['logs']=[]
    for path in (PRODUCT/'test_logs').glob('*.csv'):
        with path.open(encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row.get('roi_label') in LABELS:
                    result['logs'].append(row)
    write_json('classification.json',result)

    # Locate in memory only: do not use the API that rewrites LabelMe sidecars.
    service=ShapeLocateService()
    recipe=load_recipe_for_product(str(PRODUCT),'cam1')
    recipe.model_path=str(PRODUCT/'shape/cam1/model.json')
    for stem in TARGETS:
        path=PRODUCT/'debug_capture'/f'{stem}.png'
        image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        located=service.locate(image,recipe)
        match=located.result.match
        detector=service.runner_for_model(recipe.model_path).detector
        metadata=detector.get_template_meta(match.class_id,match.template_id)
        saved={s['label']:s for s in read_json(path.with_suffix('.json'))['shapes']}
        info={'similarity':match.similarity,'template_id':match.template_id,'backend':match.backend,
              'metadata':metadata,'locate_ms':located.locate_ms,'regions':{}}
        runtime_crops=[]
        runtime_labels=[]
        for region in located.result.regions:
            if region.label_name not in LABELS:
                continue
            label=region.label_name
            old=np.asarray(saved[label]['points'])
            new=np.asarray(region.points)
            info['regions'][label]={'bbox':region.bbox,'points':region.points,
                                   'max_point_delta_px':float(np.abs(old-new).max())}
            context=_build_image_roi_context(str(path),require_shapes=True)
            context.shape_by_label[label]={'shape_type':'polygon','points':region.points}
            runtime_crops.append(_resolve_roi_image(context,label_name=label))
            runtime_labels.append(label)
        for label,e in zip(runtime_labels,infer_crops(runner,runtime_crops)):
            info['regions'][label]['score']=classify(e,models[label])
        result['targets'][stem]['relocated']=info
        write_json('classification.json',result)
        print('LOCATED',stem,json.dumps(info),flush=True)
    print('DONE',str(OUT),flush=True)


if __name__=='__main__':
    main()
