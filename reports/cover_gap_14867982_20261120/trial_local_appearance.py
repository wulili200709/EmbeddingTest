"""Offline appearance trial, using good images only to build appearance models.

Bad examples are never used to train templates or choose score thresholds.
Two independently held out good images are kept outside appearance training.
This is a product-specific experiment, not a production integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
LABELS=('roi33','roi34')
TARGETS=('cam1_debug_cam_20261119_134622_249531.png','cam1_debug_cam_20261119_163833_081283.png')
# Explicitly confirmed by the user after inspecting initial trial false alarms.
NG_OVERRIDES={'cam1_debug_cam_20260920_102403_453877.png'}
PAD=12


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def image_read(path):
    im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if im is None:raise RuntimeError(f'Cannot decode {path}')
    return im


def geometry(shape,dx=0,dy=0):
    p=np.asarray(shape['points'],np.float32)
    if len(p)==2:
        (x0,y0),(x1,y1)=p
        p=np.asarray([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float32)
    p=p+np.asarray([dx,dy],np.float32)
    w=int(round(np.linalg.norm(p[1]-p[0])))
    h=int(round(np.linalg.norm(p[3]-p[0])))
    dest=np.asarray([[PAD,PAD],[PAD+w,PAD],[PAD+w,PAD+h],[PAD,PAD+h]],np.float32)
    return dict(matrix=cv2.getPerspectiveTransform(p,dest),w=w,h=h,
                angle=0 if p[1,0]>p[0,0] else 180,points=p.tolist())


def patch_from_scene(scene,geo):
    patch=cv2.warpPerspective(scene,geo['matrix'],(geo['w']+2*PAD,geo['h']+2*PAD))
    return cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)


def grad(gray):
    x=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    y=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    return cv2.magnitude(x,y)


def masks(label,h,w):
    stable=np.zeros((h,w),np.uint8)
    inspection=np.zeros((h,w),np.uint8)
    if label=='roi33':
        stable[int(h*.77):,:]=255  # gasket under lower cover edge
        inspection[int(h*.16):int(h*.79),:]=255
    else:
        stable[:int(h*.53),:]=255  # upper gasket and outer frame
        inspection[int(h*.45):,:]=255
    stable[:,:4]=stable[:,-4:]=0
    inspection[:,:4]=inspection[:,-4:]=0
    return stable,inspection


def align(gray,reference,stable):
    score=cv2.matchTemplate(grad(gray),grad(reference),cv2.TM_CCORR_NORMED,mask=stable)
    score=np.nan_to_num(score,nan=-1.,posinf=-1.,neginf=-1.)
    _,correlation,_,pos=cv2.minMaxLoc(score)
    x,y=pos
    h,w=reference.shape
    return gray[y:y+h,x:x+w],dict(dx=x-PAD,dy=y-PAD,correlation=correlation)


def normalize(gray,reference,stable):
    # Fit one affine brightness mapping on the stationary gasket/frame only.
    valid=stable>0
    a=gray[valid].astype(np.float32)
    b=reference[valid].astype(np.float32)
    # Robustly discard extremes rather than fitting specular points.
    low,high=np.percentile(a,[10,90])
    keep=(a>=low)&(a<=high)
    aa,bb=a[keep],b[keep]
    variance=float(np.var(aa))
    gain=float(np.mean((aa-aa.mean())*(bb-bb.mean()))/variance) if variance>1 else 1.
    gain=float(np.clip(gain,.65,1.5))
    offset=float(np.median(bb-gain*aa))
    corrected=cv2.addWeighted(gray,gain,gray,0.,offset)
    return corrected,dict(gain=gain,offset=offset)


class Appearance:
    def __init__(self,label,samples):
        self.label=label
        self.names=[s['file'] for s in samples]
        h,w=samples[0]['geo']['h'],samples[0]['geo']['w']
        self.stable,self.inspection=masks(label,h,w)
        reference=samples[0]['patch'][PAD:PAD+h,PAD:PAD+w]
        aligned=[]
        self.training_alignment=[]
        for sample in samples:
            gray,loc=align(sample['patch'],reference,self.stable)
            corrected,light=normalize(gray,reference,self.stable)
            aligned.append(corrected)
            self.training_alignment.append(dict(file=sample['file'],**loc,**light))
        stack=np.asarray(aligned,np.float32)
        self.reference=np.rint(np.median(stack,axis=0)).astype(np.uint8)
        median=np.median(stack,axis=0)
        deviation=1.4826*np.median(np.abs(stack-median),axis=0)
        # Fixed initial tolerances, declared before scoring any bad images.
        self.tolerance=np.maximum(10.,3.*deviation).astype(np.float32)
        # Small residual edge variation is tolerated using good reference only.
        edge_range=cv2.dilate(self.reference,np.ones((3,3),np.uint8)).astype(np.float32)-cv2.erode(self.reference,np.ones((3,3),np.uint8)).astype(np.float32)
        self.tolerance=np.maximum(self.tolerance,.30*edge_range)

    def score(self,patch,details=False):
        aligned,loc=align(patch,self.reference,self.stable)
        corrected,light=normalize(aligned,self.reference,self.stable)
        difference=self.reference.astype(np.float32)-corrected.astype(np.float32)
        excess=np.maximum(difference-self.tolerance,0)
        binary=np.where((excess>0)&(self.inspection>0),255,0).astype(np.uint8)
        binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        n,components,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
        areas=stats[1:,cv2.CC_STAT_AREA]
        largest=int(areas.max()) if len(areas) else 0
        total=int(np.count_nonzero(binary))
        row=dict(score=largest,abnormal_area=total,**loc,**light)
        if details:
            return row,dict(reference=self.reference,corrected=corrected,binary=binary,excess=excess,
                            tolerance=self.tolerance,stable=self.stable,inspection=self.inspection)
        return row


def inventory(product):
    annotations=read_json(product/'sample_annotations.json').get('images',{})
    images={}
    rows=[]
    missing=[]
    for relative,annotation in annotations.items():
        status=dict(annotation.get('roi_status',{}))
        if Path(relative).name in NG_OVERRIDES:
            status.update({f'cam1::{l}':'NG' for l in LABELS})
        labels=[l for l in LABELS if status.get(f'cam1::{l}') in ('OK','NG')]
        path=product/relative
        if not labels:continue
        if not path.exists():
            # A copied order may leave old relative paths; search only its tree.
            matches=list(product.rglob(path.name))
            if len(matches)==1:path=matches[0]
        if not path.exists() or not path.with_suffix('.json').exists():
            missing.append(dict(file=path.name,status={l:status.get(f'cam1::{l}') for l in labels}))
            continue
        if path.name not in images:images[path.name]=image_read(path)
        shapes={s['label']:s for s in read_json(path.with_suffix('.json'))['shapes']}
        for label in labels:
            if label not in shapes:continue
            geo=geometry(shapes[label])
            rows.append(dict(file=path.name,path=path,label=label,gt=status[f'cam1::{label}'],
                             geo=geo,shape=shapes[label],patch=patch_from_scene(images[path.name],geo),
                             label_source='user confirmed mislabeled OK is actually NG' if path.name in NG_OVERRIDES else 'order sample_annotations.json'))
    # Both user-specified images are defective at both ends, per explicit request.
    for name in TARGETS:
        if name not in images:
            matches=list(product.rglob(name))
            if len(matches)!=1:continue
            images[name]=image_read(matches[0])
        path=next(product.rglob(name))
        shapes={s['label']:s for s in read_json(path.with_suffix('.json'))['shapes']}
        for label in LABELS:
            existing=next((r for r in rows if r['file']==name and r['label']==label),None)
            if existing:
                existing['gt']='NG'
                existing['label_source']='user described gaps at both ends'
            else:
                geo=geometry(shapes[label])
                rows.append(dict(file=name,path=path,label=label,gt='NG',geo=geo,shape=shapes[label],
                                 patch=patch_from_scene(images[name],geo),label_source='user described gaps at both ends'))
    return rows,images,missing


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--product',type=Path)
    parser.add_argument('--output',type=Path,default=HERE/'appearance_trial')
    args=parser.parse_args()
    product=args.product or Path(read_json(HERE/'classification.json')['product'])
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    rows,images,missing=inventory(product)
    groups=defaultdict(list)
    for r in rows:
        if r['gt']=='OK':groups[(r['label'],r['geo']['angle'])].append(r)
    # A fixed named holdout chosen among good examples, independent of bad scores.
    holdout_names={'cam1_debug_cam_20260920_102403_453877.png','cam1_debug_cam_20260920_102436_520974.png'}
    train={k:sorted([s for s in v if s['file'] not in holdout_names],key=lambda s:s['file']) for k,v in groups.items()}
    models={k:Appearance(k[0],v) for k,v in train.items() if v}
    threshold={}
    calibrations=[]
    # Score leave-one-out GOOD examples to select score cutoffs; never bad images.
    for key,samples in train.items():
        if not samples:continue
        scores=[]
        if len(samples)>1:
            for held in samples:
                model=Appearance(key[0],[s for s in samples if s is not held])
                value=model.score(held['patch'])
                scores.append(value['score'])
                calibrations.append(dict(file=held['file'],label=held['label'],angle=key[1],method='leave_one_good_out',**value))
        else:
            # One good only: synthetic residual shifts are not independent validation.
            held=samples[0]
            for dx,dy in ((0,0),(-2,0),(2,0),(0,-2),(0,2)):
                g=geometry(held['shape'],dx,dy)
                value=models[key].score(patch_from_scene(images[held['file']],g))
                scores.append(value['score'])
                calibrations.append(dict(file=held['file'],label=held['label'],angle=key[1],method='one_good_synthetic_shift',input_dx=dx,input_dy=dy,**value))
        threshold[key]=max(25,int(np.ceil(max(scores)*1.2+10)))
    predictions=[]
    detail_images=[]
    for r in rows:
        key=(r['label'],r['geo']['angle'])
        if key not in models:continue
        t0=time.perf_counter_ns()
        value,details=models[key].score(r['patch'],details=True)
        ms=(time.perf_counter_ns()-t0)/1e6
        role='target_bad' if r['file'] in TARGETS else 'other_bad' if r['gt']=='NG' else 'holdout_good' if r['file'] in holdout_names else 'training_good'
        pred='NG' if value['score']>threshold[key] else 'OK'
        row=dict(file=r['file'],label=r['label'],angle=key[1],gt=r['gt'],label_source=r['label_source'],
                 split=role,pred=pred,threshold=threshold[key],inference_ms=ms,**value)
        predictions.append(row)
        detail_images.append((row,details))
        print('RESULT',json.dumps(row),flush=True)
    stresses=[]
    for r in rows:
        if not (r['file'] in TARGETS or r['file'] in holdout_names):continue
        key=(r['label'],r['geo']['angle'])
        if key not in models:continue
        cases=[(dx,dy,1.) for dx,dy in ((-8,0),(-4,0),(-2,0),(0,0),(2,0),(4,0),(8,0),(0,-8),(0,-4),(0,-2),(0,2),(0,4),(0,8))]
        cases += [(0,0,gain) for gain in (.8,.9,1.1,1.2)]
        for dx,dy,gain in cases:
            scene=images[r['file']]
            if gain!=1.:scene=cv2.addWeighted(scene,gain,scene,0.,0.)
            g=geometry(r['shape'],dx,dy)
            value=models[key].score(patch_from_scene(scene,g))
            stresses.append(dict(file=r['file'],label=r['label'],angle=key[1],gt=r['gt'],input_dx=dx,input_dy=dy,
                                 brightness_gain=gain,pred='NG' if value['score']>threshold[key] else 'OK',threshold=threshold[key],**value))
    summary={}
    for split in ('training_good','holdout_good','target_bad','other_bad'):
        selected=[r for r in predictions if r['split']==split]
        summary[split]=dict(roi_count=len(selected),correct=sum(r['pred']==r['gt'] for r in selected),
                            files=len({r['file'] for r in selected}),wrong=[dict(file=r['file'],label=r['label'],gt=r['gt'],pred=r['pred']) for r in selected if r['gt']!=r['pred']])
    stress_summary=[]
    for name in sorted({r['file'] for r in stresses}):
        for label in LABELS:
            selected=[r for r in stresses if r['file']==name and r['label']==label]
            if selected:stress_summary.append(dict(file=name,label=label,count=len(selected),correct=sum(r['pred']==r['gt'] for r in selected),min_score=min(r['score'] for r in selected),max_score=max(r['score'] for r in selected)))
    report=dict(product=str(product),scope='Offline prototype; thresholds selected from good-only leave-one-out or synthetic shifts, no bad training. Sparse orientation-180 normal coverage.',
                settings=dict(padding_px=PAD,dark_difference_floor=10,normal_deviation_factor=3,edge_tolerance_factor=.30,score='largest connected dark residual area',holdout_good_names=sorted(holdout_names)),
                model_counts={f'{k[0]}_{k[1]}':len(v) for k,v in train.items()},thresholds={f'{k[0]}_{k[1]}':v for k,v in threshold.items()},
                ground_truth_corrections={name:{l:'NG' for l in LABELS} for name in NG_OVERRIDES},
                summary=summary,stress_summary=stress_summary,missing=missing,calibrations=calibrations,predictions=predictions,stresses=stresses)
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    arrays={}
    for (label,angle),model in models.items():
        prefix=f'{label}_{angle}'
        for field in ('reference','tolerance','stable','inspection'):
            arrays[f'{prefix}_{field}']=getattr(model,field)
    np.savez_compressed(out/'reference_models.npz',**arrays)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    selected=[(r,d) for r,d in detail_images if r['split']!='training_good']
    canvas=Image.new('RGB',(1550,100+len(selected)*205),'#f5f6f8')
    draw=ImageDraw.Draw(canvas)
    draw.text((18,15),'良品外观对比试验：左为良品参考，中为校正后的实拍，右为残留暗区',font=font,fill='#172435')
    draw.text((18,48),'错装图未参与训练；红色为超出正常波动的像素。当前为离线试验，尚未接入生产。',font=small,fill='#384b60')
    for i,(r,d) in enumerate(selected):
        y=100+i*205
        draw.text((15,y),f"{r['file']} / {r['label']} / {r['angle']}° / 标注 {r['gt']} / 判定 {r['pred']} / 暗区 {r['score']} / 阈值 {r['threshold']}",font=small,fill='#172435')
        over=cv2.cvtColor(d['corrected'],cv2.COLOR_GRAY2RGB)
        over[d['binary']>0]=[230,40,50]
        for col,arr in enumerate((d['reference'],d['corrected'],over)):
            im=Image.fromarray(arr).convert('RGB')
            im.thumbnail((495,150))
            canvas.paste(im,(15+col*510,y+36))
    canvas.save(out/'comparison.png')
    print('SUMMARY',json.dumps(summary),flush=True)
    print('STRESSES',json.dumps(stress_summary),flush=True)
    print('OUT',str(out),flush=True)


if __name__=='__main__':main()
