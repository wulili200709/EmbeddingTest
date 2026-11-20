"""Time a representative local comparison pipeline, not a validated detector.

Images and normal appearance statistics are loaded once, outside timed loops.
Production models, recipes, images and sidecars are never written.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = json.loads((OUT/'classification.json').read_text(encoding='utf-8'))
PRODUCT = Path(SOURCE['product'])
LABELS = ('roi33', 'roi34')
PAD = 8


def annotation(path):
    return {s['label']:s for s in json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))['shapes']}


def read_image(path):
    return cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)


def geometry(shape):
    points=np.asarray(shape['points'],np.float32)
    w=int(round(float(np.linalg.norm(points[1]-points[0]))))
    h=int(round(float(np.linalg.norm(points[3]-points[0]))))
    destination=np.asarray([[PAD,PAD],[PAD+w,PAD],[PAD+w,PAD+h],[PAD,PAD+h]],np.float32)
    return dict(matrix=cv2.getPerspectiveTransform(points,destination),w=w,h=h,
                orientation=0 if points[1,0]>points[0,0] else 180)


def crop_padded(image,geo):
    patch=cv2.warpPerspective(image,geo['matrix'],(geo['w']+2*PAD,geo['h']+2*PAD),flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)


def gradient(gray):
    dx=cv2.convertScaleAbs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3))
    dy=cv2.convertScaleAbs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))
    return cv2.add(dx,dy)


def build_models():
    banks={}
    for row in SOURCE['samples']:
        if row['gt']!='OK':continue
        path=PRODUCT/'debug_capture'/row['file']
        image=read_image(path)
        geo=geometry(annotation(path)[row['label']])
        padded=crop_padded(image,geo)
        crop=padded[PAD:PAD+geo['h'],PAD:PAD+geo['w']]
        banks.setdefault((row['label'],geo['orientation']),[]).append(crop)
    models={}
    for key,crops in banks.items():
        stack=np.asarray(crops,np.float32)
        reference=np.rint(stack.mean(axis=0)).astype(np.uint8)
        h,w=reference.shape
        stable=np.zeros((h,w),np.uint8)
        stable[:max(4,h//5),:]=255
        stable[-max(4,h//8):,:]=255
        mean,std=cv2.meanStdDev(reference,mask=stable)
        models[key]=dict(reference=reference,gradient=gradient(reference),stable=stable,
                         mean=float(mean[0,0]),std=float(std[0,0]),
                         threshold=np.clip(8+3*stack.std(axis=0),8,255).astype(np.uint8),
                         normal_count=len(crops))
    return models


def run(image,geometries,models,record_stages=False):
    stages=dict(crop_and_gray=0.,local_alignment=0.,brightness=0.,difference_and_regions=0.)
    largest=[]
    for label,geo in geometries.items():
        model=models[(label,geo['orientation'])]
        t0=time.perf_counter_ns()
        gray=crop_padded(image,geo)
        t1=time.perf_counter_ns()
        response=cv2.matchTemplate(gradient(gray),model['gradient'],cv2.TM_CCORR_NORMED,mask=model['stable'])
        response=np.nan_to_num(response,nan=-1,posinf=-1,neginf=-1)
        _,_,_,location=cv2.minMaxLoc(response)
        x,y=location
        aligned=gray[y:y+geo['h'],x:x+geo['w']]
        t2=time.perf_counter_ns()
        mean,std=cv2.meanStdDev(aligned,mask=model['stable'])
        gain=float(np.clip(model['std']/max(float(std[0,0]),1.),.8,1.25))
        offset=model['mean']-gain*float(mean[0,0])
        corrected=cv2.addWeighted(aligned,gain,aligned,0.,offset)
        t3=time.perf_counter_ns()
        dark_difference=cv2.subtract(model['reference'],corrected)
        mask=cv2.compare(dark_difference,model['threshold'],cv2.CMP_GT)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        count,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        largest.append(int(stats[1:,cv2.CC_STAT_AREA].max()) if count>1 else 0)
        t4=time.perf_counter_ns()
        if record_stages:
            for name,duration in zip(stages,((t1-t0),(t2-t1),(t3-t2),(t4-t3))):
                stages[name]+=duration/1e6
    return largest,stages


def summary(values):
    return {k:float(v) for k,v in zip(('median_ms','p95_ms','min_ms','max_ms'),
            (np.median(values),np.percentile(values,95),min(values),max(values)))}


def main():
    cv2.setNumThreads(1)
    t0=time.perf_counter()
    models=build_models()
    preparation_ms=(time.perf_counter()-t0)*1000
    cases=[]
    for stem in SOURCE['targets']:
        path=PRODUCT/'debug_capture'/f'{stem}.png'
        shapes=annotation(path)
        cases.append((stem,read_image(path),{label:geometry(shapes[label]) for label in LABELS}))
    for _ in range(10):
        for _,image,geometries in cases:run(image,geometries,models)
    result=dict(scope='Prototype CPU compute for two local ROIs per product. Detection accuracy NOT validated. Excludes camera exposure, IO, file reading, whole-product localization, model preparation, UI, other inspection items.',
                opencv_version=cv2.__version__,opencv_threads=cv2.getNumThreads(),padding_px=PAD,
                model_preparation_including_disk_ms=preparation_ms,
                normal_counts={f'{label}_{angle}':m['normal_count'] for (label,angle),m in models.items()},
                cases={})
    timing={stem:[] for stem,_,_ in cases}
    breakdown={stem:[] for stem,_,_ in cases}
    for _ in range(80):
        for stem,image,geometries in cases:
            started=time.perf_counter_ns()
            _,stages=run(image,geometries,models,record_stages=True)
            timing[stem].append((time.perf_counter_ns()-started)/1e6)
            breakdown[stem].append(stages)
    for stem,_,geometries in cases:
        result['cases'][stem]=dict(roi_sizes={l:[g['w'],g['h']] for l,g in geometries.items()},
                                  repetitions=len(timing[stem]),**summary(timing[stem]),
                                  median_stages_ms={name:float(np.median([r[name] for r in breakdown[stem]])) for name in breakdown[stem][0]})
    result['combined']=summary([v for values in timing.values() for v in values])
    (OUT/'local_comparison_timing.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
