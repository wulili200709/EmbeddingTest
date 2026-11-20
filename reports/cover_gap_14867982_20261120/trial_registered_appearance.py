"""Compare the same fixed appearance model after independent frame registration.

Reuses the offline trial runner; replaces its preparation/normalization hooks
only. No production modules, order files, annotations or models are changed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

import trial_local_appearance as trial

ORIGINAL_INVENTORY=trial.inventory
ORIGINAL_GEOMETRY=trial.geometry
REGISTRATION=[]


def affine33(matrix):return np.vstack((matrix,np.asarray([0.,0.,1.])))


def register_one(raw,reference_gray,matrix):
    started=time.perf_counter_ns()
    source_gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
    anchors=[('TL',(180,90,120,130)),('TR',(1200,80,110,130)),
             ('BL',(190,1650,110,130)),('BR',(1200,1650,110,130))]
    found=[];ideal=[];details=[]
    for key,(x,y,w,h) in anchors:
        local=np.asarray([[1.,0.,-x+16],[0.,1.,-y+16],[0.,0.,1.]])@affine33(matrix)
        search=cv2.warpAffine(source_gray,local[:2],(w+32,h+32))
        patch=reference_gray[y:y+h,x:x+w]
        response=cv2.matchTemplate(search,patch,cv2.TM_CCOEFF_NORMED)
        _,score,_,pos=cv2.minMaxLoc(response)
        dx,dy=pos[0]-16,pos[1]-16
        details.append(dict(anchor=key,dx=dx,dy=dy,correlation=score))
        if score>=.75:
            found.append((x+w/2+dx,y+h/2+dy))
            ideal.append((x+w/2,y+h/2))
    if len(found)<3:raise RuntimeError(f'Independent frame alignment failed: {details}')
    correction,_=cv2.estimateAffinePartial2D(np.asarray(found),np.asarray(ideal),method=cv2.LMEDS)
    final=(affine33(correction)@affine33(matrix))[:2]
    return final,dict(anchors=details,coarse_to_reference=matrix.tolist(),
                      correction=correction.tolist(),final_transform=final.tolist(),
                      core_registration_ms=(time.perf_counter_ns()-started)/1e6)


def registered_geometry(shape,dx=0,dy=0):
    geo=ORIGINAL_GEOMETRY(shape,dx,dy)
    if 'body_matrix' in shape:
        geo['matrix']=geo['matrix']@affine33(np.asarray(shape['body_matrix']))
        geo['angle']=shape['body_angle']
    return geo


def registered_inventory(product):
    rows,images,missing=ORIGINAL_INVENTORY(product)
    reference_name='cam1_debug_cam_20260805_151316_234965.png'
    reference=images[reference_name]
    reference_path=next(product.rglob(reference_name))
    reference_shapes={s['label']:s for s in trial.read_json(reference_path.with_suffix('.json'))['shapes']}
    reference_gray=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
    transforms={}
    for name,raw in list(images.items()):
        t0=time.perf_counter_ns()
        path=next(product.rglob(name))
        shapes={s['label']:s for s in trial.read_json(path.with_suffix('.json'))['shapes']}
        src=[];dst=[]
        for label in sorted(reference_shapes.keys() & shapes.keys()):
            a=reference_shapes[label]['points'];b=shapes[label]['points']
            if len(a)==4 and len(b)==4:
                src.extend(b);dst.extend(a)
        matrix,_=cv2.estimateAffinePartial2D(np.asarray(src),np.asarray(dst),method=cv2.LMEDS)
        final,info=register_one(raw,reference_gray,matrix)
        transforms[name]=final
        REGISTRATION.append(dict(file=name,**info,total_ms=(time.perf_counter_ns()-t0)/1e6))
    for r in rows:
        angle=r['geo']['angle']
        r['shape']=dict(reference_shapes[r['label']],body_matrix=transforms[r['file']].tolist(),body_angle=angle)
        r['geo']=registered_geometry(r['shape'])
        r['patch']=trial.patch_from_scene(images[r['file']],r['geo'])
    return rows,images,missing


def already_registered(gray,reference,stable):
    h,w=reference.shape
    return gray[trial.PAD:trial.PAD+h,trial.PAD:trial.PAD+w],dict(dx=0,dy=0,correlation=None)


def normalize_percentiles(gray,reference,stable):
    a=gray[stable>0].astype(np.float32)
    b=reference[stable>0].astype(np.float32)
    aq=np.percentile(a,[10,90]);bq=np.percentile(b,[10,90])
    gain=float(np.clip((bq[1]-bq[0])/max(aq[1]-aq[0],1.),.65,1.5))
    offset=float(np.median(b)-gain*np.median(a))
    return cv2.addWeighted(gray,gain,gray,0.,offset),dict(gain=gain,offset=offset)


if __name__=='__main__':
    if '--output' not in sys.argv:
        sys.argv.extend(['--output',str(trial.HERE/'appearance_body_registered_trial')])
    trial.inventory=registered_inventory
    trial.geometry=registered_geometry
    trial.align=already_registered
    trial.normalize=normalize_percentiles
    trial.main()
    output=Path(sys.argv[sys.argv.index('--output')+1])
    import json
    (output/'registration.json').write_text(json.dumps(REGISTRATION,ensure_ascii=False,indent=2),encoding='utf-8')
