"""Cached end-to-end core timing and a compact, candid result contact sheet."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

import trial_registered_appearance as registration

trial=registration.trial
OUT=trial.HERE/'copied_order_final'


def main():
    cv2.setNumThreads(1)
    trial.align=registration.already_registered
    trial.normalize=registration.normalize_percentiles
    report=trial.read_json(OUT/'results.json')
    product=Path(report['product'])
    refs=np.load(OUT/'reference_models.npz',allow_pickle=False)
    models={}
    for key in report['model_counts']:
        label,angle=key.rsplit('_',1)
        model=trial.Appearance.__new__(trial.Appearance)
        model.label=label
        for field in ('reference','tolerance','stable','inspection'):
            setattr(model,field,refs[f'{key}_{field}'])
        models[(label,int(angle))]=model
    refpath=next(product.rglob('cam1_debug_cam_20260805_151316_234965.png'))
    reference=trial.image_read(refpath)
    reference_gray=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
    shapes={s['label']:s for s in trial.read_json(refpath.with_suffix('.json'))['shapes']}
    records={r['file']:r for r in trial.read_json(OUT/'registration.json')}
    names=[*trial.TARGETS,'cam1_debug_cam_20260920_102436_520974.png']
    cases={name:trial.image_read(next(product.rglob(name))) for name in names}
    angles={r['file']:r['angle'] for r in report['predictions']}

    def run(name,details=False):
        t0=time.perf_counter_ns()
        final,reg=registration.register_one(cases[name],reference_gray,np.asarray(records[name]['coarse_to_reference']))
        t1=time.perf_counter_ns()
        rows=[]
        for label in trial.LABELS:
            geo=trial.geometry(shapes[label])
            geo['matrix']=geo['matrix']@registration.affine33(final)
            patch=trial.patch_from_scene(cases[name],geo)
            value=models[(label,angles[name])].score(patch,details=details)
            if details:score,data=value
            else:score,data=value,None
            threshold=report['thresholds'][f'{label}_{angles[name]}']
            rows.append(dict(label=label,pred='NG' if score['score']>threshold else 'OK',
                             threshold=threshold,metrics=score,details=data))
        t2=time.perf_counter_ns()
        return rows,dict(registration_ms=(t1-t0)/1e6,comparison_ms=(t2-t1)/1e6,total_ms=(t2-t0)/1e6)

    for _ in range(5):
        for name in trial.TARGETS:run(name)
    times=[]
    for _ in range(40):
        for name in trial.TARGETS:
            rows,elapsed=run(name)
            assert all(r['pred']=='NG' for r in rows)
            times.append(dict(file=name,**elapsed))
    summary={k:dict(median_ms=float(np.median([r[k] for r in times])),
                     p95_ms=float(np.percentile([r[k] for r in times],95)),
                     max_ms=float(max(r[k] for r in times))) for k in ('registration_ms','comparison_ms','total_ms')}
    result=dict(scope='Both ends, single-thread CPU, cached appearance templates; includes full-frame grayscale conversion, four external frame anchors, ROI extraction, brightness correction, differences and connected components. Excludes existing whole-product shape locator, camera, file IO, one-time model learning and UI.',
                repetitions_per_target=40,summary=summary,runs=times)
    (OUT/'timing.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

    canvas=Image.new('RGB',(1200,760),'#f5f6f8')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    draw.text((20,12),'离线外观对比结果：两张错装检出，另有一张留出良品上端误报',font=font,fill='#152637')
    draw.text((20,46),'红色为超出正常波动的区域；显示统一提亮。错装图未参与良品参考学习。',font=small,fill='#3c4f63')
    for i,name in enumerate(names):
        rows,_=run(name,details=True)
        by_label={r['label']:r for r in rows}
        yy=95+i*215
        gt='用户确认错装 NG' if name in trial.TARGETS else '订单标注良品 OK，独立于本次参考学习'
        draw.text((20,yy),f'{name} / {gt}',font=small,fill='#152637')
        order=('roi33','roi34') if angles[name]==180 else ('roi34','roi33')
        for col,label in enumerate(order):
            row=by_label[label]
            xx=20+col*590
            actual='上端' if col==0 else '下端'
            draw.text((xx,yy+32),f"{actual} / {label} / 判定 {row['pred']} / 暗区 {row['metrics']['score']}，阈值 {row['threshold']}",font=small,fill='#152637')
            data=row['details']
            gray=np.clip(data['corrected'].astype(np.float32)*2.3,0,255).astype(np.uint8)
            over=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            active=data['binary']>0
            over[active]=(.35*over[active]+.65*np.asarray([235,45,50])).astype(np.uint8)
            im=Image.fromarray(over)
            im.thumbnail((560,130))
            canvas.paste(im,(xx,yy+70))
    canvas.save(OUT/'target_and_good_comparison.png')
    print('TIMING',json.dumps(summary),flush=True)
    print('SUMMARY',json.dumps(report['summary']),flush=True)
    print('OUT',str(OUT),flush=True)


if __name__=='__main__':main()
