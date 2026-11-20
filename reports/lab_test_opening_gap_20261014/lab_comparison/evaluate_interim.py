"""Read-only compensation and preset sampling-width experiment, not deployment."""
from pathlib import Path
import sys
import json
from dataclasses import replace
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from algorithms.measurement_opening_gap import measure_opening_gap, regions_from_params

def col(n):
    s = ""
    while n:
        n, r = divmod(n-1, 26)
        s = chr(65+r)+s
    return s

def stats(pred, target, names):
    e = pred-target
    at = np.unravel_index(np.argmax(np.abs(e)), e.shape)
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt(np.mean(e**2))),
            "max_abs": float(np.abs(e).max()), "worst": [int(at[0])+1, names[at[1]]],
            "within_003": int(np.sum(np.abs(e)<=.03)),
            "point_max_abs": np.abs(e).max(axis=0).tolist()}

def fit(x, y, model):
    k = float(np.mean(y.mean(axis=0)/x.mean(axis=0)))
    if model == "global_scale":
        return k, 0.
    if model == "point_mean_offset":
        return k, np.mean(y-k*x, axis=0)
    if model == "point_median_offset":
        return k, np.median(y-k*x, axis=0)
    if model == "point_scale":
        return y.mean(axis=0)/x.mean(axis=0), 0.
    raise ValueError(model)

def evaluate(x, y, names):
    output = {}
    for model in ("global_scale", "point_mean_offset", "point_median_offset", "point_scale"):
        k,b=fit(x,y,model)
        held=np.zeros_like(y)
        for i in range(len(y)):
            keep=np.arange(len(y))!=i
            kk,bb=fit(x[keep],y[keep],model)
            held[i]=kk*x[i]+bb
        output[model]={"k":np.asarray(k).tolist(),"b":np.asarray(b).tolist(),
                       "same_sample":stats(k*x+b,y,names),"heldout":stats(held,y,names),
                       "heldout_errors":(held-y).tolist()}
    return output

cells=json.loads((HERE/'workbook_latest_extracted.json').read_text(encoding='utf-8'))['sheets'][0]['cells']
names=[cells[f'{col(c)}1']['value'] for c in range(20,34)]
lab=np.array([[cells[f'{col(c)}{r}']['value'] for c in range(20,34)] for r in range(2,8)])
old=json.loads((HERE/'corrected_calibration_exploration.json').read_text())
assert np.allclose(lab, old['lab1'])
report=HERE.parent
payload=json.loads((report/'measurements.json').read_text(encoding='utf-8'))
config=json.loads((report/'config.json').read_text(encoding='utf-8'))
measurements={f'{m}_{w}':[] for m in ['dark_plateau','peak'] for w in [49,81,101]}
quality={key:[] for key in measurements}
for item in payload['images']:
    gray=cv2.imdecode(np.fromfile(str(report/item['url']),np.uint8),cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    regions=regions_from_params({'regions':item['regions']})
    for key in measurements:
        mode,width=key.rsplit('_',1)
        readings=[]
        qs=[]
        for region in regions:
            rr=measure_opening_gap(gray,replace(region,width=int(width)),edge_mode=mode,
                include_column_edges=False,
                **{k:config[k] for k in ['blur_ksize','min_strength','min_valid_ratio','upper_dark_fraction','lower_dark_fraction']})
            readings.append(rr.distance_px)
            qs.append({'valid':rr.valid_columns,'mad':rr.mad_px})
        measurements[key].append(readings[::-1])
        quality[key].append(qs[::-1])
out={'lab_names':names,'lab':lab.tolist(),'note':'Same six images used to inspect algorithm and select models. Heldout results are exploratory; not independent final validation. All raw images unchanged.', 'experiments':{}}
for key, values in measurements.items():
    x=np.array(values)
    models=evaluate(x,lab,names)
    out['experiments'][key]={'pixels':values,'quality':quality[key],'models':models}
    print(key)
    for m, results in models.items():
        print(m,json.dumps(results['heldout']))
    if key=='dark_plateau_49':
        print('BASELINE COEFFS',json.dumps({m:{'k':r['k'],'b':r['b'],'same_sample':r['same_sample']} for m,r in models.items()}))
(HERE/'interim_compensation_experiments.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
