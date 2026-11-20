"""Reproduce saved, pre-fix gray/background profiles without laboratory data."""
from pathlib import Path
import json
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
REPORT=HERE.parent
cases=json.loads((HERE/'before_background_fix_cases.json').read_text(encoding='utf-8'))
profiles=[]
for case in cases:
    if case['point']['name']!='L5':
        continue
    raw=cv2.imdecode(np.fromfile(REPORT/'originals'/case['filename'],dtype=np.uint8),0)
    gray=cv2.GaussianBlur(raw.astype(np.float32),(7,9),0)
    region=case['point']['region'];edge=case['point']['auto']['lower']
    lo,hi=region['lower_y']
    for dx in [-55,0,55]:
        x=region['center_x']+dx;p=gray[:,x];record=edge['background_diagnostics'][dx+70]
        a,b=record['background_coefficients']
        profiles.append({'image':case['filename'],'x':x,
                         'chosen':edge['points'][dx+70][1],'background':record,
                         'values_y_gray_gradient_bgResidual':[
                             [y,float(p[y]),float(p[y+1]-p[y]),float(p[y]-(a*y+b))]
                             for y in range(lo-10,hi-10)]})
(HERE/'before_background_fix_profiles.json').write_text(json.dumps(profiles,indent=2),encoding='utf-8')
