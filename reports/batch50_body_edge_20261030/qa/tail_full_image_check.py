"""One independent direct-full-image measurement at the end of the new batch."""
from pathlib import Path
from dataclasses import replace
import cv2, numpy as np, json, sys, hashlib, time
QA=Path(__file__).resolve().parent
ROOT=QA.parents[2]
sys.path.insert(0,str(ROOT))
from algorithms.measurement_opening_gap import regions_from_params
from algorithms.measurement_opening_body_edge import measure_opening_body_edges
SOURCE=Path('C:/Users/goney/Desktop/DC尺寸数据测试/1160079 1818098 1561359/50')
paths=sorted(SOURCE.glob('*.png'),key=lambda p:p.name)
config=json.loads((ROOT/'config/examples/0916_opening_gaps.json').read_text(encoding='utf-8'))
cv2.setNumThreads(2)
read=lambda p:cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_GRAYSCALE)
image=read(paths[-1]);reference=read(paths[0]);x0,y0,x1,y1=1180,2470,3880,2800
sx,sy=x0-180,y0-120
response=cv2.matchTemplate(image[sy:y1+120,sx:x1+180],reference[y0:y1,x0:x1],cv2.TM_CCOEFF_NORMED)
_,score,_,pos=cv2.minMaxLoc(response)
dx,dy=pos[0]+sx-x0,pos[1]+sy-y0
base=next(r for r in regions_from_params(config) if r.name=='L2')
region=replace(base,name='U6',center_x=base.center_x+dx,upper_y=tuple(y+dy for y in base.upper_y),lower_y=tuple(y+dy for y in base.lower_y))
started=time.perf_counter()
m=measure_opening_body_edges(image,region,fit_width=101,blur_ksize=9,min_strength=5)
result={'image':paths[-1].name,'index':50,'name':'U6','source_roi_name':'L2','direct_full_image':True,
        'image_sha256':hashlib.sha256(paths[-1].read_bytes()).hexdigest(),
        'algorithm_sha256':hashlib.sha256((ROOT/'algorithms/measurement_opening_body_edge.py').read_bytes()).hexdigest(),
        'localization':{'dx':dx,'dy':dy,'score':score},'duration_seconds':time.perf_counter()-started,
        'measurement':{k:m.get(k) for k in ['status','reasons','distance_px','reverse_distance_px','diagnostic_distance_px','dimension_segment','reverse_dimension_segment','angle_difference_deg','span']}}
for side in ['upper','lower']:
    result['measurement'][side]={k:m[side].get(k) for k in ['coefficients','segment','points','valid_mask','inlier_mask','anchor_mask','anchor_zone_mask','selection_mode','measurement_failed']}
(QA/'tail_full_image.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='measurement'},ensure_ascii=False))
print(json.dumps({k:v for k,v in result['measurement'].items() if k not in ['upper','lower']},ensure_ascii=False))
