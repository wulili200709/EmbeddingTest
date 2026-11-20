from pathlib import Path
from dataclasses import replace
import cv2, numpy as np, json, sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from algorithms.measurement_opening_gap import regions_from_params
from algorithms.measurement_opening_body_edge import measure_opening_body_edges
SOURCE=Path('C:/Users/goney/Desktop/DC尺寸数据测试/1160079 1818098 1561359/50')
OUT=Path(__file__).resolve().parent
paths=sorted(SOURCE.glob('*.png'))
read=lambda p:cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_GRAYSCALE)
reference=read(paths[0])
config=json.loads((ROOT/'config/examples/0916_opening_gaps.json').read_text(encoding='utf-8'))
regions=regions_from_params(config)
templates={'U':(1100,830,3850,1140),'L':(1180,2470,3880,2800)}
results=[]
for index in [0,12,24,37,49]:
    image=read(paths[index]);record={'image':paths[index].name,'index':index+1,'rows':{},'points':[]}
    for row,(x0,y0,x1,y1) in templates.items():
        margin_x,margin_y=180,120
        sx,sy=x0-margin_x,y0-margin_y
        response=cv2.matchTemplate(image[sy:y1+margin_y,sx:x1+margin_x],reference[y0:y1,x0:x1],cv2.TM_CCOEFF_NORMED)
        _,score,_,pos=cv2.minMaxLoc(response)
        dx,dy=pos[0]+sx-x0,pos[1]+sy-y0
        record['rows'][row]={'dx':dx,'dy':dy,'score':score}
    for region in regions:
        delta=record['rows'][region.name[0]];r=replace(region,center_x=region.center_x+delta['dx'],upper_y=tuple(y+delta['dy'] for y in region.upper_y),lower_y=tuple(y+delta['dy'] for y in region.lower_y))
        # Translation of image coordinates only, retain ample filter/background context.
        x0=r.center_x-90;y0=r.upper_y[0]-90;x1=r.center_x+91;y1=r.lower_y[1]+90
        crop=image[y0:y1,x0:x1]
        local=replace(r,center_x=r.center_x-x0,upper_y=tuple(y-y0 for y in r.upper_y),lower_y=tuple(y-y0 for y in r.lower_y))
        m=measure_opening_body_edges(crop,local,fit_width=101,blur_ksize=9,min_strength=5)
        record['points'].append({'original_roi':r.name,'physical_label':('L' if r.name[0]=='U' else 'U')+str(8-int(r.name[1:])), 'center_x':r.center_x,'px':m['distance_px'],'status':m['status'],'reasons':m['reasons']})
    results.append(record)
    print(json.dumps(record,ensure_ascii=False),flush=True)
(OUT/'geometry_audit.json').write_text(json.dumps({'templates':templates,'sample_indices':[1,13,25,38,50],'results':results},ensure_ascii=False,indent=2),encoding='utf-8')
