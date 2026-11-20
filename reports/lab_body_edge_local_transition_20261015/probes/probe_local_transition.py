from pathlib import Path
import sys, json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from algorithms.measurement_opening_gap import OpeningGapRegion
from algorithms.measurement_opening_body_edge import measure_opening_body_edges

def main():
    old=ROOT/'reports/lab_body_edge_short_20261015'
    out=Path(__file__).resolve().parent
    data=json.loads((old/'measurements.json').read_text(encoding='utf-8'))
    im=next(i for i in data['images'] if i['name']=='2.png')
    bgr=cv2.imdecode(np.fromfile(old/'originals/2.png',np.uint8),cv2.IMREAD_COLOR)
    results=[]
    panels=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for name in ['U2','U4','U5']:
        p=next(p for p in im['points'] if p['name']==name)
        r=OpeningGapRegion(**p['region'])
        auto=measure_opening_body_edges(bgr,r,fit_width=101)
        results.append({'name':name,'region':p['region'],'auto':auto})
        x0,x1=r.center_x-110,r.center_x+110
        y0,y1=r.upper_y[0]-12,r.lower_y[1]+12
        panel=Image.new('RGB',(440,(y1-y0)*2+60),'#18262d')
        crop=bgr[y0:y1,x0:x1].copy()
        for side in ['upper','lower']:
            edge=auto[side]
            for pt,onset,valid in zip(edge['points'],edge['onset_points'],edge['valid_mask']):
                if valid:
                    cv2.circle(crop,(round(pt[0]-x0),round(pt[1]-y0)),0,(0,255,0),-1)
                    cv2.circle(crop,(round(onset[0]-x0),round(onset[1]-y0)),0,(0,140,255),-1)
            if edge['segment']:
                a,b=edge['segment']
                cv2.line(crop,(round(a[0]-x0),round(a[1]-y0)),(round(b[0]-x0),round(b[1]-y0)),(255,225,20),1)
        panel.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((440,(y1-y0)*2)),(0,60))
        draw=ImageDraw.Draw(panel)
        draw.text((5,3),f'{name} {auto["status"]} cyan: gradient line',font=font,fill='white')
        draw.text((5,28),'green: actual points; orange: old onset',font=font,fill='white')
        panels.append(panel)
        print(name,auto['status'],auto['distance_px'])
        for side in ['upper','lower']:
            e=auto[side]
            print(side,e['coefficients'],e['anchor_rmse_px'],e['quality_flags'])
    canvas=Image.new('RGB',(1320,max(p.height for p in panels)),'#18262d')
    for i,p in enumerate(panels):canvas.paste(p,(i*440,0))
    canvas.save(out/'U2_U4_U5_local_transition.png')
    (out/'local_transition_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':main()
