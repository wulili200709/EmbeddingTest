"""Image-only fit-width probe. Does not edit production algorithms or inputs."""
from pathlib import Path
import sys
import json
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from algorithms.measurement_opening_gap import OpeningGapRegion
from algorithms.measurement_opening_body_edge import measure_opening_body_edges

OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'reports/lab_body_edge_comparison_20261015'
data=json.loads((SOURCE/'measurements.json').read_text(encoding='utf-8'))
item=next(i for i in data['images'] if i['name']=='2.png')
path=SOURCE/item['url']
img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
names=['U2','U4','U5']; widths=[141,101,81]
records=[]
canvas=Image.new('RGB',(1440,1600),'#142029')
draw=ImageDraw.Draw(canvas)
draw.text((12,6),'Same source pixels/crop. Cyan=fit, green=actual side anchors, magenta=central contour; no lab data.',font=small,fill='white')
for c,name in enumerate(names):
    p=next(p for p in item['points'] if p['name']==name)
    region=OpeningGapRegion(**p['region'])
    cx=region.center_x
    x0=cx-115;x1=cx+115
    y0=region.upper_y[0]-20;y1=region.lower_y[1]+20
    for r,width in enumerate(widths):
        auto=measure_opening_body_edges(gray,region,fit_width=width)
        summary={'point':name,'width':width,'status':auto['status'],'distance_px':auto['distance_px'],
                 'angle_difference_deg':auto['angle_difference_deg'],'reasons':auto['reasons']}
        for side in ['upper','lower']:
            edge=auto[side]
            summary[side]={'angle_deg':math.degrees(math.atan(edge['coefficients'][0])) if edge['coefficients'] else None,
                          **{k:edge[k] for k in ['segment','anchor_rmse_px','left_anchor_support','right_anchor_support','middle_max_deviation_px','middle_deviation_max_run','quality_flags']}}
        records.append({'summary':summary,'auto':auto})
        crop=img[y0:y1,x0:x1].copy()
        # Render from the original crop, enlarged exactly 2x; no estimation on resized pixels.
        crop=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        def loc(pt):return (int(round((pt[0]-x0)*2)),int(round((pt[1]-y0)*2)))
        for side in ['upper','lower']:
            edge=auto[side]
            for pt,valid,anchor in zip(edge['points'],edge['valid_mask'],edge['anchor_mask']):
                if valid:cv2.circle(crop,loc(pt),1,(80,255,80) if anchor else (255,60,220),-1)
            if edge['segment']:
                cv2.line(crop,loc(edge['segment'][0]),loc(edge['segment'][1]),(255,225,20),2,cv2.LINE_AA)
                for pt in edge['segment']:cv2.circle(crop,loc(pt),3,(255,225,20),1)
        panel=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
        yy=32+r*520
        canvas.paste(panel,(c*480+10,yy+64))
        draw.text((c*480+10,yy),f'{name} width={width}  {auto["status"]}',font=font,fill='white')
        ua=summary['upper']['angle_deg'];la=summary['lower']['angle_deg']
        draw.text((c*480+10,yy+24),f'upper={ua:.2f} deg lower={la:.2f} deg' if ua is not None and la is not None else 'missing fit',font=small,fill='#9de6dc')
        draw.text((c*480+10,yy+44),f'anchor RMSE: {summary["upper"]["anchor_rmse_px"]:.2f} / {summary["lower"]["anchor_rmse_px"]:.2f}' if summary['upper']['anchor_rmse_px'] is not None and summary['lower']['anchor_rmse_px'] is not None else 'missing anchors',font=small,fill='#c9d9e0')
        panel.save(OUT/f'2_{name}_width_{width}.png')
canvas.save(OUT/'2_U2_U4_U5_width_comparison.png')
(OUT/'width_comparison.json').write_text(json.dumps({'source':str(path),'image_sha256':item['sha256'],'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([r['summary'] for r in records],ensure_ascii=False,indent=2))
