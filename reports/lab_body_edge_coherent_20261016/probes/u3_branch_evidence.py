"""Read-only image-space audit. Fixed windows document visible events, not detector rules."""
from pathlib import Path
import json, math
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
OLD=OUT.parents[1]/'lab_body_edge_short_20261015'
d=json.loads((OLD/'measurements.json').read_text(encoding='utf8'))
im=next(i for i in d['images'] if i['name']=='2.png')
p=next(p for p in im['points'] if p['name']=='U3');cx=p['region']['center_x']
bgr=cv2.imdecode(np.fromfile(OLD/'originals/2.png',np.uint8),cv2.IMREAD_COLOR)
gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
blur=cv2.GaussianBlur(gray,(7,3),0)
g=np.diff(blur,axis=0)
rows=[]
for x in range(cx-50,cx+51):
 # Explicit audit windows inferred from visible outer lip / bright face.
 ng=1673+np.argmin(g[1673:1682,x]);pg=1679+np.argmax(g[1679:1687,x])
 a,b,c=-g[ng-1:ng+2,x];delta=.5*(a-c)/(a-2*b+c) if abs(a-2*b+c)>1e-9 else 0.
 yy=ng+.5+float(np.clip(delta,-.5,.5))
 rows.append({'x':x,'side':'left' if cx-46<=x<=cx-29 else 'right' if cx+29<=x<=cx+46 else 'center_or_margin',
  'outer_dark_lip_negative_peak_y':yy,'outer_negative_peak_strength':float(-g[ng,x]),
  'internal_positive_peak_y':float(pg+.5),'internal_positive_peak_strength':float(g[pg,x]),
  'background_texture_max_abs_gradient':float(np.max(np.abs(g[1652:1669,x]))),
  'outer_drop_1673_to_1680':float(blur[1673,x]-np.min(blur[1673:1681,x]))})
summaries={}
for side in ['left','right']:
 rr=[r for r in rows if r['side']==side]
 ss=np.array([r['outer_negative_peak_strength'] for r in rr])
 summaries[side]={'count':len(rr),'negative_peak_min_median_max':list(map(float,[np.min(ss),np.median(ss),np.max(ss)])),
  'count_at_least_5':int(np.count_nonzero(ss>=5)), 'count_at_least_4':int(np.count_nonzero(ss>=4)),
  'count_at_least_3':int(np.count_nonzero(ss>=3)), 'x_and_y_strength':[[r['x'],round(r['outer_dark_lip_negative_peak_y'],3),round(r['outer_negative_peak_strength'],3)] for r in rr]}
selected=[r for r in rows if r['side'] in ['left','right'] and r['outer_negative_peak_strength']>=3]
fit=np.polyfit([r['x'] for r in selected],[r['outer_dark_lip_negative_peak_y'] for r in selected],1)
res=np.array([r['outer_dark_lip_negative_peak_y']-np.polyval(fit,r['x']) for r in selected])
summaries['audit_line']={'coefficients':fit.tolist(),'center_y':float(np.polyval(fit,cx)),
 'angle_deg':math.degrees(math.atan(float(fit[0]))),'rmse_px':float(np.sqrt(np.mean(res**2))),
 'notice':'Evidence fit using manually identified y audit window; not general detector nor ground-truth annotation.'}
fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for col,(side,xs) in enumerate([('left anchors',range(cx-46,cx-28)),('representative columns',[cx-38,cx,cx+38]),('right anchors',range(cx+29,cx+47))]):
 for x in xs:
  yy=np.arange(1650,1700)
  alpha=1 if col==1 else .35
  axs[0,col].plot(yy,blur[yy,x],alpha=alpha,label=f'x={x}')
  axs[1,col].plot(yy+.5,g[yy,x],alpha=alpha)
 for ar in [axs[0,col],axs[1,col]]:
  ar.axvspan(1675,1681,color='cyan',alpha=.08)
  ar.grid(alpha=.2);ar.set_xlabel('Original image y (downward)')
 axs[0,col].set_title(side)
 axs[0,col].set_ylabel('Gray intensity 7x3 blur')
 axs[1,col].set_ylabel('Adjacent gradient (gray / pixel)')
 if col==1:axs[0,col].legend()
fig.savefig(OUT/'U3_lower_anchor_profiles.png',dpi=140)
crop=bgr[1645:1710,cx-80:cx+81];big=cv2.resize(crop,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)
for r in selected:
 cv2.circle(big,(round((r['x']-(cx-80))*6),round((r['outer_dark_lip_negative_peak_y']-1645)*6)),2,(0,255,0),-1)
for x,y in zip([cx-50,cx+50],np.polyval(fit,[cx-50,cx+50])):pass
a=(round(30*6),round((np.polyval(fit,cx-50)-1645)*6));b=(round(130*6),round((np.polyval(fit,cx+50)-1645)*6))
cv2.line(big,a,b,(255,255,0),1)
cv2.imencode('.png',big)[1].tofile(OUT/'U3_lower_outer_lip_evidence.png')
(OUT/'U3_lower_branch_evidence.json').write_text(json.dumps({'notice':'Image-only fixed audit windows; no laboratory dimensions or new detector tuning.', 'summary':summaries,'rows':rows},indent=2),encoding='utf8')
print(json.dumps(summaries,indent=2))
