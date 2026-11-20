from pathlib import Path
import json
import cv2
import numpy as np
OUT=Path(__file__).resolve().parent
REPORT=OUT.parent
d=json.loads((REPORT/'measurements.json').read_text(encoding='utf8'))
im=next(im for im in d['images'] if im['name']=='5.png')
p=next(p for p in im['points'] if p['name']=='U4');cx=p['region']['center_x']
bgr=cv2.imdecode(np.fromfile(REPORT/im['url'],np.uint8),cv2.IMREAD_COLOR)
gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
blur=cv2.GaussianBlur(gray,(7,3),0);g=np.diff(blur,axis=0)
rows=[]
for x in list(range(cx-46,cx-28))+list(range(cx+29,cx+47)):
 y=1642+int(np.argmax(g[1642:1648,x]))
 a,b,c=g[y-1:y+2,x];delta=.5*(a-c)/(a-2*b+c) if abs(a-2*b+c)>1e-9 else 0
 rows.append({'x':x,'side':'left' if x<cx else 'right','outer_peak_y':float(y+.5+np.clip(delta,-.5,.5)),
  'positive_gradient_strength':float(g[y,x]),'lip_valley_y':int(1640+np.argmin(blur[1640:1646,x])),
  'peak_polarity':'positive dI/dy (dark lip to opening background)',
  'new_line_y':float(np.polyval(p['auto']['upper']['coefficients'],x))})
summaries={}
for side in ['left','right']:
 rr=[r for r in rows if r['side']==side];v=np.array([r['positive_gradient_strength'] for r in rr])
 summaries[side]={'count':len(rr),'count_ge3':int(np.sum(v>=3)),'count_ge4':int(np.sum(v>=4)),'count_ge5':int(np.sum(v>=5)),
  'min_median_max_strength':list(map(float,[min(v),np.median(v),max(v)])),
  'peak_y_min_max':[min(r['outer_peak_y'] for r in rr),max(r['outer_peak_y'] for r in rr)]}
usable=[r for r in rows if r['positive_gradient_strength']>=3]
fit=np.polyfit([r['x'] for r in usable],[r['outer_peak_y'] for r in usable],1)
summaries['outer_audit_fit']={'center_y':float(np.polyval(fit,cx)),'coefficients':fit.tolist(),
 'support_count':len(usable),'rmse_px':float(np.sqrt(np.mean([(r['outer_peak_y']-np.polyval(fit,r['x']))**2 for r in usable])))}
(OUT/'5_U4_outer_lip_evidence.json').write_text(json.dumps({'notice':'Independent visible-event audit with fixed y window for evidence only; not a detector rule or ground-truth labels.','summary':summaries,'rows':rows},indent=2),encoding='utf8')
print(json.dumps(summaries,indent=2))
