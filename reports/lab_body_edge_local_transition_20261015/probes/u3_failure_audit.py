from pathlib import Path
import json,sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT))
from algorithms.measurement_opening_body_edge import measure_opening_body_edges
from algorithms.measurement_opening_gap import OpeningGapRegion
OLD=OUT.parents[1]/'lab_body_edge_short_20261015'
d=json.loads((OLD/'measurements.json').read_text(encoding='utf8'))
im=next(i for i in d['images'] if i['name']=='2.png')
p=next(p for p in im['points'] if p['name']=='U3');cx=p['region']['center_x']
bgr=cv2.imdecode(np.fromfile(OLD/'originals/2.png',np.uint8),cv2.IMREAD_COLOR)
gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
a=measure_opening_body_edges(bgr,OpeningGapRegion(**p['region']),fit_width=101)
e=a['lower']
(OUT/'U3_current_auto.json').write_text(json.dumps(a,indent=2),encoding='utf8')
fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
profiles=[(cv2.GaussianBlur(gray,(7,1),0),'horizontal only','0.5'),(cv2.GaussianBlur(gray,(7,3),0),'vertical k3','blue'),(cv2.GaussianBlur(gray,(7,9),0),'vertical k9','green')]
rows=[]
for c,x in enumerate([cx-38,cx,cx+38]):
 j=x-(cx-50);rec=e['background_diagnostics'][j];yy=np.arange(1660,1705)
 line=float(np.polyval(e['coefficients'],x));point=e['points'][j][1]
 for profile,label,color in profiles:
  axs[0,c].plot(yy,profile[yy,x],label=label,color=color)
  axs[1,c].plot(yy,np.gradient(profile[:,x])[yy],label=label,color=color)
 for r in [0,1]:
  axs[r,c].axvline(line,color='cyan',label='current fitted line')
  axs[r,c].axvline(point,color='red',label='selected point')
  axs[r,c].axvline(rec['seed_y'],color='orange',linestyle='--',label='seed')
  axs[r,c].grid(alpha=.2)
 axs[0,c].set_title(f'U3 lower x={x}; line={line:.2f}, point={point:.2f}')
 axs[0,c].legend(fontsize=7)
 rows.append({'x':x,'line':line,'point':point,'record':rec})
fig.savefig(OUT/'U3_lower_profiles.png',dpi=140)
crop=bgr[1645:1730,cx-85:cx+86].copy();big=cv2.resize(crop,None,fx=5,fy=5,interpolation=cv2.INTER_NEAREST)
for ee in [a['upper'],a['lower']]:
 for aa,bb in zip(ee['segment'][:-1],ee['segment'][1:]):
  cv2.line(big,(round((aa[0]-(cx-85))*5),round((aa[1]-1645)*5)),(round((bb[0]-(cx-85))*5),round((bb[1]-1645)*5)),(255,255,0),1)
cv2.imencode('.png',big)[1].tofile(OUT/'U3_lower_pixel_crop.png')
print(json.dumps({'coefficients':e['coefficients'],'reasons':a['reasons'],'rows':rows},indent=2))
