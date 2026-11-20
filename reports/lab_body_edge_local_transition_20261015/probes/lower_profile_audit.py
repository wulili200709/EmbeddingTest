from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
OLD=OUT.parents[1]/'lab_body_edge_short_20261015'
d=json.loads((OLD/'measurements.json').read_text(encoding='utf-8'))
im=next(i for i in d['images'] if i['name']=='2.png')
updated=json.loads((OUT/'local_transition_results.json').read_text(encoding='utf8'))
bgr=cv2.imdecode(np.fromfile(OLD/'originals/2.png',np.uint8),cv2.IMREAD_COLOR)
gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
profiles=[(cv2.GaussianBlur(gray,(7,1),0),'horizontal smoothing only','0.5'),
          (cv2.GaussianBlur(gray,(7,3),0),'vertical k3','blue'),
          (cv2.GaussianBlur(gray,(7,9),0),'vertical k9','green')]
results=[]
for name in ['U2','U4','U5']:
 p=next(p for p in im['points'] if p['name']==name);cx=p['region']['center_x']
 e=p['auto']['lower'];n=next(p for p in updated if p['name']==name)['auto']['lower']
 fy=lambda x:float(np.polyval(e['coefficients'],x))
 fn=lambda x:float(np.polyval(n['coefficients'],x))
 fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
 fig.suptitle(f'2.png {name} lower: actual pixels, no laboratory dimensions')
 for col,x in enumerate([cx-38,cx,cx+38]):
  j=x-(cx-50);rec=e['background_diagnostics'][j];onset=e['points'][j][1];seed=rec['seed_y']
  yy=np.arange(int(onset)-10,int(seed)+12)
  for profile,label,c in profiles:
   axs[0,col].plot(yy,profile[yy,x],label=label,color=c)
   axs[1,col].plot(yy,np.gradient(profile[:,x])[yy],label=label,color=c)
  for row in range(2):
   axs[row,col].axvline(onset,color='red',label='old first departure')
   axs[row,col].axvline(fy(x),color='gray',linestyle='--',label='old fitted line')
   axs[row,col].axvline(fn(x),color='cyan',linestyle='--',label='new fitted line')
   axs[row,col].axvline(seed,color='orange',linestyle=':',label='bright-face seed')
   axs[row,col].set_xlabel('Original image y (downward)');axs[row,col].grid(alpha=.2)
  axs[0,col].set_title(f'x={x}; old onset={onset:.2f}; new line={fn(x):.2f}')
  axs[0,col].set_ylabel('Gray level');axs[1,col].set_ylabel('Gradient dI/dy')
  axs[0,col].legend(fontsize=7)
  results.append({'name':name,'x':x,'seed':seed,'first_departure_y':onset,'old_fitted_y':fy(x),'new_fitted_y':fn(x),'new_point_y':n['points'][j][1],
                  'profile':[{'y':int(y),'horizontal_only':float(profiles[0][0][y,x]),'blur3':float(profiles[1][0][y,x]),'blur9':float(profiles[2][0][y,x]),'gradient3':float(np.gradient(profiles[1][0][:,x])[y])} for y in yy]})
 fig.savefig(OUT/f'{name}_lower_profiles.png',dpi=140);plt.close(fig)
 y0=int(min(v[1] for v in e['segment']))-25;y1=y0+65;x0=cx-75;x1=cx+76
 crop=bgr[y0:y1,x0:x1].copy();big=cv2.resize(crop,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)
 for x in [cx-38,cx,cx+38]:
  xx=(x-x0)*6;cv2.line(big,(xx,0),(xx,big.shape[0]-1),(255,0,0),1)
 for ee,color in [(e,(0,140,255)),(n,(255,255,0))]:
  for a,b in zip(ee['segment'][:-1],ee['segment'][1:]):
   cv2.line(big,(round((a[0]-x0)*6),round((a[1]-y0)*6)),(round((b[0]-x0)*6),round((b[1]-y0)*6)),color,1)
 cv2.imencode('.png',big)[1].tofile(OUT/f'{name}_lower_pixel_crop.png')
(OUT/'lower_profiles.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print(OUT)
