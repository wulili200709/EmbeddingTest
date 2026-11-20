from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'lab_body_edge_short_20261015'
d=json.loads((OLD/'measurements.json').read_text(encoding='utf-8'))
im=next(i for i in d['images'] if i['name']=='2.png')
bgr=cv2.imdecode(np.fromfile(OLD/'originals/2.png',np.uint8),cv2.IMREAD_COLOR)
gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
blur1=cv2.GaussianBlur(gray,(7,1),0)
blur3=cv2.GaussianBlur(gray,(7,3),0)
blur9=cv2.GaussianBlur(gray,(7,9),0)
results=[]
for name in ['U2','U4','U5']:
 p=next(p for p in im['points'] if p['name']==name);cx=p['region']['center_x']
 e=p['auto']['upper'];fy=lambda x:float(np.polyval(e['coefficients'],x))
 fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
 fig.suptitle(f'2.png {name} upper: actual pixels, no laboratory dimensions')
 crops=[]
 for col,x in enumerate([cx-38,cx,cx+38]):
  j=x-(cx-50);rec=e['background_diagnostics'][j];onset=e['points'][j][1];seed=rec['seed_y']
  yy=np.arange(int(seed)-8,int(onset)+12)
  for profile,label,c in [(blur1,'horizontal smoothing only','0.5'),(blur3,'vertical k3','blue'),(blur9,'vertical k9','green')]:
   axs[0,col].plot(yy,profile[yy,x],label=label,color=c)
   axs[1,col].plot(yy,np.gradient(profile[:,x])[yy],label=label,color=c)
  for row in range(2):
   axs[row,col].axvline(onset,color='red',label='current first departure')
   axs[row,col].axvline(fy(x),color='cyan',linestyle='--',label='current fitted line')
   axs[row,col].axvline(seed,color='orange',linestyle=':',label='bright-face seed')
   axs[row,col].set_xlabel('Original image y (downward)');axs[row,col].grid(alpha=.2)
  axs[0,col].set_title(f'x={x}; onset={onset:.2f}; line={fy(x):.2f}')
  axs[0,col].set_ylabel('Gray level');axs[1,col].set_ylabel('Gradient dI/dy')
  axs[0,col].legend(fontsize=7)
  vals={'x':x,'seed':seed,'first_departure_y':onset,'fitted_y':fy(x),'profile':[{'y':int(y),'horizontal_only':float(blur1[y,x]),'blur3':float(blur3[y,x]),'blur9':float(blur9[y,x]),'gradient3':float(np.gradient(blur3[:,x])[y])} for y in yy]}
  results.append({'name':name,**vals})
 fig.savefig(OUT/f'{name}_upper_profiles.png',dpi=140);plt.close(fig)
 y0=int(min(v[1] for v in e['segment']))-35;y1=y0+65;x0=cx-75;x1=cx+76
 crop=bgr[y0:y1,x0:x1].copy();big=cv2.resize(crop,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)
 for x in [cx-38,cx,cx+38]:
  xx=(x-x0)*6;cv2.line(big,(xx,0),(xx,big.shape[0]-1),(255,0,0),1)
 for a,b in zip(e['segment'][:-1],e['segment'][1:]):
  cv2.line(big,(round((a[0]-x0)*6),round((a[1]-y0)*6)),(round((b[0]-x0)*6),round((b[1]-y0)*6)),(255,255,0),1)
 cv2.imencode('.png',big)[1].tofile(OUT/f'{name}_upper_pixel_crop.png')
(OUT/'upper_profiles.json').write_text(json.dumps(results,indent=2),encoding='utf8')
windows={'U2':(1615,1619,1),'U4':(1617,1620,1),'U5':(1620,1624,-1)}
evidence=[]
for row in results:
 lo,hi,sign=windows[row['name']]
 candidates=[z for z in row['profile'] if lo<=z['y']<=hi]
 peak=max(candidates,key=lambda z:sign*z['gradient3'])
 evidence.append({'name':row['name'],'x':row['x'], 'old_line_y':row['fitted_y'],
                  'visible_local_transition_integer_peak_y':peak['y'],
                  'gradient3_at_peak':peak['gradient3'],
                  'old_line_minus_peak_px':row['fitted_y']-peak['y']})
(OUT/'visible_transition_evidence.json').write_text(json.dumps({'notice':'Image-only audit at nine representative columns. Peak windows were visually identified from original pixel profiles for evidence, not a production detector or manual truth annotation. No laboratory dimensions are used.', 'points':evidence},indent=2),encoding='utf8')
print(json.dumps(evidence,indent=2))
print(OUT)
