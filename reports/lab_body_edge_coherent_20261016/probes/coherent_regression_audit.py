"""Independent, image-only review of the first coherent-branch prototype."""
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
OLD=OUT.parents[1]/'lab_body_edge_local_transition_20261015'
d=json.loads((OLD/'measurements.json').read_text(encoding='utf8'))
im=next(i for i in d['images'] if i['name']=='2.png')
new=json.loads((OUT/'first_coherent.json').read_text(encoding='utf8'))
bgr=cv2.imdecode(np.fromfile(OLD/'originals/2.png',np.uint8),cv2.IMREAD_COLOR)
gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
blur=cv2.GaussianBlur(gray,(7,3),0);g=np.diff(blur,axis=0)
rows=[]
for name,side in [('U3','upper'),('U5','upper'),('U2','lower')]:
 old=next(p for p in im['points'] if p['name']==name)
 latest=next(p for p in new if p['name']==name)
 cx=old['region']['center_x'];oldedge=old['auto'][side];newedge=latest['auto'][side]
 oldfit=oldedge['coefficients'];newfit=newedge['coefficients']
 fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
 fig.suptitle(f'2.png {name} {side}: old local-transition versus new coherent prototype')
 yy=np.arange(int(min(np.polyval(oldfit,cx),np.polyval(newfit,cx)))-15,int(max(np.polyval(oldfit,cx),np.polyval(newfit,cx)))+16)
 for col,x in enumerate([cx-38,cx,cx+38]):
  oldy=float(np.polyval(oldfit,x));newy=float(np.polyval(newfit,x));j=x-(cx-50)
  axs[0,col].plot(yy,gray[yy,x],alpha=.45,color='gray',label='raw single column')
  axs[0,col].plot(yy,blur[yy,x],color='blue',label='7x3 local smoothing')
  axs[1,col].plot(yy+.5,g[yy,x],color='blue')
  for row in range(2):
   axs[row,col].axvline(oldy,color='orange',label=f'previous line {oldy:.2f}')
   axs[row,col].axvline(newy,color='cyan',label=f'new coherent line {newy:.2f}')
   axs[row,col].grid(alpha=.2);axs[row,col].set_xlabel('Original image y (downward)')
  axs[0,col].set_title(f'x={x}, new point={newedge["points"][j][1]:.2f}')
  axs[0,col].legend(fontsize=7);axs[0,col].set_ylabel('Gray intensity');axs[1,col].set_ylabel('Adjacent gradient')
  rr={'name':name,'side':side,'x':x,'old_line_y':oldy,'new_line_y':newy,'old_point_y':oldedge['points'][j][1], 'new_point_y':newedge['points'][j][1],
   'profile':[{'y':int(y),'gray':float(blur[y,x]),'gradient_y_plus_half':float(g[y,x])} for y in yy]}
  rows.append(rr)
 fig.savefig(OUT/f'{name}_{side}_regression_profiles.png',dpi=140);plt.close(fig)
 y0=int(np.polyval(oldfit,cx))-25;y1=y0+55;x0=cx-75;x1=cx+76
 panels=[]
 for mode in ['original','old','new']:
  panel=cv2.resize(bgr[y0:y1,x0:x1],None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
  if mode!='original':
   fit=oldfit if mode=='old' else newfit;color=(0,140,255) if mode=='old' else (255,255,0)
   a=(25*4,round((np.polyval(fit,cx-50)-y0)*4));b=(125*4,round((np.polyval(fit,cx+50)-y0)*4))
   cv2.line(panel,a,b,color,1)
  panel=cv2.copyMakeBorder(panel,25,0,0,0,cv2.BORDER_CONSTANT,value=(25,35,40))
  cv2.putText(panel,f'{name} {side}: {mode}',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
  panels.append(panel)
 cv2.imencode('.png',np.hstack(panels))[1].tofile(OUT/f'{name}_{side}_regression_pixels.png')
(OUT/'coherent_regression_profiles.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
print(OUT)
