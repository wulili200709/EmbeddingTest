from pathlib import Path
import json
import cv2
import numpy as np
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
OLD=OUT.parents[1]/'lab_body_edge_local_transition_20261015'
old=json.loads((OLD/'measurements.json').read_text(encoding='utf8'))
new=json.loads((NEW/'measurements.json').read_text(encoding='utf8'))
oldindex={(im['name'],p['name']):p for im in old['images'] for p in im['points']}
result=[]
for im in new['images']:
 bgr=cv2.imdecode(np.fromfile(NEW/im['url'],np.uint8),cv2.IMREAD_COLOR)
 gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
 blur=cv2.GaussianBlur(gray,(7,3),0);gradient=np.diff(blur,axis=0)
 for p in im['points']:
  previous=oldindex[(im['name'],p['name'])]
  for side in ['upper','lower']:
   before=previous['auto'][side];after=p['auto'][side]
   if before['coefficients']==after['coefficients']:continue
   cx=p['region']['center_x'];oldfit=before['coefficients'];newfit=after['coefficients']
   oldcy=float(np.polyval(oldfit,cx));newcy=float(np.polyval(newfit,cx))
   key=im['name'].replace('.png','')+'_'+p['name']+'_'+side
   row={'key':key,'image':im['name'],'name':p['name'],'side':side,'status':p['auto']['status'],
    'old_center_y':oldcy,'new_center_y':newcy,'old_coefficients':oldfit,'new_coefficients':newfit,
    'selected_mode':after.get('selected_mode'), 'profiles':[]}
   yy=np.arange(int(min(oldcy,newcy))-16,int(max(oldcy,newcy))+17)
   for x in [cx-38,cx,cx+38]:
    j=x-(cx-50)
    row['profiles'].append({'x':x,'old_line_y':float(np.polyval(oldfit,x)),
     'new_line_y':float(np.polyval(newfit,x)), 'new_point_y':after['points'][j][1],
     'valid':after['valid_mask'][j],'inlier':after['inlier_mask'][j],
     'samples':[{'y':int(y),'gray':float(blur[y,x]),'gradient_y_plus_half':float(gradient[y,x])} for y in yy]})
   y0=int(min(oldcy,newcy))-25;y1=int(max(oldcy,newcy))+31;x0=cx-75;x1=cx+76
   panels=[]
   for mode in ['original','old','new']:
    panel=cv2.resize(bgr[y0:y1,x0:x1],None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
    if mode!='original':
     fit=oldfit if mode=='old' else newfit;color=(0,140,255) if mode=='old' else (255,255,0)
     a=(25*4,round((np.polyval(fit,cx-50)-y0)*4));b=(125*4,round((np.polyval(fit,cx+50)-y0)*4))
     cv2.line(panel,a,b,color,1)
    panel=cv2.copyMakeBorder(panel,25,0,0,0,cv2.BORDER_CONSTANT,value=(25,35,40))
    cv2.putText(panel,f'{key}: {mode}',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    panels.append(panel)
   cv2.imencode('.png',np.hstack(panels))[1].tofile(OUT/f'final_{key}_pixels.png')
   result.append(row)
(OUT/'final_changed_edge_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps([{k:v for k,v in r.items() if k!='profiles'} for r in result],indent=2))
