"""Fixed-image ROI translation and prototype diagnostics; no source data writes."""
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze import (
    ROOT, PRODUCT, OUT, TARGETS, LABELS, read_json, write_json, bbox,
    make_runner, infer_crops, classify, load_register_model_npz, torch, np,
    _build_image_roi_context, _resolve_roi_image, Image, ImageDraw, ImageFont,
)


def main():
    torch.set_num_threads(4)
    runner=make_runner()
    models={l:load_register_model_npz(str(PRODUCT/f'cam1__{l}_register_model_b0.npz')) for l in LABELS}
    source=read_json(OUT/'classification.json')
    rows=[]
    offsets=[(0,dy) for dy in (-8,-4,-2,0,2,4,8)]+[(dx,0) for dx in (-8,-4,-2,2,4,8)]
    for stem in TARGETS:
        path=PRODUCT/'debug_capture'/f'{stem}.png'
        context=_build_image_roi_context(str(path),require_shapes=True)
        for label in LABELS:
            x,y,w,h=bbox(context.shape_by_label[label])
            crops=[context.image.crop((x+dx,y+dy,x+w+dx,y+h+dy)) for dx,dy in offsets]
            for (dx,dy),e in zip(offsets,infer_crops(runner,crops)):
                row=dict(file=stem,label=label,dx=dx,dy=dy,**classify(e,models[label]))
                rows.append(row)
            subset=rows[-len(offsets):]
            print('JITTER',stem,label,'OK',sum(r['pred']=='OK' for r in subset),'/',len(subset),
                  'diff_range',min(r['diff'] for r in subset),max(r['diff'] for r in subset),flush=True)
    result={'offsets':offsets,'rows':rows,'prototype_diagnostics':{}}
    for label,m in models.items():
        result['prototype_diagnostics'][label]={
            'cosine_ok_ng_prototypes':float(m.ok_proto.reshape(-1)@m.ng_proto.reshape(-1)),
            'ok_bank_pairwise':(m.ok_bank@m.ok_bank.T).tolist(),
            'ng_bank_pairwise':(m.ng_bank@m.ng_bank.T).tolist(),
        }
    write_json('sensitivity.json',result)
    make_visual(source)
    print('DONE',flush=True)


def make_visual(source):

    # Compare the precise input crops with known-good, labeled examples.
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    panels=[
        ('正常样本 · 180°方向','20260806_100540_662914_cam1','roi33'),
        ('第一张 · 上端缝隙',TARGETS[0],'roi33'),
        ('正常样本 · 0°方向','cam1_debug_cam_20260920_095839_933345','roi34'),
        ('第二张 · 上端缝隙（漏检）',TARGETS[1],'roi34'),
    ]
    canvas=Image.new('RGB',(1260,690),'#f4f5f7')
    draw=ImageDraw.Draw(canvas)
    draw.text((20,15),'正常与缺陷对照：红框为网络实际看到的区域',fill='#111111',font=font)
    for i,(title,stem,label) in enumerate(panels):
        path=PRODUCT/'debug_capture'/f'{stem}.png'
        context=_build_image_roi_context(str(path),require_shapes=True)
        x,y,w,h=bbox(context.shape_by_label[label])
        crop=context.image.crop((x-55,y-25,x+w+55,y+h+25))
        scale=min(580/crop.width,170/crop.height)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)))
        xx=25+(i//2)*630
        yy=130+(i%2)*285
        canvas.paste(crop,(xx,yy))
        draw.rectangle((xx+55*scale,yy+25*scale,xx+(55+w)*scale,yy+(25+h)*scale),outline='#ff3434',width=3)
        draw.text((xx,yy-42),title+' / '+label,fill='#111111',font=font)
        row=next((r for r in source['samples'] if r['file']==path.name and r['label']==label),None)
        score=row or source['targets'][stem]['scores'][label+'_rotation0']
        draw.text((xx,yy+185),f"当前判定 {score['pred']}   正常相似度 {score['sim_ok']:.4f}",fill='#333333',font=small)
        draw.text((xx,yy+213),f"缺陷相似度 {score['sim_ng']:.4f}   差值 {score['diff']:+.4f}",fill='#333333',font=small)
    canvas.save(OUT/'normal_vs_gap_crops.png')


if __name__=='__main__':main()
