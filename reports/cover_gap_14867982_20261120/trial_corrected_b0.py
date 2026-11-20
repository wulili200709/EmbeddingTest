"""Offline counterfactual: fix a user-confirmed false OK in existing B0 banks.

No source annotations or production NPZ files are modified. In the strict
variant, both original target images are removed from the negative banks.
"""
import copy
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import analyze as a
import trial_local_appearance as trial

PRODUCT=a.ROOT/'.qr_session/14867982/14867982'
OUT=HERE/'corrected_b0_trial'


def remove_vector(bank,e):
    distances=a.np.linalg.norm(bank-e,axis=1)
    index=int(distances.argmin())
    if distances[index]<.002:return a.np.delete(bank,index,axis=0),dict(index=index,l2=float(distances[index]))
    return bank,None


def main():
    a.torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    rows,_,missing=trial.inventory(PRODUCT)
    runner=a.make_runner()
    features={}
    for name in sorted({r['file'] for r in rows}):
        path=next(PRODUCT.rglob(name))
        e=a.embed_batch(str(path),runner,trial.LABELS,device='cpu')
        for label,vector in zip(trial.LABELS,e):features[(name,label)]=vector
    models={l:a.load_register_model_npz(str(PRODUCT/f'cam1__{l}_register_model_b0.npz')) for l in trial.LABELS}
    variants={}
    changes={}
    for variant in ('original','corrected_labels','original_strict_target_holdout','corrected_strict_target_holdout'):
        selected={l:copy.deepcopy(m) for l,m in models.items()}
        modifications=[]
        for label,model in selected.items():
            if variant.startswith('corrected'):
                for name in trial.NG_OVERRIDES:
                    e=features[(name,label)]
                    model.ok_bank,removed=remove_vector(model.ok_bank,e)
                    if removed is None:raise RuntimeError(f'Mislabeled sample not matched in original OK bank: {name}/{label}')
                    model.ng_bank=a.np.vstack((model.ng_bank,e))
                    modifications.append(dict(label=label,file=name,action='move_OK_vector_to_NG_bank',**removed))
            if variant.endswith('target_holdout'):
                for name in trial.TARGETS:
                    model.ng_bank,removed=remove_vector(model.ng_bank,features[(name,label)])
                    if removed:modifications.append(dict(label=label,file=name,action='remove_target_from_NG_bank',**removed))
            model.ok_proto,model.ng_proto=a.compute_prototypes(model.ok_bank,model.ng_bank)
        variants[variant]=selected
        changes[variant]=modifications
    predictions=[]
    for variant,selected in variants.items():
        for row in rows:
            e=features[(row['file'],row['label'])]
            model=selected[row['label']]
            score=a.classify(e,model)
            record=dict(variant=variant,file=row['file'],label=row['label'],gt=row['gt'],
                        ok_bank_count=len(model.ok_bank),ng_bank_count=len(model.ng_bank),**score)
            if row['gt']=='OK':
                held=copy.deepcopy(model)
                held.ok_bank,removed=remove_vector(held.ok_bank,e)
                if removed:
                    held.ok_proto,held.ng_proto=a.compute_prototypes(held.ok_bank,held.ng_bank)
                    record['good_leave_self_out']=a.classify(e,held)
            predictions.append(record)
            if row['file'] in trial.TARGETS:print('TARGET',json.dumps(record),flush=True)
    summary={}
    for variant in variants:
        group=[r for r in predictions if r['variant']==variant]
        target=[r for r in group if r['file'] in trial.TARGETS]
        good=[r for r in group if r['gt']=='OK']
        summary[variant]=dict(target_ends=len(target),detected_target_ends=sum(r['pred']=='NG' for r in target),
                              available_good_ends=len(good),good_false_alarms=sum(r['pred']=='NG' for r in good),
                              good_leave_self_out_false_alarms=sum(r.get('good_leave_self_out',r)['pred']=='NG' for r in good))
    result=dict(product=str(PRODUCT),scope='Existing embeddings reclassified with corrected labels; no fine tuning or production writes. Strict target holdout excludes both user target PNG vectors from NG banks; remaining samples may share the same physical part.',
                corrections={name:'Both ends NG, confirmed by user' for name in trial.NG_OVERRIDES},
                changes=changes,summary=summary,predictions=predictions,missing=missing)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
