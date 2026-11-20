"""Read-only source audit and matched-cohort method comparison."""
from pathlib import Path
import json, zipfile, xml.etree.ElementTree as ET, hashlib, sys, math

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('C:/Users/goney/Desktop/DC尺寸数据测试/Lab Test/新建 XLSX 工作表.xlsx')
REPORT = ROOT / 'reports/lab_body_edge_coherent_20261016/measurements.json'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from lab_line_validation import evaluate_methods, POINT_NAMES

def read_cells(path):
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in si.findall('.//s:t',ns)) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        workbook=ET.fromstring(z.read('xl/workbook.xml'))
        sheets=[dict(x.attrib) for x in workbook.findall('.//s:sheet',ns)]
        cells={}
        for c in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:c',ns):
            v=c.find('s:v',ns); val=v.text if v is not None else None
            typ=c.get('t'); f=c.find('s:f',ns)
            if typ=='s': val=strings[int(val)]
            elif typ=='inlineStr': val=''.join(t.text or '' for t in c.findall('.//s:t',ns))
            elif val is not None and typ not in ('str','e'): val=float(val)
            cells[c.get('r')]={'value':val,'formula':None if f is None else f.text,'type':typ}
        return cells,sheets

def col(n):
    s=''
    while n: n,r=divmod(n-1,26); s=chr(65+r)+s
    return s

def metrics(rows, field):
    good=[r for r in rows if r.get(field) is not None]
    errors=[r[field] for r in good]
    worst=max(good,key=lambda r:abs(r[field])) if good else None
    return {'n':len(errors), 'mae':sum(abs(e) for e in errors)/len(errors) if errors else None,
            'rmse':math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None,
            'bias':sum(errors)/len(errors) if errors else None,
            'max_abs':max(map(abs,errors)) if errors else None,
            'worst':{k:worst[k] for k in ('image','name',field)} if worst else None}

def run():
    cells,sheets=read_cells(SOURCE)
    data=json.loads(REPORT.read_text(encoding='utf-8'))
    snapshot={'sheets':sheets,'cells':cells,'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    (OUT/'source_cells.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for r in range(1,70):
        entries=[]
        for n in range(1,40):
            ref=f'{col(n)}{r}'; c=cells.get(ref)
            if c and (c['value'] is not None or c['formula'] is not None):
                if n in [1,2,3,14,15,16,17,18,19,20,21,32,33] or not isinstance(c['value'],(int,float)):
                    entries.append(f'{ref}={c["value"]}' + (f' [{c["formula"]}]' if c['formula'] else ''))
        if entries: rows.append(' | '.join(entries))
    (OUT/'source_structure.txt').write_text('\n'.join(rows),encoding='utf-8')
    v=lambda ref: cells[ref]['value']
    scale=v('P9')
    headercols={v(f'{col(c)}1'):col(c) for c in range(2,16)}
    labcols={v(f'{col(c)}1'):col(c) for c in range(20,34)}
    tablecols={v(f'{col(c)}18'):col(c) for c in range(2,16)}
    source_errors=[]; records=[]; details=[]; missing=[]; rounding=[]
    for im in data['images']:
        row=int(im['name'].split('.')[0])+1
        for p in im['points']:
            name=p['name']; c=headercols[name]; tc=tablecols[name]; lc=labcols[name]
            old=v(f'{c}{row}'); lab=v(f'{lc}{row}'); new=p['auto']['distance_px']
            assert lab==p['lab_mm']
            assert f'{lc}{row}'==p['lab_cell']
            delta=old-p['legacy']['distance_px']
            rounding.append({'image':im['name'],'name':name,'delta_px':delta})
            assert abs(delta)<0.0005001,(im['name'],name,old,p['legacy']['distance_px'])
            # Audit caches against the intended formula chain rather than rely on formula caches alone.
            checks={f'{c}{row+9}':old*scale, f'{tc}{row+17}':old*scale,
                    f'{tc}{row+25}':old*scale-lab,
                    f'{tc}{row+33}':old*scale-v(f'{tc}33'),
                    f'{tc}{row+41}':old*scale-v(f'{tc}33')-lab}
            for ref,calc in checks.items():
                if abs(v(ref)-calc)>1e-12: source_errors.append({'ref':ref,'cached':v(ref),'expected':calc})
            common=new is not None and old is not None and lab is not None
            if not common: missing.append({'image':im['name'],'name':name,'reasons':p['auto']['reasons']})
            records.append({'image':im['name'],'name':name,'legacy_px':old if common else None,
                            'auto_px':new if common else None,'lab_mm':lab,'status':p['auto']['status']})
            details.append({'image':im['name'],'name':name,'common':common,'old_px':old,'new_px':new,
                            'lab_mm':lab,'old_source_cell':f'{c}{row}','lab_source_cell':f'{lc}{row}',
                            'old_direct_mm':old*scale,'new_direct_mm':new*scale if common else None,
                            'new_minus_old_px':new-old if common else None,
                            'new_minus_old_mm_fixed_scale':(new-old)*scale if common else None,
                            'old_direct_error_mm':old*scale-lab if common else None,
                            'new_direct_error_mm':new*scale-lab if common else None,
                            'old_original_compensation_error_mm':old*scale-v(f'{tc}33')-lab if common else None,
                            'new_original_compensation_error_mm':new*scale-v(f'{tc}33')-lab if common else None})
    assert not source_errors,source_errors
    for name,c in headercols.items():
        source_mean=sum(v(f'{c}{r}') for r in range(2,8))/6
        lab_mean=sum(v(f'{labcols[name]}{r}') for r in range(2,8))/6
        assert abs(source_mean-v(f'{c}8'))<1e-12
        assert abs(source_mean*scale-lab_mean-v(f'{tablecols[name]}33'))<1e-12
        assert abs(lab_mean/source_mean-v(f'{c}9'))<1e-12
    assert abs(sum(v(f'{col(c)}9') for c in range(2,16))/14-scale)<1e-12
    evaluated=evaluate_methods(records)
    old=evaluated['methods']['legacy']; new=evaluated['methods']['auto']
    assert old['same_batch']['training_count_by_point']==new['same_batch']['training_count_by_point']
    for f1,f2 in zip(old['leave_one_product_out']['folds'],new['leave_one_product_out']['folds']):
        assert f1['training_images']==f2['training_images']
        assert f1['training_count_by_point']==f2['training_count_by_point']
        assert f1['n']==f2['n']
    for method in ('legacy','auto'):
        prefix='old' if method=='legacy' else 'new'
        for mode in ('same_batch','leave_one_product_out'):
            predictions={(r['image'],r['name']):r for r in evaluated['methods'][method][mode]['predictions']}
            for d in details:
                p=predictions[(d['image'],d['name'])]
                d[f'{prefix}_{mode}_predicted_mm']=p['predicted_mm']
                d[f'{prefix}_{mode}_error_mm']=p['error_mm']
    def slim(m):
        return {k:m[k] for k in ('n','mae','rmse','bias','max_abs','max_abs_at') if k in m}
    summary={'source_workbook':str(SOURCE),'source_sha256':snapshot['sha256'],
             'matches_report_workbook_hash':snapshot['sha256']==data['meta']['workbook_sha256'],
             'current_scale':scale,'old_provisional_Q2':v('Q2'),
             'common_n':sum(d['common'] for d in details),'total_n':len(details),'missing':missing,
             'source_cache_formula_audit_errors':source_errors,
             'legacy_px_matches_original_when_rounded_to_3_decimals':True,
             'max_original_vs_legacy_delta_px':max(abs(r['delta_px']) for r in rounding),
             'methods':{},'per_point':{},'new_minus_old':metrics(details,'new_minus_old_mm_fixed_scale'),
             'notes':['Old pixels use original workbook 3-decimal data, not rerun full-precision legacy.',
                      'Identical 79 available pairs used for training and evaluation of both methods; no exclusion by lab error.',
                      'Report validation previously compares old 84 to new 79; do not quote that as matched-cohort improvement.',
                      'Direct scale and original compensation are workbook parameters estimated from all 84 old points, not independent calibration.',
                      'Same-batch refit and product-held-out refit follow lab_line_validation.py exact ratio-average + point-offset method.',
                      'Whole-product LOO uses five products; all 14 positions of the held-out product stay excluded from training.',
                      'Six development products only; LOO is exploratory, not independent validation.',
                      'Product tolerance +/-0.12 mm is not an acceptance threshold for measurement-system error.']}
    for method,prefix in [('legacy','old'),('auto','new')]:
        m=evaluated['methods'][method]
        summary['methods'][prefix]={
            'same_scale_direct':metrics(details,f'{prefix}_direct_error_mm'),
            'original_compensation_unchanged':metrics(details,f'{prefix}_original_compensation_error_mm'),
            'refit_same_batch':slim(m['same_batch']),
            'refit_same_batch_scale':m['same_batch']['scale'],
            'refit_same_batch_offsets':m['same_batch']['offsets'],
            'refit_loo':slim(m['leave_one_product_out']),
            'loo_scale_range':[min(f['scale'] for f in m['leave_one_product_out']['folds']),max(f['scale'] for f in m['leave_one_product_out']['folds'])]}
    for name in POINT_NAMES:
        pp=[r for r in details if r['name']==name]
        summary['per_point'][name]={'n':sum(p['common'] for p in pp)}
        for mode in ('direct','same_batch','leave_one_product_out'):
            for prefix in ('old','new'):
                summary['per_point'][name][f'{prefix}_{mode}']=metrics(pp,f'{prefix}_{mode}_error_mm')
        summary['per_point'][name]['change']=metrics(pp,'new_minus_old_mm_fixed_scale')
    summary['largest_changes']=sorted([d for d in details if d['common']],key=lambda d:abs(d['new_minus_old_mm_fixed_scale']),reverse=True)[:10]
    (OUT/'independent_comparison.json').write_text(json.dumps({'summary':summary,'details':details,'validation_matched':evaluated},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('per_point','largest_changes')},ensure_ascii=False,indent=2))
    print('PER POINT LOO n oldMAE newMAE oldMAX newMAX')
    for n,s in summary['per_point'].items():
        print(n,s['n'],*[round(s[k][m],6) for k,m in [('old_leave_one_product_out','mae'),('new_leave_one_product_out','mae'),('old_leave_one_product_out','max_abs'),('new_leave_one_product_out','max_abs')]])

if __name__=='__main__': run()
