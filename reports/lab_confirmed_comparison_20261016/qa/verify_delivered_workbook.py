from pathlib import Path
import json, zipfile, xml.etree.ElementTree as ET, hashlib, math, re

OUT=Path(__file__).resolve().parent
SOURCE=Path('C:/Users/goney/Desktop/DC尺寸数据测试/Lab Test/新建 XLSX 工作表.xlsx')
BOOK=SOURCE.parent/'新版取边与原数据对比_20261016.xlsx'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def readbook(path):
    sheets={}
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in si.findall('.//s:t',NS)) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        for s in ET.fromstring(z.read('xl/workbook.xml')).findall('.//s:sheet',NS):
            target=rels[s.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')]
            target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            cells={}
            for c in ET.fromstring(z.read(target)).findall('.//s:c',NS):
                v=c.find('s:v',NS); val=v.text if v is not None else None; typ=c.get('t'); f=c.find('s:f',NS)
                if typ=='s': val=strings[int(val)]
                elif typ=='inlineStr': val=''.join(t.text or '' for t in c.findall('.//s:t',NS))
                elif val is not None and typ not in ('str','e'): val=float(val)
                cells[c.get('r')]={'value':val,'formula':None if f is None else f.text,'type':typ}
            sheets[s.get('name')]=cells
    return sheets

def inspect():
    sheets=readbook(BOOK)
    text=[]
    for name,cells in sheets.items():
        text.append('SHEET '+name)
        for ref,c in cells.items():
            r=int(re.sub('[A-Z]','',ref))
            if r<25 or ref.startswith('A') or isinstance(c['value'],str):
                if c['value'] is not None or c['formula'] is not None:
                    text.append(f'{ref}: {c["value"]}'+(f' [{c["formula"]}]' if c['formula'] is not None else ''))
    (OUT/'delivered_structure.txt').write_text('\n'.join(text),encoding='utf-8')
    print('\n'.join(text))

def verify():
    sheets=readbook(BOOK)
    independent=json.loads((OUT/'independent_comparison.json').read_text(encoding='utf-8'))
    source=json.loads((OUT/'source_cells.json').read_text(encoding='utf-8'))
    assert list(sheets)==['对比结果','逐点数据','补偿计算']
    errors=[]; checked=0; formula_count=0
    for sn,cells in sheets.items():
        for ref,c in cells.items():
            if c['formula'] is not None: formula_count+=1
            if c['type']=='e' or (isinstance(c['value'],str) and re.fullmatch(r'#(?:REF!|DIV/0!|VALUE!|NAME\?|N/A|NUM!|NULL!|SPILL!|CALC!)',c['value'])):
                errors.append((sn,ref,c))
            if c['formula'] and '#REF!' in c['formula']: errors.append((sn,ref,c))
    def check(sn,ref,wanted):
        nonlocal checked
        actual=sheets[sn].get(ref,{}).get('value'); checked+=1
        if isinstance(wanted,(int,float)):
            ok=isinstance(actual,(int,float)) and math.isclose(actual,wanted,rel_tol=1e-12,abs_tol=1e-12)
        else: ok=actual==wanted
        if not ok: errors.append({'sheet':sn,'cell':ref,'actual':actual,'expected':wanted})
    def formula(sn,ref,wanted):
        actual=sheets[sn][ref]['formula']
        if actual!=wanted: errors.append({'sheet':sn,'cell':ref,'formula':actual,'expected_formula':wanted})
    details={(d['image'],d['name']):d for d in independent['details']}
    names=[f'{p}{i}' for p in ['U','L'] for i in range(1,8)]
    images=[f'{i}.png' for i in range(1,7)]
    scale=independent['summary']['current_scale']
    for i,image in enumerate(images):
        for j,name in enumerate(names):
            r=6+i*14+j; d=details[(image,name)]; paired=d['common']; p=6+j; f=20+i*14+j
            olderr=d['old_px']*scale-d['lab_mm']
            originaloffset=-source['cells'][f'{chr(66+j)}33']['value']
            checks={'A':image,'B':name,'C':int(paired),'D':d['old_px'],'E':d['new_px'],
                    'F':d['new_minus_old_px'],'G':d['lab_mm'],'H':d['old_px']*scale,
                    'I':d['new_direct_mm'],'J':olderr,'K':d['new_direct_error_mm'],
                    'L':olderr+originaloffset,'M':d['new_original_compensation_error_mm'],
                    'N':d['old_same_batch_predicted_mm'],'O':d['new_same_batch_predicted_mm'],
                    'P':d['old_same_batch_error_mm'],'Q':d['new_same_batch_error_mm'],
                    'R':d['old_leave_one_product_out_error_mm'],'S':d['new_leave_one_product_out_error_mm'],
                    'T':d['new_same_batch_predicted_mm']-d['old_same_batch_predicted_mm'] if paired else None,
                    'U':d['old_source_cell'],'V':d['lab_source_cell'],
                    'W':'有数值；用户已确认取边' if paired else '无数值；未参与比较',
                    'AF':d['new_minus_old_mm_fixed_scale']}
            for c,v in checks.items(): check('逐点数据',f'{c}{r}',v)
            for c,k in zip(['X','Y','Z','AA','AB','AC','AD','AE'],['old_direct_error_mm','new_direct_error_mm','old_original_compensation_error_mm','new_original_compensation_error_mm','old_same_batch_error_mm','new_same_batch_error_mm','old_leave_one_product_out_error_mm','new_leave_one_product_out_error_mm']):
                check('逐点数据',f'{c}{r}',abs(d[k]) if d[k] is not None else None)
            formula('逐点数据',f'R{r}',f'IF(C{r}=1,D{r}*\'补偿计算\'!I{f}+\'补偿计算\'!K{f}-G{r},"")')
            formula('逐点数据',f'S{r}',f'IF(C{r}=1,E{r}*\'补偿计算\'!J{f}+\'补偿计算\'!L{f}-G{r},"")')
            for first,k in [(35,'new_minus_old_px'),(45,'old_same_batch_error_mm'),(55,'new_same_batch_error_mm')]:
                check('对比结果',f'{chr(66+j)}{first+i}',d[k])
            check('对比结果',f'{chr(66+j)}{65+i}',checks['T'])
    for r,mode in [(8,'same_scale_direct'),(9,'refit_same_batch'),(10,'refit_loo'),(11,'original_compensation_unchanged')]:
        old=independent['summary']['methods']['old'][mode]; new=independent['summary']['methods']['new'][mode]
        for c,v in zip('DEFGHIJ',[old['mae'],new['mae'],old['rmse'],new['rmse'],old['max_abs'],new['max_abs'],79]):check('对比结果',f'{c}{r}',v)
    for j,n in enumerate(names):
        p=independent['summary']['per_point'][n]; r=17+j
        old=p['old_leave_one_product_out']['mae']; new=p['new_leave_one_product_out']['mae']
        for c,v in zip('ABCDEFG',[n,p['n'],old,new,new-old,p['new_leave_one_product_out']['max_abs'],'减小' if new<old else '增大']):check('对比结果',f'{c}{r}',v)
    val=independent['validation_matched']['methods']
    for i,held in enumerate(['全样本']+images):
        begin=6+i*14; end=begin+13
        for j,n in enumerate(names):
            r=begin+j
            cohort=[d for d in details.values() if d['common'] and d['name']==n and (i==0 or d['image']!=held)]
            oldmean=sum(d['old_px'] for d in cohort)/len(cohort); newmean=sum(d['new_px'] for d in cohort)/len(cohort); labmean=sum(d['lab_mm'] for d in cohort)/len(cohort)
            oldfit=val['legacy']['same_batch'] if i==0 else val['legacy']['leave_one_product_out']['folds'][i-1]
            newfit=val['auto']['same_batch'] if i==0 else val['auto']['leave_one_product_out']['folds'][i-1]
            checks=[held,n,len(cohort),oldmean,newmean,labmean,labmean/oldmean,labmean/newmean,oldfit['scale'],newfit['scale'],oldfit['offsets'][n],newfit['offsets'][n]]
            for c,w in zip('ABCDEFGHIJKL',checks):check('补偿计算',f'{c}{r}',w)
            if i==0: check('补偿计算',f'M{r}',-source['cells'][f'{chr(66+j)}33']['value'])
            else:
                for c in ['C','D','E','F']:
                    actual=sheets['补偿计算'][f'{c}{r}']['formula']
                    if f"'逐点数据'!$A$6:$A$89,\"<>\"&A{r}" not in actual:
                        errors.append({'sheet':'补偿计算','cell':f'{c}{r}','issue':'held-out product filter absent'})
            formula('补偿计算',f'I{r}',f'AVERAGE(G{begin}:G{end})')
            formula('补偿计算',f'J{r}',f'AVERAGE(H{begin}:H{end})')
            formula('补偿计算',f'K{r}',f'F{r}-I{r}*D{r}')
            formula('补偿计算',f'L{r}',f'F{r}-J{r}*E{r}')
    for base in [34,44,54,64]:
        for j,name in enumerate(names): check('对比结果',f'{chr(66+j)}{base}',name)
        for i,image in enumerate(images):check('对比结果',f'A{base+i+1}',image)
    check('对比结果','E4',scale)
    check('对比结果','I4',independent['summary']['methods']['new']['refit_same_batch_scale'])
    check('对比结果','E13',1-independent['summary']['methods']['new']['refit_loo']['mae']/independent['summary']['methods']['old']['refit_loo']['mae'])
    backup=SOURCE.parent/'原数据备份_20261016_134600'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sourcehash=digest(SOURCE); backuphash=digest(backup/SOURCE.name)
    if not sourcehash==backuphash==source['sha256']:errors.append({'issue':'original or backup hash changed','source':sourcehash,'backup':backuphash,'prior':source['sha256']})
    root=OUT.parents[2]
    snapshots=[]
    for dirname,report in [('上一版取边数据','lab_body_edge_local_transition_20261015'),('已确认新版取边快照','lab_body_edge_coherent_20261016')]:
        for filename in ['auto_pixels.csv','legacy_pixels.csv','measurement_comparison.csv','measurements.json']:
            a=backup/dirname/filename; b=root/'reports'/report/filename
            ah,bh=digest(a),digest(b)
            snapshots.append({'file':str(a),'same':ah==bh})
            if ah!=bh: errors.append({'issue':'snapshot mismatch','file':str(a)})
    result={'passed':not errors,'numeric_or_identity_checks':checked,'formula_cells':formula_count,
            'source_workbook_sha256':sourcehash,'original_backup_identical':sourcehash==backuphash,
            'snapshot_file_checks':snapshots,'old_values':84,'lab_values':84,'new_values':79,'new_blanks':5,
            'errors':errors,'scope':'Read-only ZIP/XML; cached values and formula references independently checked, no native Excel recalculation or visual verification performed here.'}
    (OUT/'delivered_workbook_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': verify()
