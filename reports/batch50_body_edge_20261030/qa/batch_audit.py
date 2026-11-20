"""Independent read-only audit of the frozen 50-image measurement export."""
from pathlib import Path
from collections import Counter
import gzip, hashlib, json, math, struct

QA=Path(__file__).resolve().parent
OUT=QA.parent
ROOT=OUT.parents[1]
SOURCE=Path('C:/Users/goney/Desktop/DC尺寸数据测试/1160079 1818098 1561359/50')
ORDER=[f'{r}{i}' for r in 'UL' for i in range(1,8)]
MAPPING={'top':[f'L{i}' for i in range(7,0,-1)],'bottom':[f'U{i}' for i in range(7,0,-1)]}
ALGORITHM_SHA='ed040bfac071b01eaf6e6aa22825e7d139c3c8ea3434dc6605f49d9e4db6a7e0'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def audit():
    errors=[];warnings=[];checks=Counter();max_delta=0.;source_bytes=0
    data=json.loads((OUT/'measurements.json').read_text(encoding='utf-8'))
    reference=json.loads((ROOT/'reports/lab_holdout6_20261030/holdout6_results.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/'config/examples/0916_opening_gaps.json').read_text(encoding='utf-8'))
    expected_params=reference['methods']['new']['parameters']
    params=data['meta']['calibration']
    def ok(condition,where,why):
        checks[why]+=1
        if not condition:errors.append({'where':where,'issue':why})
    def equal(actual,expected,where,why):ok(actual==expected,where,why)
    def close(actual,expected,where,why,tol=1e-8):
        cond=isinstance(actual,(int,float)) and math.isfinite(actual) and abs(actual-expected)<=tol
        ok(cond,where,why)
    def point_close(actual,expected,where,why):
        ok(isinstance(actual,list) and len(actual)==2 and all(isinstance(a,(int,float)) and math.isfinite(a) and abs(a-b)<1e-8 for a,b in zip(actual,expected)),where,why)
    def projected(point,coef):
        x,y=point;a,b=coef;res=y-a*x-b;den=1+a*a
        return abs(res)/math.sqrt(den),[x+a*res/den,y-res/den]
    for key,val in expected_params.items():equal(params.get(key),val,'meta.calibration.'+key,'frozen_five_image_parameter')
    close(params.get('scale_mm_per_px'),0.010547240857655916,'meta.calibration','independently_verified_five_image_scale',1e-15)
    equal(params.get('training_images'),[f'{i}.png' for i in range(1,6)],'meta.calibration','training_images_exclude_sixth')
    equal(params.get('training_pairs'),66,'meta.calibration','training_pair_count')
    equal(data['meta'].get('algorithm_sha256'),ALGORITHM_SHA,'meta','algorithm_hash')
    equal(digest(ROOT/'algorithms/measurement_opening_body_edge.py'),ALGORITHM_SHA,'source algorithm','algorithm_hash')
    equal(data['meta'].get('fit_width'),101,'meta','fit_width')
    equal(data['meta'].get('mapping'),MAPPING,'meta','physical_mapping')
    equal(data['meta'].get('report_order'),ORDER,'meta','table_order')
    equal(data['meta'].get('source_pixels_resampled'),False,'meta','original_pixel_resolution')
    equal(data['meta'].get('source_config'),'0916_opening_gaps.json','meta','roi_config_identity')
    equal(json.loads((OUT/'source_config.json').read_text(encoding='utf-8')),config,'source_config.json','config_unchanged')
    files=sorted(SOURCE.glob('*.png'),key=lambda p:p.name)
    equal(len(files),50,'source','source_image_count')
    equal([i['name'] for i in data['images']],[p.name for p in files],'images','source_image_identity_and_order')
    equal(len(data['images']),50,'images','output_image_count')
    regions={r['name']:r for r in config['regions']};allpoints={};statuses=Counter();numeric=0
    for im in data['images']:
        imagekey=im['name'];path=SOURCE/imagekey
        equal(im['index'],files.index(path)+1,imagekey,'image_index')
        equal(digest(path),im['sha256'],imagekey,'image_sha256')
        source_bytes+=path.stat().st_size
        with path.open('rb') as f:header=f.read(24)
        equal(header[:8],b'\x89PNG\r\n\x1a\n',imagekey,'png_identity')
        dimensions=list(struct.unpack('>II',header[16:24]))
        equal([im['width'],im['height']],dimensions,imagekey,'image_dimensions')
        equal(dimensions,config['image_size'],imagekey,'config_image_dimensions')
        equal([p['name'] for p in im['points']],MAPPING['top']+MAPPING['bottom'],imagekey,'fourteen_physical_points')
        diagpath=OUT/'diagnostics'/f'{im["index"]:02d}.json.gz'
        with gzip.open(diagpath,'rt',encoding='utf-8') as f:diag=json.load(f)
        equal(diag['image'],imagekey,imagekey,'diagnostic_image_identity')
        equal(diag['algorithm_sha256'],ALGORITHM_SHA,imagekey,'diagnostic_algorithm_identity')
        rawpoints={p['name']:p for p in diag['points']}
        for p in im['points']:
            key=imagekey+'/'+p['name'];a=p['auto'];r=p['region'];source_name=p['source_roi_name'];base=regions[source_name]
            allpoints[(imagekey,p['name'])]=p
            expected_row='top' if source_name[0]=='U' else 'bottom';expected_col=int(source_name[1:])
            equal(p['name'],MAPPING[expected_row][expected_col-1],key,'roi_name_to_physical_name')
            equal(p['physical_row'],expected_row,key,'physical_row')
            equal(p['physical_col'],expected_col,key,'physical_column')
            equal(r['name'],p['name'],key,'region_name')
            statuses[a['status']]+=1;px=a['distance_px'];mm=a['distance_mm']
            if px is None:
                equal(mm,None,key,'missing_px_requires_missing_mm')
                equal(a['status'],'failed',key,'missing_px_failed_status')
            else:
                numeric+=1;ok(math.isfinite(px) and px>0,key,'positive_finite_px')
                ok(a['status'] in ('ok','review'),key,'numeric_status')
                close(mm,px*expected_params['scale_mm_per_px']+expected_params['offsets_add_mm'][p['name']],key,'frozen_parameter_mm',1e-12)
            if 'localization_failed' in a['reasons']:
                equal(px,None,key,'failed_localization_no_value');continue
            shift=im['localization'][source_name[0]]
            equal(r['center_x'],base['center_x']+shift['dx'],key,'roi_x_translation')
            for band in ('upper_y','lower_y'):equal(r[band],[v+shift['dy'] for v in base[band]],key,'roi_y_translation')
            cx=r['center_x'];span=[cx-50,cx+50]
            equal(a['span'],span,key,'actual_101_column_span')
            equal(a['fit_width'],101,key,'per_point_fit_width')
            close(a['anchor_abs_dx_range_px'][0],40*100/140,key,'anchor_scale')
            close(a['anchor_abs_dx_range_px'][1],65*100/140,key,'anchor_scale')
            raw=rawpoints[p['name']];ox,oy=raw['coordinate_origin_xy'];rawm=raw['measurement']
            equal(oy,0,key,'preserve_absolute_y_origin')
            equal(raw['crop_size'][1],im['height'],key,'preserve_full_image_height')
            equal(raw['source_region'],r,key,'diagnostic_region')
            equal(rawm['status'],a['status'],key,'raw_compact_status')
            equal(rawm['reasons'],a['reasons'],key,'raw_compact_reasons')
            equal(rawm['distance_px'],px,key,'raw_compact_px')
            equal([x+ox for x in rawm['span']],a['span'],key,'raw_span_translation')
            for side in ('upper','lower'):
                e=a[side];re=rawm[side];where=key+'/'+side
                for mask in ['valid_mask','inlier_mask','anchor_mask','anchor_zone_mask']:
                    equal(len(e[mask]),101,where+'/'+mask,'mask_length')
                    equal(e[mask],re[mask],where+'/'+mask,'raw_compact_mask')
                equal(len(e['points']),101,where,'point_count')
                equal([pt[0] for pt in e['points']],list(range(cx-50,cx+51)),where,'point_full_image_x_coordinates')
                for full,local in zip(e['points'],re['points']):point_close(full,[local[0]+ox,local[1]+oy],where,'point_origin_translation')
                zone=[29<=abs(x-cx)<=46 for x in range(cx-50,cx+51)]
                equal(e['anchor_zone_mask'],zone,where,'actual_anchor_zone')
                equal(e['anchor_mask'],[bool(v and z) for v,z in zip(e['valid_mask'],zone)],where,'valid_anchor_selection')
                equal(e['anchor_x_ranges'],[[cx-46,cx-29],[cx+29,cx+46]],where,'anchor_x_ranges_global')
                equal(e['valid_columns'],sum(e['valid_mask']),where,'valid_count')
                equal(e['left_anchor_support'],sum(e['anchor_mask'][:50]),where,'left_anchor_count')
                equal(e['right_anchor_support'],sum(e['anchor_mask'][51:]),where,'right_anchor_count')
                coef=e['coefficients'];seg=e['segment']
                if coef is None:
                    equal(seg,None,where,'no_fit_no_segment')
                else:
                    close(coef[0],re['coefficients'][0],where,'slope_translation')
                    close(coef[1],re['coefficients'][1]+oy-re['coefficients'][0]*ox,where,'intercept_translation')
                    point_close(seg[0],[span[0],coef[0]*span[0]+coef[1]],where,'global_segment_start')
                    point_close(seg[1],[span[1],coef[0]*span[1]+coef[1]],where,'global_segment_end')
            if all(a[s]['coefficients'] is not None for s in ('upper','lower')):
                uc=a['upper']['coefficients'];lc=a['lower']['coefficients'];um=[cx,uc[0]*cx+uc[1]];lm=[cx,lc[0]*cx+lc[1]]
                computed,foot=projected(lm,uc);reverse,rf=projected(um,lc)
                if px is not None:
                    close(px,computed,key,'lower_midpoint_to_upper_line_px')
                    close(a['reverse_distance_px'],reverse,key,'upper_midpoint_to_lower_line_px')
                    point_close(a['dimension_segment'][0],foot,key,'global_perpendicular_foot')
                    point_close(a['dimension_segment'][1],lm,key,'global_lower_midpoint')
                    point_close(a['reverse_dimension_segment'][0],um,key,'global_upper_midpoint')
                    point_close(a['reverse_dimension_segment'][1],rf,key,'global_reverse_foot')
                    close(a['angle_difference_deg'],abs(math.degrees(math.atan(uc[0])-math.atan(lc[0]))),key,'line_angle')
                    ok(all(lc[0]*x+lc[1]>uc[0]*x+uc[1] for x in span),key,'positive_gap_in_span')
                elif a.get('diagnostic_distance_px') is not None:
                    close(a['diagnostic_distance_px'],computed,key,'diagnostic_failed_geometry')
            for k in ('dimension_segment','reverse_dimension_segment','diagnostic_dimension_segment'):
                if rawm.get(k):
                    for full,local in zip(a[k],rawm[k]):point_close(full,[local[0]+ox,local[1]+oy],key+'/'+k,'dimension_origin_translation')
                else:equal(a.get(k),None,key+'/'+k,'missing_dimension_not_filled')
        ok((OUT/im['detail_image']).exists(),imagekey,'detail_image_exists')
        ok((OUT/'overviews'/f'{im["index"]:02d}.jpg').exists(),imagekey,'overview_exists')
    equal(len(allpoints),700,'total','point_count')
    equal(data['meta']['measurement_count'],700,'meta','declared_point_count')
    equal(data['meta']['numeric_count'],numeric,'meta','declared_numeric_count')
    equal(data['meta']['status_counts'],dict(statuses),'meta','declared_status_counts')
    geometry_path=QA/'geometry_audit_preserve_y.json'
    strict_sample_comparison=geometry_path.exists()
    if not strict_sample_comparison:geometry_path=QA/'geometry_audit.json'
    geometry=json.loads(geometry_path.read_text(encoding='utf-8'))
    prior_count=0;prior_px_delta=0.;prior_changes=[]
    for im in geometry['results']:
        outim=next(i for i in data['images'] if i['name']==im['image'])
        for row in ['U','L']:
            for coord in ['dx','dy']:equal(outim['localization'][row][coord],im['rows'][row][coord],im['image'],'independent_locator_translation')
        for p in im['points']:
            other=allpoints[(im['image'],p['physical_label'])];a=other['auto'];prior_count+=1
            equal(other['region']['center_x'],p['center_x'],im['image']+'/'+p['physical_label'],'independent_roi_center')
            sample_where=im['image']+'/'+p['physical_label']
            matching_px=(a['distance_px'] is None and p['px'] is None) or (a['distance_px'] is not None and p['px'] is not None and abs(a['distance_px']-p['px'])<=1e-8)
            if strict_sample_comparison:
                equal(a['status'],p['status'],sample_where,'independent_sample_status')
                equal(a['reasons'],p['reasons'],sample_where,'independent_sample_reasons')
                ok(matching_px,sample_where,'independent_sample_px')
            if not matching_px or a['status']!=p['status'] or a['reasons']!=p['reasons']:
                prior_changes.append({'where':sample_where,'reference_px':p['px'],'final_px':a['distance_px'],
                                      'reference_status':p['status'],'final_status':a['status'],
                                      'reason_change':a['reasons']!=p['reasons']})
            if a['distance_px'] is not None and p['px'] is not None:prior_px_delta=max(prior_px_delta,abs(a['distance_px']-p['px']))
    equal(prior_count,70,'geometry_audit','independent_sample_count')
    fullpath=QA/'full_vs_crop.json'
    full=json.loads(fullpath.read_text(encoding='utf-8')) if fullpath.exists() else []
    if full:
        equal(len(full),14,'full_vs_crop','full_image_comparison_count')
        first=data['images'][0]['name']
        for p in full:
            a=allpoints[(first,p['name'])]['auto'];equal(a['status'],p['status'],first+'/'+p['name'],'full_image_status')
            if p['full_px'] is None:equal(a['distance_px'],None,first+'/'+p['name'],'full_image_missing_px')
            else:
                close(a['distance_px'],p['full_px'],first+'/'+p['name'],'full_image_px')
                close(p['crop_px'],p['full_px'],first+'/'+p['name'],'full_image_vs_crop_px')
    else:warnings.append('No direct full-image comparison file in final output; independent geometry_audit itself also uses padded crops.')
    if not strict_sample_comparison:warnings.append('Prior 70-point geometry audit translates Y; differences are diagnostic because coarse seed 2px absolute-Y bins are not translation invariant. Not a strict final pixel reference.')
    tailpath=QA/'tail_full_image.json'
    tail_checked=False
    if tailpath.exists():
        tail=json.loads(tailpath.read_text(encoding='utf-8'));p=allpoints[(tail['image'],tail['name'])];a=p['auto'];t=tail['measurement']
        where=tail['image']+'/'+tail['name']
        equal(tail['algorithm_sha256'],ALGORITHM_SHA,where,'independent_tail_algorithm_hash')
        equal(tail['image_sha256'],digest(SOURCE/tail['image']),where,'independent_tail_image_hash')
        equal(a['status'],t['status'],where,'independent_tail_full_status')
        equal(a['reasons'],t['reasons'],where,'independent_tail_full_reasons')
        for k in ['distance_px','reverse_distance_px','diagnostic_distance_px','angle_difference_deg']:
            if t[k] is None:equal(a.get(k),None,where+'/'+k,'independent_tail_full_scalar')
            else:close(a.get(k),t[k],where+'/'+k,'independent_tail_full_scalar')
        equal(a['span'],t['span'],where,'independent_tail_span')
        for side in ['upper','lower']:
            e=a[side];te=t[side]
            for k in ['valid_mask','inlier_mask','anchor_mask','anchor_zone_mask','selection_mode','measurement_failed']:
                equal(e[k],te[k],where+'/'+side+'/'+k,'independent_tail_full_edge')
            if te['coefficients'] is None:equal(e['coefficients'],None,where+'/'+side,'independent_tail_full_coefficients')
            else:
                for actual,expected in zip(e['coefficients'],te['coefficients']):close(actual,expected,where+'/'+side,'independent_tail_full_coefficients')
            for actual,expected in zip(e['points'],te['points']):point_close(actual,expected,where+'/'+side,'independent_tail_full_points')
        for k in ['dimension_segment','reverse_dimension_segment']:
            if t[k] is None:equal(a.get(k),None,where+'/'+k,'independent_tail_full_dimension')
            else:
                for actual,expected in zip(a[k],t[k]):point_close(actual,expected,where+'/'+k,'independent_tail_full_dimension')
        tail_checked=True
    else:warnings.append('Independent direct-full-image tail check not yet available.')
    result={'passed':not errors,'errors':errors,'warnings':warnings,'checks':dict(checks),
            'image_count':len(data['images']),'point_count':len(allpoints),'numeric_count':numeric,
            'status_counts':dict(statuses),'image_sha256_bytes_checked':source_bytes,
            'frozen_scale_mm_per_px':params['scale_mm_per_px'],'training_images':params['training_images'],
            'training_pairs':params['training_pairs'],'independent_sample_points':prior_count,
            'independent_sample_file':str(geometry_path),'strict_independent_sample_comparison':strict_sample_comparison,
            'independent_sample_changes':prior_changes,
            'independent_sample_max_abs_difference_px':prior_px_delta,'full_image_comparison_points':len(full),
            'independent_tail_direct_full_image_checked':tail_checked,
            'measurement_file_sha256':digest(OUT/'measurements.json'),
            'notes':['Read-only audit; no measurement or source file modified.',
                     'geometry_audit.json translates Y; geometry_audit_preserve_y.json, when present, preserves global Y for equivalent seed bins.',
                     'Raw diagnostics use declared crop origin; compact geometry is checked against full-image coordinates.',
                     'Transferred millimetre numbers are checked numerically, not claimed independently calibrated for this batch.']}
    (QA/'batch_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':audit()
