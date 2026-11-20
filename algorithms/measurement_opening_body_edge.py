"""Candidate measurement of the opening-facing body contour, in image pixels.

The bright-face gradient is only a coarse locator. A persistent background
departure identifies the opening-facing transition and its polarity; the
measured contour is its local gradient peak, not its blurred onset. Side
anchors define each line; central departures are retained as diagnostics and
do not pull that line. If local side observations conflict, a continuous branch
supported by both sides can recover weak contours from all local candidates.
Already-supported contours are not replaced. No laboratory dimensions are used.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from algorithms.measurement_opening_gap import OpeningGapRegion
from algorithms.measurement_opening_lines import _candidate_points, _project


def _robust_fit(x, y):
    center = float(np.mean(x))
    design = np.column_stack((x - center, np.ones(len(x))))
    fit = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(15):
        residual = y - design @ fit
        scale = max(.4, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        weights = np.sqrt(np.minimum(1., 1.345 * scale / np.maximum(np.abs(residual), 1e-9)))
        fit = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0]
    return np.array([fit[0], fit[1] - fit[0] * center])


def _longest(mask):
    longest = run = 0
    for item in mask:
        run = run + 1 if item else 0
        longest = max(longest, run)
    return longest


def _transition_peak(profile, scan, k, polarity, noise, min_strength):
    """Locate the same local transition at subpixel gradient-peak precision.

    The scan runs opening-to-metal. A small vertical filter is used here so a
    narrow dark lip is not merged with the opposite-polarity bright face. Stop
    at the first sign reversal after the first significant gradient lobe; a
    stronger internal bright edge cannot replace the external contour.
    """
    response = np.diff(profile[scan]) * polarity
    first = max(0, k-2)
    last = min(len(response), k+9)
    threshold = max(.5*min_strength, 1.5*noise)
    significant = np.flatnonzero(response[first:last] >= threshold)
    evidence = {'search_y_span':[int(scan[first]), int(scan[last])],
                'gradient_polarity':int(polarity), 'peak_y':None,
                'peak_strength':None, 'event_y_span':None,
                'event_excursion':None, 'status':'no_local_transition'}
    if not len(significant):
        return None, evidence
    begin = first + int(significant[0])
    # Recover the whole positive lobe, including its weak leading samples.
    left = begin
    while left > first and response[left-1] > 0:
        left -= 1
    right = begin
    while right+1 < last and response[right+1] > 0:
        right += 1
    # A dark opening can brighten first into a gray lip and then again into
    # the bright face without reversing polarity. Split these events at a
    # resolved gradient valley and retain the first peak, not the strongest
    # later peak. This is a local image distinction, not an angle constraint.
    peak = left + int(np.argmax(response[left:right+1]))
    for candidate in range(max(left,1),min(right+1,len(response)-1)):
        value=float(response[candidate])
        if value < threshold or value < response[candidate-1] or value <= response[candidate+1]:
            continue
        end=candidate+1
        while end < right and response[end+1] <= response[end]:
            end += 1
        if value-float(response[end]) >= max(.2*value,.5*noise):
            peak=candidate
            right=end
            break
    strength = float(response[peak])
    excursion = float(response[left:right+1].sum())
    evidence.update({'peak_strength':strength, 'event_excursion':excursion,
                     'event_y_span':[int(scan[left]),int(scan[right+1])]})
    if excursion < max(5., 2*noise):
        evidence['status'] = 'insufficient_local_excursion'
        return None,evidence
    if peak == 0 or peak+1 >= len(response) or peak == last-1:
        evidence['status'] = 'gradient_peak_not_bracketed'
        return None,evidence
    a,b,c = map(float,response[peak-1:peak+2])
    denominator = a - 2*b + c
    offset = float(np.clip(.5*(a-c)/denominator,-.5,.5)) if denominator < -1e-9 else 0.
    direction = int(scan[1]-scan[0])
    position = float((scan[peak]+scan[peak+1])*.5 + direction*offset)
    evidence.update({'peak_y':position, 'status':'localized',
                     'subpixel_offset':offset,
                     'gradient_samples':[[float((scan[i]+scan[i+1])*.5),float(response[i])]
                                         for i in range(first,last)]})
    return position,evidence


def _local_candidates(profile, seed, bounds, direction, noise, min_strength):
    """All resolved local intensity transitions near the coarse material face."""
    lo,hi=bounds
    first=int(round(seed-direction*27))
    last=int(round(seed+direction*8))
    scan=np.arange(first,last+direction,direction)
    scan=scan[(scan>=max(0,lo))&(scan<min(len(profile),hi))]
    if len(scan)<5:
        return []
    response=np.diff(profile[scan])
    candidates=[]
    for polarity in (-1,1):
        signed=response*polarity
        threshold=max(.5*min_strength,noise)
        for k in range(1,len(signed)-1):
            strength=float(signed[k])
            if strength<threshold or strength<signed[k-1] or strength<=signed[k+1]:
                continue
            left=k-1
            while left>0 and signed[left-1]>0 and signed[left-1]<=signed[left]:
                left-=1
            right=k+1
            while right+1<len(signed) and signed[right+1]>0 and signed[right+1]<=signed[right]:
                right+=1
            prominence=min(strength-float(signed[left]),strength-float(signed[right]))
            if prominence<max(.2*strength,.5*noise):
                continue
            excursion=float(np.maximum(signed[left:right+1],0).sum())
            if excursion<max(5.,2*noise):
                continue
            a,b,c=map(float,signed[k-1:k+2])
            denominator=a-2*b+c
            offset=float(np.clip(.5*(a-c)/denominator,-.5,.5)) if denominator < -1e-9 else 0.
            y=float((scan[k]+scan[k+1])*.5+direction*offset)
            candidates.append({'y':y,'polarity':polarity,'strength':strength,
                               'prominence':prominence,'excursion':excursion,
                               'event_y_span':[int(scan[left]),int(scan[right+1])]})
    return sorted(candidates,key=lambda item:direction*item['y'])


def _coherent_branch(candidates,xs,cx,left_zone,right_zone,direction,min_strength,
                     side_fraction=.6,total_fraction=.4,strong_fraction=.25):
    """Fit only hypotheses supported by both physical side sections."""
    branches=[]
    dx=xs-cx
    for polarity in (-1,1):
        rows=[[p for p in column if p['polarity']==polarity] for column in candidates]
        max_candidates=max(map(len,rows),default=0)
        if max_candidates==0:
            continue
        ys=np.full((len(xs),max_candidates),np.inf)
        for j,row in enumerate(rows):
            ys[j,:len(row)]=[p['y'] for p in row]
        left=[(int(xs[j]),p['y']) for j in np.flatnonzero(left_zone) for p in rows[j] if p['strength']>=min_strength]
        right=[(int(xs[j]),p['y']) for j in np.flatnonzero(right_zone) for p in rows[j] if p['strength']>=min_strength]
        hypotheses={}
        for xl,yl in left:
            for xr,yr in right:
                slope=(yr-yl)/(xr-xl)
                if abs(slope)>.5:
                    continue
                center=yl+slope*(cx-xl)
                hypotheses[(round(slope/.015),round(center/.5))]=(slope,center)
        scored=[]
        for slope,center in hypotheses.values():
            distance=np.abs(ys-(center+slope*dx)[:,None])
            index=np.argmin(distance,axis=1)
            nearest=distance[np.arange(len(xs)),index]
            supported=nearest<=1.75
            nl=int((supported&left_zone).sum());nr=int((supported&right_zone).sum())
            if min(nl/max(1,left_zone.sum()),nr/max(1,right_zone.sum()))<side_fraction:
                continue
            if supported.sum()<total_fraction*len(xs):
                continue
            anchors=supported&(left_zone|right_zone)
            fitted=_robust_fit(xs[anchors].astype(float),ys[np.arange(len(xs)),index][anchors])
            prediction=np.polyval(fitted,xs)
            distance=np.abs(ys-prediction[:,None])
            index=np.argmin(distance,axis=1)
            nearest=distance[np.arange(len(xs)),index]
            supported=nearest<=1.75
            nl=int((supported&left_zone).sum());nr=int((supported&right_zone).sum())
            if min(nl/max(1,left_zone.sum()),nr/max(1,right_zone.sum()))<side_fraction or supported.sum()<total_fraction*len(xs):
                continue
            anchors=supported&(left_zone|right_zone)
            strong=np.array([bool(supported[j] and rows[j][index[j]]['strength']>=min_strength)
                             for j in range(len(xs))])
            if min((strong&left_zone).sum()/max(1,left_zone.sum()),
                   (strong&right_zone).sum()/max(1,right_zone.sum()))<strong_fraction:
                continue
            strength=float(np.median([rows[j][index[j]]['strength'] for j in np.flatnonzero(anchors)]))
            item={'polarity':polarity,'coefficients':fitted.tolist(),
                  'center_y':float(np.polyval(fitted,cx)), 'support':int(supported.sum()),
                  'left_support':nl,'right_support':nr,
                  'left_strong_support':int((strong&left_zone).sum()),
                  'right_strong_support':int((strong&right_zone).sum()),
                  'anchor_rmse_px':float(np.sqrt(np.mean(nearest[anchors]**2))),
                  'median_anchor_strength':strength}
            scored.append(item)
        # Group hypotheses for the same physical contour, keeping the one
        # with greatest observed support instead of averaging separate edges.
        for item in sorted(scored,key=lambda b:(-(b['left_support']+b['right_support']),-b['support'],b['anchor_rmse_px'])):
            if any(b['polarity']==polarity and abs(b['center_y']-item['center_y'])<2.5
                   and abs(b['coefficients'][0]-item['coefficients'][0])<.06 for b in branches):
                continue
            branches.append(item)
    branches.sort(key=lambda b:direction*b['center_y'])
    return (branches[0] if branches else None),branches


def _trace_branch(candidates,xs,branch,anchor_zone):
    """Trace measured candidates; central protrusions are not replaced by a line."""
    prediction=np.polyval(branch['coefficients'],xs)
    rows=[]
    for j,column in enumerate(candidates):
        row=[p for p in column if p['polarity']==branch['polarity']]
        rows.append(row+[None])
    costs=[];parents=[]
    for j,row in enumerate(rows):
        y=np.array([p['y'] if p else prediction[j] for p in row])
        unary=np.array([1./max(p['strength'],1) if p else 4. for p in row])
        if anchor_zone[j]:
            unary+=np.minimum(np.abs(y-prediction[j]),10.)*2
        if j==0:
            current=unary+np.minimum(np.abs(y-prediction[j]),10.)
            parents.append(np.full(len(row),-1))
        else:
            previous_y=np.array([p['y'] if p else prediction[j-1] for p in rows[j-1]])
            delta=np.abs(y[:,None]-previous_y[None,:])
            transition=delta*.8+np.maximum(delta-3,0)*3
            matrix=costs[-1][None,:]+transition
            predecessor=np.argmin(matrix,axis=1)
            current=unary+matrix[np.arange(len(row)),predecessor]
            parents.append(predecessor)
        costs.append(current)
    selected=[None]*len(rows)
    index=int(np.argmin(costs[-1]))
    for j in range(len(rows)-1,-1,-1):
        selected[j]=rows[j][index]
        index=int(parents[j][index])
    return selected


def _body_edge(gray, region, span, side, blur_ksize, min_strength):
    x0, last = span
    xs = np.arange(x0, last + 1)
    cx = float(region.center_x)
    # Both seed validation and the final contour fit use the same physical
    # sections. Narrowing the measurement must also narrow its seed support.
    scale = min(1., (len(xs)-1)/140.)
    abs_dx = np.abs(xs-cx)
    anchor_bounds = [40*scale, 65*scale]
    anchor_zone = (abs_dx >= anchor_bounds[0]) & (abs_dx <= anchor_bounds[1])
    left_zone, right_zone = anchor_zone & (xs < cx), anchor_zone & (xs > cx)
    anchor_x_ranges = [[int(xs[z][0]), int(xs[z][-1])] if z.any() else None
                       for z in (left_zone, right_zone)]
    min_side = max(5, math.ceil(26*scale*.3))
    lo, hi = map(int, region.upper_y if side == 'upper' else region.lower_y)
    direction = -1 if side == 'upper' else 1  # opening towards metal
    coarse = _candidate_points(gray, x0, last + 1, (lo, hi), side == 'upper',
                               blur_ksize, min_strength)
    branches = coarse.get('branches', [])
    # A seed needs support over both sides of the sampled span. Among these
    # branches choose the opening-facing one, not the strongest backing plate.
    # Small inner branches are retained as evidence, never extrapolated into a
    # full-width seed; the new contour observations decide their significance.
    plausible = [b for b in branches if b['eligible']]
    seed_from_side_support = False
    if not plausible:
        # A seed is a locator, not a measured contour. Verify real candidates
        # on both side sections instead of failing solely because a branch's
        # extreme-point coverage is 0.79 rather than 0.80.
        for branch in branches:
            predicted = branch['center_y'] + branch['slope'] * (xs-cx)
            supported = np.array([any(abs(y-predicted[i]) <= 2.5 for y,_ in peaks)
                                  for i,peaks in enumerate(coarse['candidate_peaks'])])
            left_support = np.count_nonzero(supported & left_zone)
            right_support = np.count_nonzero(supported & right_zone)
            if min(left_support,right_support) >= min_side and supported.sum() >= .45*len(xs):
                plausible.append(branch)
        seed_from_side_support = bool(plausible)
    seed_branch = ((max if side == 'upper' else min)(plausible, key=lambda b:b['center_y'])
                   if plausible else None)
    if seed_branch:
        seeds = seed_branch['center_y'] + seed_branch['slope'] * (xs - cx)
    else:
        seeds = np.array([p[1] for p in coarse['points']], dtype=float)
    # Horizontal smoothing measures local material rather than individual noise
    # speckles. It does not resize, rotate or otherwise resample the image.
    filtered = cv2.GaussianBlur(gray.astype(np.float32), (7, blur_ksize), 0)
    localized = cv2.GaussianBlur(gray.astype(np.float32), (7, min(3,blur_ksize)), 0)
    points, onset_points, valid, records = [], [], [], []
    for j, x in enumerate(xs):
        profile = filtered[:, x]
        seed = int(round(float(seeds[j])))
        # 20--32 pixels is beyond the visibly dark lip (up to ~18 pixels from
        # the bright-face transition in these images); 14--26 can include it.
        onset = None
        edge_position = None
        record = {'seed_y':float(seeds[j]), 'background_coefficients':None,
                  'background_span':None, 'noise':None, 'threshold':None,
                  'departure_polarity':None, 'valid':False,'background_shift_px':0,
                  'raw_transition_strength':None,'onset_rejections':[],
                  'onset_y':None,'transition_localization':None}
        # One image-driven retry moves background away from the metal if a
        # protrusion has already entered the nominal scan. Its evidence remains
        # explicit. The retry cannot cross the opposite edge's search band.
        for shift in (0, 8):
            background_y = seed - direction * np.arange(20+shift, 33+shift)
            scan = np.arange(seed - direction * (19+shift), seed + direction * 8, direction)
            gap_safe = (background_y.max() < region.lower_y[0] if side == 'upper'
                        else background_y.min() > region.upper_y[1])
            safe = (seed_branch is not None and gap_safe and background_y.min() >= 0
                    and background_y.max() < gray.shape[0] and scan.min() >= 0
                    and scan.max() < gray.shape[0])
            if not safe:
                continue
            fit = _robust_fit(background_y.astype(float), profile[background_y])
            residual_bg = profile[background_y] - np.polyval(fit, background_y)
            noise = max(.5, 1.4826 * float(np.median(np.abs(residual_bg))))
            contrast = abs(float(np.percentile(profile[lo:hi],95)) - float(np.polyval(fit,seed)))
            threshold = max(5., 3. * noise, .03 * contrast)
            residual = profile[scan] - np.polyval(fit, scan)
            hits = np.abs(residual) >= threshold
            for k in range(1, len(scan)-2):
                if hits[k:k+3].all() and not hits[k-1] and (
                        np.sign(residual[k:k+3]).min() == np.sign(residual[k:k+3]).max()):
                    first, second = abs(residual[k-1]), abs(residual[k])
                    fraction = float(np.clip((threshold-first)/(second-first),0,1)) if second != first else .5
                    candidate = float(scan[k-1] + direction * fraction)
                    polarity = int(np.sign(residual[k]))
                    # Extrapolation of a sloping background can cross a
                    # residual threshold while the actual image is flat.
                    # Require a localized image transition of the same sign
                    # in this small event window and measure its gradient
                    # peak. Background-model slope is deliberately NOT
                    # subtracted from this image evidence.
                    position,evidence = _transition_peak(localized[:,x],scan,k,polarity,noise,min_strength)
                    if position is None:
                        record['onset_rejections'].append({'y':candidate,
                            'reason':'background_extrapolation_without_local_transition',
                            'transition_localization':evidence})
                        continue
                    onset = candidate
                    edge_position = position
                    record['raw_transition_strength'] = evidence['peak_strength']
                    record['departure_polarity'] = polarity
                    record['onset_y'] = onset
                    record['transition_localization'] = evidence
                    break
            record.update({'background_coefficients':fit.tolist(),
                           'background_span':[int(background_y.min()),int(background_y.max())],
                           'noise':noise,'threshold':threshold,'valid':onset is not None,
                           'background_shift_px':shift})
            if onset is not None:
                break
        points.append([int(x),float(seeds[j]) if edge_position is None else edge_position])
        onset_points.append([int(x),float(seeds[j]) if onset is None else onset])
        valid.append(onset is not None)
        records.append(record)
    initial_points=points
    initial_valid=valid
    initial_anchor=np.asarray(valid,dtype=bool)&anchor_zone
    initial_left=int((initial_anchor&left_zone).sum())
    initial_right=int((initial_anchor&right_zone).sum())
    initial_fit=None
    initial_rmse=None
    recovery_reasons=[]
    if min(initial_left,initial_right)<min_side:
        recovery_reasons.append('initial_insufficient_two_sided_anchors')
    else:
        initial_ys=np.asarray([p[1] for p in points])
        initial_fit=_robust_fit(xs[initial_anchor].astype(float),initial_ys[initial_anchor])
        initial_residual=(initial_ys-np.polyval(initial_fit,xs))/math.hypot(float(initial_fit[0]),1)
        initial_rmse=float(np.sqrt(np.mean(initial_residual[initial_anchor]**2)))
        if initial_rmse>5.:
            recovery_reasons.append('initial_side_anchors_do_not_define_single_body_line')
    local_candidates=[_local_candidates(localized[:,x],seeds[j],(lo,hi),direction,
                                       records[j]['noise'] or .5,min_strength)
                      if seed_branch is not None else [] for j,x in enumerate(xs)]
    branch_proposal,spatial_branches=(_coherent_branch(local_candidates,xs,cx,left_zone,right_zone,direction,min_strength)
                                    if recovery_reasons else (None,[]))
    # Spatial recovery repairs conflicting or missing local selections; it
    # does not replace already-supported physical contours merely because an
    # internal bright-face branch happens to be straighter or stronger.
    selected_branch=branch_proposal if recovery_reasons else None
    unresolved_outer_branches=[]
    if selected_branch is not None:
        proposal_y=np.polyval(selected_branch['coefficients'],xs)
        opening_candidates=[[p for p in column if direction*(p['y']-proposal_y[j]) < -2.5]
                            for j,column in enumerate(local_candidates)]
        # Evidence insufficient to measure an outer contour is still enough
        # to prevent substituting a well-lit internal face. These lower
        # support thresholds are a refusal rule, never measurement eligibility.
        guard_fraction=min_side/max(1,min(int(left_zone.sum()),int(right_zone.sum())))
        _,unresolved_outer_branches=_coherent_branch(opening_candidates,xs,cx,left_zone,right_zone,
                            direction,min_strength,side_fraction=guard_fraction,
                            total_fraction=.2,strong_fraction=guard_fraction)
        if unresolved_outer_branches:
            selected_branch=None
    if selected_branch is not None:
        chosen=_trace_branch(local_candidates,xs,selected_branch,anchor_zone)
        points=[[int(x),float(p['y']) if p is not None else float(seeds[j])]
                for j,(x,p) in enumerate(zip(xs,chosen))]
        valid=[p is not None for p in chosen]
        for record,p in zip(records,chosen):
            record['selected_spatial_candidate']=p
    else:
        chosen=[None]*len(xs)
    recovery_conflict=np.zeros(len(xs),dtype=bool)
    if selected_branch is not None:
        predicted=np.polyval(selected_branch['coefficients'],xs)
        for j,column in enumerate(local_candidates):
            if anchor_zone[j] and chosen[j] is None:
                recovery_conflict[j]=any(p['polarity']==selected_branch['polarity']
                                        and p['strength']>=min_strength
                                        and abs(p['y']-predicted[j])>2.5 for p in column)
    ys = np.asarray([p[1] for p in points])
    valid = np.asarray(valid,dtype=bool)
    # No center observations set the line. These are the same side sections
    # used to establish the coarse seed above.
    anchor = valid & anchor_zone
    left, right = anchor & (xs < cx), anchor & (xs > cx)
    flags = []
    result = {'points':points,'valid_mask':valid.tolist(),'anchor_mask':anchor.tolist(),
              'anchor_zone_mask':anchor_zone.tolist(),'inlier_mask':[False]*len(xs),
              'anchor_scale':scale,'anchor_abs_dx_range_px':anchor_bounds,
              'anchor_x_ranges':anchor_x_ranges,'fit_width':len(xs),'span':list(span),
              'segment':None,'coefficients':None,'rmse_px':None,'all_points_rmse_px':None,
              'anchor_rmse_px':None,'inlier_ratio':0.,'valid_ratio':float(valid.mean()),
              'valid_columns':int(valid.sum()),'quality_flags':flags,
              'background_diagnostics':records,'onset_points':onset_points,
              'initial_local_points':initial_points,'initial_local_valid_mask':initial_valid,
              'local_transition_candidates':local_candidates,
              'spatial_body_branches':spatial_branches,'selected_spatial_body_branch':selected_branch,
              'spatial_branch_proposal':branch_proposal,
              'unresolved_openingward_transition_branches':unresolved_outer_branches,
              'spatial_recovery_blocked_by_outer_contour':bool(unresolved_outer_branches),
              'outer_contour_guard_minimum_strong_support_per_side':min_side,
              'spatial_branch_search_performed':bool(recovery_reasons),
              'selection_mode':'coherent_branch_recovery' if selected_branch else 'local_transition',
              'spatial_recovery_reasons':recovery_reasons,
              'initial_local_anchor_fit':None if initial_fit is None else initial_fit.tolist(),
              'initial_local_anchor_rmse_px':initial_rmse,
              'recovery_unmatched_side_transition_mask':recovery_conflict.tolist(),
              'point_definition':('coherent_local_transition_candidate' if selected_branch
                                  else 'same_local_transition_subpixel_gradient_peak'),
              'localization_filter_ksize':[7,min(3,blur_ksize)],
              'seed_points':[[int(x),float(y)] for x,y in zip(xs,seeds)],
              'seed_branches':branches,'seed_branch':seed_branch,
              'seed_from_two_sided_candidate_support':seed_from_side_support,
              'candidate_peaks':coarse['candidate_peaks'],
              'left_anchor_support':int(left.sum()),'right_anchor_support':int(right.sum()),
              'middle_deviation_mask':[False]*len(xs),'middle_max_deviation_px':None,
              'middle_deviation_max_run':0,'residuals_px':[None]*len(xs),
              'measurement_failed':False}
    if seed_branch is None:
        flags.append('no_coherent_lip_seed')
    if recovery_reasons and branch_proposal is None:
        flags.append('no_two_sided_coherent_body_transition')
        result['measurement_failed'] = True
    if _longest(recovery_conflict)>=4:
        flags.append('spatial_recovery_conflicts_with_observed_side_contour')
        result['measurement_failed'] = True
    if unresolved_outer_branches:
        flags.append('spatial_recovery_behind_unresolved_outer_contour')
        result['measurement_failed'] = True
    if min(int(left.sum()),int(right.sum())) < min_side:
        flags.append('insufficient_two_sided_contour_anchors')
        result['measurement_failed'] = True
        return result
    fit = _robust_fit(xs[anchor].astype(float),ys[anchor])
    # The external body contour must lie openingward of its bright-face seed.
    # Evaluate observed candidates, not the branch's extrapolated center. A
    # slanted branch can join an internal face on one side to a central bump;
    # its extrapolation across empty background is not another physical edge.
    # The backing-plate safeguard remains when an actual openingward branch
    # has the original minimum support and span.
    unresolved_inner=[]
    branch_evidence=[]
    for branch in branches:
        if branch['eligible']:
            continue
        observed=[]
        for j,x in enumerate(xs):
            predicted=branch['center_y']+branch['slope']*(x-cx)
            matches=[float(y) for y,_ in coarse['candidate_peaks'][j] if abs(y-predicted)<=2.5]
            if not matches:
                continue
            y=min(matches,key=lambda value:abs(value-predicted))
            openingward=(y-float(np.polyval(fit,x)))*(1 if side=='upper' else -1)
            if openingward>5:
                observed.append([int(x),y,float(openingward)])
        coverage=(observed[-1][0]-observed[0][0])/max(1,len(xs)-1) if observed else 0.
        unresolved=len(observed)>=.2*len(xs) and coverage>=.3
        branch_evidence.append({'branch':branch,'openingward_observations':observed,
                                'openingward_support':len(observed),
                                'openingward_coverage':float(coverage),'unresolved':unresolved})
        if unresolved:
            unresolved_inner.append(branch)
    result['inner_seed_branch_evidence']=branch_evidence
    result['unresolved_inner_seed_branches'] = unresolved_inner
    if unresolved_inner:
        flags.append('body_contour_behind_unsupported_inner_branch')
        result['measurement_failed'] = True
    residual = (ys - np.polyval(fit,xs)) / math.hypot(float(fit[0]),1)
    inlier_tolerance = 2.5
    inliers = valid & (np.abs(residual) <= inlier_tolerance)
    middle = valid & (abs_dx < 40*scale)
    deviation = middle & (np.abs(residual)>2.5)
    if _longest((~valid) & (abs_dx < 40*scale)) >= 4:
        flags.append('middle_contour_not_observable')
    left_inlier = int((inliers & left).sum());right_inlier = int((inliers & right).sum())
    if min(left_inlier,right_inlier) < min_side:
        flags.append('insufficient_two_sided_inlier_support')
        result['measurement_failed'] = True
    if int(anchor.sum()) < math.ceil(int(anchor_zone.sum())*.8):
        flags.append('partial_side_anchor_support')
    anchor_rmse = float(np.sqrt(np.mean(residual[anchor]**2)))
    all_rmse = float(np.sqrt(np.mean(residual[valid]**2)))
    # A line fitted between mutually incompatible side observations can cross
    # plain background while retaining a few inliers on each side. Such a
    # result is not a body-line measurement. Twice the existing point-inlier
    # tolerance is the explicit maximum RMS inconsistency of its side data.
    rejection_rmse = 2*inlier_tolerance
    result.update({'inlier_tolerance_px':inlier_tolerance,
                   'anchor_rmse_rejection_threshold_px':rejection_rmse})
    if anchor_rmse > rejection_rmse:
        flags.append('side_anchors_do_not_define_single_body_line')
        result['measurement_failed'] = True
    if anchor_rmse > 2.:
        flags.append('side_contour_not_straight_or_ambiguous')
    if _longest(deviation) >= 4:
        flags.append('middle_contour_departs_from_body_line')
    if int((inliers & anchor).sum()) < .85*int(anchor.sum()):
        flags.append('side_contour_outliers')
    if all_rmse > 2.5:
        flags.append('full_contour_variation')
    result.update({'coefficients':fit.tolist(),
                   'segment':[[int(xs[0]),float(np.polyval(fit,xs[0]))],[int(xs[-1]),float(np.polyval(fit,xs[-1]))]],
                   'inlier_mask':inliers.tolist(),
                   'rmse_px':float(np.sqrt(np.mean(residual[inliers]**2))) if inliers.any() else None,
                   'all_points_rmse_px':all_rmse,'anchor_rmse_px':anchor_rmse,
                   'inlier_ratio':float(inliers.sum()/max(1,valid.sum())),
                   'middle_deviation_mask':deviation.tolist(),
                   'middle_max_deviation_px':float(np.max(np.abs(residual[middle]))) if middle.any() else None,
                   'middle_deviation_max_run':_longest(deviation),
                   'residuals_px':[float(r) if v else None for r,v in zip(residual,valid)]})
    return result


def measure_opening_body_edges(image, region: OpeningGapRegion, fit_width=141, blur_ksize=9, min_strength=5.0):
    """Return body-line distance plus complete image-space contour diagnostics.

    ``distance_px`` is lower-line midpoint to upper infinite line; reverse is
    reported separately. Background departure identifies a transition; its
    local gradient peak locates the contour. This image-space candidate is not
    a claim of calibrated physical size.
    """
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError('image must be grayscale or BGR')
    if not np.isfinite(gray).all(): raise ValueError('image must be finite')
    if int(fit_width)!=fit_width or fit_width<21 or fit_width%2!=1:
        raise ValueError('fit_width must be an odd integer >= 21')
    if int(blur_ksize)!=blur_ksize or blur_ksize<1 or blur_ksize%2!=1:
        raise ValueError('blur_ksize must be a positive odd integer')
    if not math.isfinite(min_strength) or min_strength<=0: raise ValueError('invalid min_strength')
    fit_width,blur_ksize=int(fit_width),int(blur_ksize)
    x0=int(region.center_x)-fit_width//2;last=x0+fit_width-1
    ulo,uhi=map(int,region.upper_y);llo,lhi=map(int,region.lower_y)
    if not (0<=x0<=last<gray.shape[1] and 0<=ulo<uhi<llo<lhi<gray.shape[0]):
        raise ValueError('sampling region outside image or overlapping bands')
    span=[x0,last]
    upper=_body_edge(gray,region,span,'upper',blur_ksize,min_strength)
    lower=_body_edge(gray,region,span,'lower',blur_ksize,min_strength)
    reasons=[f'{side}:{flag}' for side,edge in [('upper',upper),('lower',lower)] for flag in edge['quality_flags']]
    result={'name':region.name,'type':'opening_body_edges_candidate','status':'failed','reasons':reasons,
            'upper':upper,'lower':lower,'span':span,'fit_width':fit_width,
            'distance_px':None,'reverse_distance_px':None,'diagnostic_distance_px':None,
            'dimension_segment':None,'reverse_dimension_segment':None,'angle_difference_deg':None,
            'distance_definition':'lower_segment_midpoint_to_upper_infinite_line',
            'contour_definition':'local_transition_peak_with_two_sided_spatial_recovery',
            'anchor_definition':('left_and_right_%g_to_%g_pixels_from_center' %
                                 tuple(upper['anchor_abs_dx_range_px'])),
            'anchor_scale':upper['anchor_scale'],
            'anchor_abs_dx_range_px':upper['anchor_abs_dx_range_px'],
            'anchor_x_ranges':upper['anchor_x_ranges']}
    if upper['coefficients'] is None or lower['coefficients'] is None:return result
    au,bu=upper['coefficients'];al,bl=lower['coefficients'];cx=float(region.center_x)
    if any(al*x+bl<=au*x+bu for x in span):
        reasons.append('lines_cross_or_gap_nonpositive_within_span');return result
    um=[cx,au*cx+bu];lm=[cx,al*cx+bl]
    distance,uf=_project(lm,upper['coefficients']);reverse,lf=_project(um,lower['coefficients'])
    angle=abs(math.degrees(math.atan(al)-math.atan(au)))
    if angle>.5:reasons.append('upper_lower_lines_not_parallel')
    if any(not x0<=p[0]<=last for p in [uf,lf]):reasons.append('perpendicular_foot_outside_sampled_span')
    result.update({'distance_px':distance,'reverse_distance_px':reverse,'dimension_segment':[uf,lm],
                   'reverse_dimension_segment':[um,lf],'angle_difference_deg':angle,
                   'status':'review' if reasons else 'ok'})
    if upper['measurement_failed'] or lower['measurement_failed']:
        result.update({'status':'failed','diagnostic_distance_px':distance,
                       'diagnostic_reverse_distance_px':reverse,'diagnostic_dimension_segment':[uf,lm],
                       'distance_px':None,'reverse_distance_px':None,
                       'dimension_segment':None,'reverse_dimension_segment':None})
    return result
