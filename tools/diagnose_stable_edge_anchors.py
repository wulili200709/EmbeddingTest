"""Independent image-only probe of lip/background contours; no laboratory inputs.

It deliberately does not replace the production candidate. Orange lines are
the previous gradient fit; cyan lines use the gap-facing contour and side
anchors. Every proposed point and anchor mask is saved for inspection.
"""
from pathlib import Path
import json
import math
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/reports/lab_line_comparison_20261015'
OUTPUT = REPORT / 'stable_anchor_experiment'


def robust_fit(x, y):
    center = float(np.mean(x))
    A = np.column_stack((x - center, np.ones(len(x))))
    fit = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(12):
        residual = y - A @ fit
        scale = max(.4, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        w = np.sqrt(np.minimum(1, 1.345 * scale / np.maximum(np.abs(residual), 1e-9)))
        fit = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    return np.array([fit[0], fit[1] - fit[0] * center])


def probe(gray, region, side, half_width=70):
    """Find first departure from a local gap background, in either polarity.

    The ordinary strong gradient only places the background sampling window.
    Background is fit 14--26 pixels gapward of it, and the first persistent
    departure of at least 5 gray levels / 3% contrast / 3 noise scales is used.
    This is an image-space experimental definition, not a calibrated boundary.
    """
    cx = region['center_x']
    lo, hi = region[side + '_y']
    xs = np.arange(cx - half_width, cx + half_width + 1)
    blur = cv2.GaussianBlur(gray[:, xs[0]-3:xs[-1]+4].astype(np.float32), (7, 9), 0)
    points, records = [], []
    direction = -1 if side == 'upper' else 1  # opening -> metal
    for j, x in enumerate(xs):
        p = blur[:, j + 3]
        response = np.diff(p) * direction
        seed = lo + int(np.argmax(response[lo:hi]))
        bg_y = seed - direction * np.arange(14, 27)
        bg_fit = robust_fit(bg_y.astype(float), p[bg_y])
        noise = max(.5, 1.4826 * np.median(np.abs(p[bg_y] - np.polyval(bg_fit, bg_y))))
        bg_at_seed = float(np.polyval(bg_fit, seed))
        contrast = abs(float(np.percentile(p[lo:hi], 95)) - bg_at_seed)
        threshold = max(5., 3 * noise, .03 * contrast)
        scan = np.arange(seed - direction * 13, seed + direction * 8, direction)
        residual = p[scan] - np.polyval(bg_fit, scan)
        hits = np.abs(residual) >= threshold
        onset = None
        for k in range(1, len(scan)-2):
            if hits[k:k+3].all() and not hits[k-1]:
                # Require a consistent sign; textured backgrounds are not edges.
                if np.sign(residual[k:k+3]).min() != np.sign(residual[k:k+3]).max():
                    continue
                a, b = abs(residual[k-1]), abs(residual[k])
                fraction = float(np.clip((threshold-a)/(b-a), 0, 1)) if b != a else .5
                onset = float(scan[k-1] + direction * fraction)
                break
        if onset is None:
            points.append([int(x), None])
        else:
            points.append([int(x), onset])
        records.append({'x':int(x), 'seed_y':int(seed), 'background_coefficients':bg_fit.tolist(),
                        'threshold':float(threshold), 'noise':float(noise), 'onset_y':onset})
    valid = np.array([p[1] is not None for p in points])
    anchor = valid & (np.abs(xs - cx) >= 40) & (np.abs(xs - cx) <= 65)
    ys = np.array([p[1] if p[1] is not None else np.nan for p in points])
    fit = robust_fit(xs[anchor].astype(float), ys[anchor])
    residual = ys - np.polyval(fit, xs)
    return {'points':points, 'anchor_mask':anchor.tolist(), 'coefficients':fit.tolist(),
            'segment':[[int(xs[0]),float(np.polyval(fit,xs[0]))],[int(xs[-1]),float(np.polyval(fit,xs[-1]))]],
            'anchor_rmse_px':float(np.sqrt(np.mean(residual[anchor]**2))),
            'all_valid_rmse_px':float(np.sqrt(np.mean(residual[valid]**2))),
            'records':records}


def main():
    OUTPUT.mkdir(exist_ok=True, parents=True)
    source = json.loads((REPORT / 'measurements.json').read_text(encoding='utf-8'))
    record = next(i for i in source['images'] if i['name'] == '2.png')
    bgr = cv2.imdecode(np.fromfile(REPORT/'originals/2.png',dtype=np.uint8),cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    results = []
    for name in ['U2','U4','U5']:
        point = next(p for p in record['points'] if p['name'] == name)
        region = point['region']
        result = {'name':name,'region':region}
        for side in ['upper','lower']:
            result[side] = probe(gray, region, side)
        a,b = result['upper']['coefficients']; c,d = result['lower']['coefficients']
        cx = region['center_x']
        result['diagnostic_distance_px'] = abs((c-a)*cx+d-b)/math.hypot(a,1)
        result['angle_difference_deg'] = abs(math.degrees(math.atan(c)-math.atan(a)))
        x0,x1 = cx-105,cx+106
        y0,y1 = region['upper_y'][0]-10,region['lower_y'][1]+10
        canvas = cv2.resize(bgr[y0:y1,x0:x1],None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
        def pixel(p): return (round((p[0]-x0)*4),round((p[1]-y0)*4))
        for side in ['upper','lower']:
            previous = point['auto'][side].get('segment')
            if previous:
                cv2.line(canvas,pixel(previous[0]),pixel(previous[1]),(0,140,255),2)
            edge = result[side]
            for p,anchor in zip(edge['points'],edge['anchor_mask']):
                if p[1] is not None:
                    cv2.circle(canvas,pixel(p),2,(0,220,0) if anchor else (230,80,230),-1)
            cv2.line(canvas,pixel(edge['segment'][0]),pixel(edge['segment'][1]),(255,255,0),2)
        cv2.putText(canvas,f'{name} CYAN=background/side anchors ORANGE=previous',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1,cv2.LINE_AA)
        cv2.imencode('.png',canvas)[1].tofile(OUTPUT/f'{name}_contour_probe.png')
        results.append(result)
        print(name, 'distance',round(result['diagnostic_distance_px'],3),'angle',round(result['angle_difference_deg'],3))
        for side in ['upper','lower']:
            e=result[side];print(side,'segment',e['segment'],'anchor_rmse',round(e['anchor_rmse_px'],2),'all_rmse',round(e['all_valid_rmse_px'],2))
    (OUTPUT/'contour_probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__': main()
