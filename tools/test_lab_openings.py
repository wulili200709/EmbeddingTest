"""Run the existing opening-gap demo algorithm on Lab Test, with row tracking.

Only ROIs translate. All edge measurements use the unresampled original image.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.measurement_opening_gap import measure_opening_gap, regions_from_params


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def locate_rows(gray, reference, config):
    result = {}
    mx, my = config["search_margin"]
    for name, (x0, y0, x1, y1) in config["rows"].items():
        sx0, sy0 = max(0, x0 - mx), max(0, y0 - my)
        sx1, sy1 = min(gray.shape[1], x1 + mx), min(gray.shape[0], y1 + my)
        response = cv2.matchTemplate(gray[sy0:sy1, sx0:sx1], reference[y0:y1, x0:x1], cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(response)
        if score < config["min_score"]:
            raise RuntimeError(f"{name}: localization score {score:.4f} below threshold")
        if x in (0, response.shape[1]-1) or y in (0, response.shape[0]-1):
            raise RuntimeError(f"{name}: localization reached search boundary")
        result[name] = {"dx": x + sx0 - x0, "dy": y + sy0 - y0, "score": score}
    return result


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def annotate(image, regions, measurements, labels=True):
    result = image.copy()
    for region in regions:
        m = measurements.get(region.name)
        if m is None:
            continue
        x = m["measurement_x"]
        upper, lower = round(m["upper_y"]), round(m["lower_y"])
        edges = np.asarray(m["column_edges"])
        for column in (1, 2):
            points = np.rint(edges[:, [0, column]]).astype(np.int32)
            cv2.polylines(result, [points], False, (255, 0, 200), 1, cv2.LINE_AA)
        cv2.line(result, (x, upper), (x, lower), (50, 220, 20), 2, cv2.LINE_AA)
        for y in (upper, lower):
            cv2.line(result, (x-12, y), (x+12, y), (255, 0, 200), 2, cv2.LINE_AA)
        if labels:
            text = f"{region.name}: {m['distance_px']:.2f} px"
            pos = (region.center_x - 92, region.upper_y[0] - 12)
            cv2.putText(result, text, pos, cv2.FONT_HERSHEY_SIMPLEX, .57, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(result, text, pos, cv2.FONT_HERSHEY_SIMPLEX, .57, (20, 230, 20), 1, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/examples/lab_test_opening_gaps.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/reports" / f"lab_test_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    source, out = args.input_dir.resolve(), args.output_dir.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = sorted(source.glob(config["filename_glob"]))
    if not paths:
        raise SystemExit("No input images")
    out.mkdir(parents=True, exist_ok=True)
    (out / "originals").mkdir(exist_ok=True)
    (out / "annotated").mkdir(exist_ok=True)
    (out / "details").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    reference = cv2.cvtColor(read_image(source / config["reference_image"]), cv2.COLOR_BGR2GRAY)
    if list(reference.shape[::-1]) != config["image_size"]:
        raise SystemExit("Reference image size does not match config")
    regions = regions_from_params(config)
    names = [r.name for r in regions]
    modes = ("dark_plateau", "peak")
    rows = {mode: [] for mode in modes}
    all_images, errors, details, alignments = [], [], [], []
    for path in paths:
        bgr = read_image(path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if list(gray.shape[::-1]) != config["image_size"]:
            raise ValueError(f"Unexpected image size: {path.name}")
        offsets = locate_rows(gray, reference, config["localization"])
        actual = []
        for region in regions:
            shift = offsets[region.name[0]]
            actual.append(replace(region, center_x=region.center_x + shift["dx"],
                                  upper_y=tuple(y + shift["dy"] for y in region.upper_y),
                                  lower_y=tuple(y + shift["dy"] for y in region.lower_y)))
        for group, shift in offsets.items():
            alignments.append({"image": path.name, "row": group, **shift})
        copied = out / "originals" / path.name
        shutil.copy2(path, copied)
        item = {"name": path.name, "url": f"originals/{path.name}", "modes": {},
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "localization": offsets,
                "regions": [{**asdict(r), "upper_dark_fraction": config["upper_dark_fraction"],
                             "lower_dark_fraction": config["lower_dark_fraction"]} for r in actual]}
        for mode in modes:
            row = {"image": path.name}
            item["modes"][mode] = {}
            for region in actual:
                try:
                    result = measure_opening_gap(gray, region, edge_mode=mode, include_column_edges=True,
                                                 **{key: config[key] for key in ("blur_ksize", "min_strength", "min_valid_ratio", "upper_dark_fraction", "lower_dark_fraction")})
                    m = asdict(result)
                    item["modes"][mode][region.name] = m
                    row[region.name] = round(result.distance_px, 3)
                    details.append({"image": path.name, "mode": mode,
                                    **{k: v for k, v in m.items() if k != "column_edges"}})
                except (ValueError, RuntimeError) as exc:
                    errors.append({"image": path.name, "mode": mode, "region": region.name, "error": str(exc)})
            rows[mode].append(row)
        annotated = annotate(bgr, actual, item["modes"]["dark_plateau"])
        cv2.imencode(".png", annotated)[1].tofile(str(out / "annotated" / path.name))
        # Contact panels retain original pixel scale and show all 14 edge pairs.
        panels = Image.new("RGB", (7*280, 2*255), "#182028")
        edge_only = annotate(bgr, actual, item["modes"]["dark_plateau"], labels=False)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        for index, region in enumerate(actual):
            crop = edge_only[max(0, region.upper_y[0]-15):region.lower_y[1]+20,
                             max(0, region.center_x-135):region.center_x+135]
            panel = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            px, py = (index % 7)*280+5, (index // 7)*255+5
            panels.paste(panel, (px, py+28))
            measurement = item["modes"]["dark_plateau"].get(region.name)
            label = f"{region.name}: {measurement['distance_px']:.3f} px" if measurement else f"{region.name}: FAILED"
            ImageDraw.Draw(panels).text((px+8, py+2), label, font=font, fill="#94edc9")
        panels.save(out / "details" / path.name)
        all_images.append(item)
        print(f"{path.name}: {len(item['modes']['dark_plateau'])}/{len(regions)} gaps; shifts {offsets}", flush=True)
    stats = {}
    for mode in modes:
        write_csv(out / f"lab_test_{mode}.csv", rows[mode], ["image", *names])
        stats[mode] = {}
        for name in names:
            readings = [item["modes"][mode][name] for item in all_images if name in item["modes"][mode]]
            values = [r["distance_px"] for r in readings]
            if values:
                stats[mode][name] = {"valid_images": len(values), "min_px": min(values), "median_px": float(np.median(values)),
                                     "max_px": max(values), "range_px": max(values)-min(values),
                                     "min_valid_columns": min(r["valid_columns"] for r in readings),
                                     "max_column_mad_px": max(r["mad_px"] for r in readings)}
    write_csv(out / "localization.csv", alignments, ["image", "row", "dx", "dy", "score"])
    if details:
        write_csv(out / "measurement_details.csv", details, list(details[0]))
    summary = {"input_dir": str(source), "image_count": len(paths), "measurement_unit": "px",
               "measurements_expected_per_mode": len(paths)*len(regions), "config": str(args.config.resolve()),
               "algorithm": "algorithms.measurement_opening_gap.measure_opening_gap",
               "measurement_definition": "Vertical distance between opposing bright tabs; median of 49 paired columns",
               "limitations": ["No mm calibration or dimensional acceptance limits", "New ROIs; no manual ground truth", "Variation across images is not established repeatability"],
               "errors": errors, "modes": stats,
               "review_notes": [
                   "5.png U7: dark-side boundary remains sensitive to the gray background and requires physical-edge confirmation",
                   "6.png L1: smaller gap; upper tab visibly extends downward, no ground-truth dimensional confirmation",
                   "4.png U1: large difference between edge modes; peak follows a different bright transition",
                   "Search windows were narrowed after the initial trial to avoid a gray-background ramp at 5.png L6"
               ]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"images": all_images, "regions": all_images[0]["regions"], "image_size": config["image_size"]}
    (out / "measurements.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    template = (ROOT / "tools/50_2_edge_viewer_template.html").read_text(encoding="utf-8")
    template = template.replace("50-2", "Lab Test").replace("12 处位置概览", f"{len(regions)} 处位置概览")
    template = template.replace('const [width, height] = data.image_size;', 'const regions = item.regions;\n    const [width, height] = data.image_size;')
    template = template.replace('data.regions.forEach(region => drawRegion', 'regions.forEach(region => drawRegion')
    template = template.replace('const region = data.regions.find', 'const region = regions.find')
    template = template.replace('regionSelect.value = data.regions[0].name;', 'regionSelect.value = "all";')
    (out / "edge_viewer.html").write_text(template.replace("__MEASUREMENT_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")), encoding="utf-8")
    table = "<tr><th>图片</th>" + "".join(f"<th>{n}</th>" for n in names) + "</tr>"
    for row in rows["dark_plateau"]:
        table += f'<tr><th><a href="annotated/{html.escape(row["image"])}">{html.escape(row["image"])}</a></th>' + "".join(f'<td>{row[n]:.3f}</td>' if n in row else '<td>失败</td>' for n in names) + '</tr>'
    table += '<tr><th>极差</th>' + ''.join(f'<td>{stats["dark_plateau"][n]["range_px"]:.3f}</td>' if n in stats['dark_plateau'] else '<td>—</td>' for n in names) + '</tr>'
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Lab Test 开口尺寸测试</title>
<style>body{{font:16px "Microsoft YaHei",sans-serif;margin:32px;color:#183040}}table{{border-collapse:collapse;font-size:14px}}td,th{{border:1px solid #bbc9d2;padding:8px;text-align:right}}th{{background:#edf3f7}}a{{color:#007795}}img{{max-width:100%}}p{{line-height:1.7}}</style>
<h1>Lab Test：{len(paths)} 张图的 {len(regions)} 处开口尺寸试测</h1><p>单位：<b>像素 px</b>。上排从左到右 U1–U7，下排从左到右 L1–L7；测量相对金属凸片之间的竖直间距。原 demo 算法，49 列取中位数，9 px 高斯平滑，上下暗侧比例 3%。上下排分别做模板平移定位，测量使用未缩放的原图。</p>
<p><a href="edge_viewer.html">逐图放大查看测量边缘</a> · <a href="lab_test_dark_plateau.csv">主要结果 CSV</a> · <a href="lab_test_peak.csv">最大梯度对照 CSV</a> · <a href="measurement_details.csv">有效列数及 MAD 明细</a></p>
<p>两种模式的算法错误合计：{len(errors)}。获得读数不等于尺寸合格；没有毫米标定、人工真值及公差。不同图之间的极差包含工件差异、姿态和光照影响，不能直接解释为测量重复性。</p><div style="overflow:auto"><table>{table}</table></div>
<p><b>取边复核：</b>5.png 的 U7 仍对灰色背景较敏感；6.png 的 L1 读数较小，图中上侧凸片明显向下延伸；4.png 的 U1 在两种判边方式下差异明显。这些值是当前算法读数，尚未用实体真值验证。首轮发现 5.png 的 L6 上边缘追入灰色背景，现已统一收窄搜索窗；首轮配置及数据保存在 diagnostic_broad_windows 目录，可追溯对照。</p>
<p>紫色线表示原算法找到的上下边缘，绿色线表示间距。下面每张局部总览均为上排 U1–U7、下排 L1–L7。反光及灰色背景可能影响暗侧取边，需结合原图核对。</p>'''
    for path in paths:
        report += f'<h2>{html.escape(path.name)}</h2><a href="annotated/{html.escape(path.name)}">全图标注</a><p><img src="details/{html.escape(path.name)}" alt="{html.escape(path.name)} 的14处取边"></p>'
    (out / "report.html").write_text(report + '</html>', encoding="utf-8")
    print(f"Report: {out / 'report.html'}; errors={len(errors)}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
