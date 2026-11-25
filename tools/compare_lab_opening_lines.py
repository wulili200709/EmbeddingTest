"""Build a reproducible 84-point exploratory line-measurement comparison.

The source workbook and legacy algorithm are read-only. Image row order is
explicitly mapped to laboratory names; the workbook only supplies references,
never edge locations or algorithm parameters. All measurements use original
pixels. Human four-point measurements start empty in the companion HTML tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from algorithms.measurement_opening_gap import measure_opening_gap, regions_from_params
from algorithms.measurement_opening_lines import measure_opening_lines
from algorithms.measurement_opening_body_edge import measure_opening_body_edges
from lab_line_validation import POINT_NAMES, evaluate_methods
from test_lab_openings import locate_rows, read_image, write_csv

MAPPING = {"top": [f"L{i}" for i in range(7, 0, -1)],
           "bottom": [f"U{i}" for i in range(7, 0, -1)]}
DISTANCE_RULE = "下线段中点到上直线的垂距（待与实验室确认）"
LAB_COLUMNS = ["T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF", "AG"]


def physical_identity(old_name):
    """Old U/L identifiers describe top/bottom camera rows, not lab names."""
    if old_name not in {f"{r}{i}" for r in "UL" for i in range(1, 8)}:
        raise ValueError(f"Unexpected old camera region: {old_name}")
    physical_row = "top" if old_name[0] == "U" else "bottom"
    col = int(old_name[1:])
    return MAPPING[physical_row][col - 1], physical_row, col


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lab_workbook(path):
    """Read raw XLSX references via ZIP/XML without opening or changing Excel."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cells = {}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.text or "" for t in si.findall(".//s:t", ns))
                       for si in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        for cell in root.findall(".//s:c", ns):
            value = cell.find("s:v", ns)
            raw = value.text if value is not None else None
            kind = cell.get("t")
            if kind == "s":
                raw = strings[int(raw)]
            elif kind == "inlineStr":
                raw = "".join(t.text or "" for t in cell.findall(".//s:t", ns))
            elif raw is not None and kind not in {"str", "e"}:
                raw = float(raw)
            cells[cell.attrib["r"]] = {"value": raw, "formula": cell.find("s:f", ns) is not None}
    headers = [cells.get(f"{c}1", {}).get("value") for c in LAB_COLUMNS]
    if headers != list(POINT_NAMES):
        raise ValueError(f"Workbook lab headers T1:AG1 differ: {headers}")
    expected_camera_headers = MAPPING["top"] + MAPPING["bottom"]
    camera_headers = [cells.get(f"{chr(66+i)}1", {}).get("value") for i in range(14)]
    if camera_headers != expected_camera_headers:
        raise ValueError("Workbook B1:O1 does not match the user's confirmed image order")
    lab, provenance = {}, []
    for r in range(2, 8):
        filename = cells.get(f"A{r}", {}).get("value")
        if filename != f"{r-1}.png":
            raise ValueError(f"Unexpected workbook product identity A{r}: {filename!r}")
        for col, name in zip(LAB_COLUMNS, POINT_NAMES):
            ref = f"{col}{r}"
            entry = cells[ref]
            value = entry["value"]
            if entry["formula"] or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Expected a positive, raw laboratory reading at {ref}")
            lab[(filename, name)] = value
            provenance.append({"image": filename, "name": name, "sheet": "Sheet1", "cell": ref, "lab_mm": value})
    scale = cells.get("P9", {}).get("value")
    if not isinstance(scale, (float, int)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Workbook P9 has no finite positive cached comparison scale")
    return lab, float(scale), provenance


def draw_fitted_segment(image, segment, color, diagnostic=False):
    """Failed fits remain visible as red dashes, distinct from measured lines."""
    start, end = np.asarray(segment, dtype=float)
    delta = end-start
    length = float(np.linalg.norm(delta))
    if not diagnostic:
        cv2.line(image, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), color, 1, cv2.LINE_AA)
        return
    for distance in np.arange(0, length, 7.):
        a = start+delta*(distance/max(length, 1e-9))
        b = start+delta*(min(distance+4., length)/max(length, 1e-9))
        cv2.line(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), (70, 70, 245), 1, cv2.LINE_AA)


def draw_measurements(bgr, points, labels=True, show_previous=False):
    out = bgr.copy()
    for point in points:
        old, auto = point["legacy"], point["auto"]
        body = "anchor_mask" in auto.get("upper", {})
        if old.get("distance_px") is not None and not body:
            columns = np.asarray(old["column_edges"])
            for index in (1, 2):
                poly = np.rint(columns[:, [0, index]]).astype(np.int32)
                cv2.polylines(out, [poly], False, (245, 60, 220), 1, cv2.LINE_AA)
        for side in ("upper", "lower"):
            edge = auto.get(side, {})
            if show_previous:
                previous_segment = point.get("previous_auto", {}).get(side, {}).get("segment")
                if previous_segment:
                    ends = [tuple(np.rint(p).astype(int)) for p in previous_segment]
                    cv2.line(out, *ends, (0, 140, 255), 1, cv2.LINE_AA)
            for index, (sample, valid, inlier) in enumerate(zip(edge.get("points", []), edge.get("valid_mask", []), edge.get("inlier_mask", []))):
                if body and valid and edge.get("anchor_mask", [False] * len(edge["points"]))[index]:
                    cv2.circle(out, tuple(np.rint(sample).astype(int)), 1, (30, 215, 70) if inlier else (70, 70, 245), -1)
                elif not body and (not valid or not inlier):
                    cv2.circle(out, tuple(np.rint(sample).astype(int)), 1, (40, 60, 250), -1)
            segment = edge.get("segment")
            if segment:
                draw_fitted_segment(out, segment, (255, 225, 20) if body else (30, 225, 250), auto['status']=='failed')
        if auto.get("dimension_segment"):
            ends = [tuple(np.rint(p).astype(int)) for p in auto["dimension_segment"]]
            cv2.line(out, *ends, (30, 240, 100), 2, cv2.LINE_AA)
        if labels:
            region = point["region"]
            value = auto.get("distance_px")
            label = f"{point['name']} {value:.2f}px" if value is not None else f"{point['name']} FAILED"
            position = (region["center_x"] - 95, region["upper_y"][0] - 12)
            color = (90, 230, 130) if auto["status"] == "ok" else (40, 180, 250)
            cv2.putText(out, label, position, cv2.FONT_HERSHEY_SIMPLEX, .57, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, label, position, cv2.FONT_HERSHEY_SIMPLEX, .57, color, 1, cv2.LINE_AA)
    return out


def save_image(path, bgr):
    cv2.imencode(".png", bgr)[1].tofile(str(path))


def save_panels(path, bgr, points):
    annotated = draw_measurements(bgr, points, labels=False)
    panels = Image.new("RGB", (7 * 280, 2 * 265), "#17242c")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    for index, point in enumerate(points):
        region, auto = point["region"], point["auto"]
        x0, x1 = region["center_x"] - 135, region["center_x"] + 135
        y0, y1 = region["upper_y"][0] - 15, region["lower_y"][1] + 20
        crop = Image.fromarray(cv2.cvtColor(annotated[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        px, py = index % 7 * 280 + 5, index // 7 * 265 + 5
        panels.paste(crop, (px, py + 46))
        draw = ImageDraw.Draw(panels)
        value = auto.get("distance_px")
        label = f"{point['name']}: {value:.3f} px" if value is not None else f"{point['name']}: no result"
        draw.text((px + 5, py), label, fill="#9cebc3", font=font)
        draw.text((px + 5, py + 22), 'FAILED - diagnostic only' if auto['status']=='failed' else auto["status"], fill="#ffce78", font=font)
    panels.save(path)


def save_target_comparison(path, bgr, points):
    """Compare the same original pixels at identical scale, before and after."""
    selected = [next(p for p in points if p["name"] == n) for n in ("U2", "U4", "U5")]
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
    crop_height = max(p["region"]["lower_y"][1] - p["region"]["upper_y"][0] + 24 for p in selected)
    row_height = crop_height * 2 + 55
    canvas = Image.new("RGB", (3 * 480, row_height * 2), "#17242c")
    draw = ImageDraw.Draw(canvas)
    for col, point in enumerate(selected):
        r = point["region"]
        x0, x1 = r["center_x"] - 115, r["center_x"] + 115
        y0, y1 = r["upper_y"][0] - 12, r["lower_y"][1] + 12
        for row, key in enumerate(("previous_auto", "auto")):
            annotated = bgr.copy()
            result = point.get(key, {})
            color = (0, 140, 255) if row == 0 else (255, 225, 20)
            for side in ("upper", "lower"):
                segment = result.get(side, {}).get("segment")
                if segment:
                    draw_fitted_segment(annotated, segment, color, row==1 and result.get('status')=='failed')
            crop = Image.fromarray(cv2.cvtColor(annotated[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            crop = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.NEAREST)
            px, py = col * 480 + 10, row * row_height
            coefficients = result.get("upper", {}).get("coefficients")
            center_y = coefficients[0]*r['center_x']+coefficients[1] if coefficients else None
            width = result.get("fit_width", "?")
            text = f"{point['name']} {'BEFORE' if row == 0 else 'AFTER'} {width}cols  upper y={center_y:.2f}" if center_y is not None else f"{point['name']} no fit"
            draw.text((px, py + 8), text, font=font, fill="#f2ba73" if row == 0 else "#72ece9")
            canvas.paste(crop, (px, py + 45))
    canvas.save(path)


def json_write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_boundary_detail(path, bgr, points, name, side):
    """Uncovered original beside prior/current lines at the identical pixel crop."""
    p = next(p for p in points if p['name'] == name)
    cx = p['region']['center_x']
    old_coeff = p.get('previous_auto', {}).get(side, {}).get('coefficients')
    center_y = round(old_coeff[0]*cx+old_coeff[1]) if old_coeff else sum(p['region'][side+'_y'])//2
    x0, x1, y0, y1 = cx-80, cx+81, center_y-27, center_y+18
    factor, panel_width, header = 4, (x1-x0)*4, 48
    canvas = Image.new('RGB', (3*panel_width, header+(y1-y0)*factor), '#17242c')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 21)
    draw = ImageDraw.Draw(canvas)
    for col, key in enumerate((None, 'previous_auto', 'auto')):
        annotated = bgr.copy()
        title = f'{name} ORIGINAL PIXELS - no overlay'
        color = (0,140,255) if col == 1 else (255,225,20)
        if key:
            result = p.get(key, {})
            edge = result.get(side, {})
            if edge.get('segment'):
                draw_fitted_segment(annotated, edge['segment'], color, key=='auto' and result.get('status')=='failed')
            coeff = edge.get('coefficients')
            text_y = f"{coeff[0]*cx+coeff[1]:.2f}" if coeff else 'no fit'
            title = f"{name} {'PREVIOUS' if col == 1 else 'REVISED'} {result.get('status','')} - {side} y={text_y}"
        crop = Image.fromarray(cv2.cvtColor(annotated[y0:y1,x0:x1],cv2.COLOR_BGR2RGB))
        canvas.paste(crop.resize((panel_width,(y1-y0)*factor),Image.Resampling.NEAREST),(col*panel_width,header))
        draw.text((col*panel_width+10,12),title,font=font,fill='#ffffff' if col==0 else '#f2ba73' if col==1 else '#72ece9')
    canvas.save(path)


def number(value):
    return "—" if value is None else f"{value:.5f}"


def write_report(out, data, flat):
    if data["meta"].get("method") == "body_outline":
        return write_body_report(out, data)
    counts = Counter(p["auto"]["status"] for im in data["images"] for p in im["points"])
    report = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lab Test：两条线测量对照</title>
<style>body{font:16px/1.65 "Microsoft YaHei",sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#203843;background:#f5f8fa}h1{font-size:27px}h2{font-size:21px}table{border-collapse:collapse;background:white}td,th{border:1px solid #cbd6dc;padding:7px 10px;text-align:right;white-space:nowrap}th{background:#e4edf2}a{color:#006d91}img{max-width:100%}.scroll{overflow:auto}.note{background:#fff4d7;padding:12px 18px;border-left:4px solid #d69c22}.links{display:flex;gap:20px;flex-wrap:wrap}.review{color:#985a00}</style>
<h1>Lab Test：上下两排、84 个位置的两条线测量对照</h1>
<p><b>图片上排从左到右 L7 → L1，下排 U7 → U1。</b>数据表统一按 U1～U7、L1～L7 排列；逐点明细同时记录物理行、列与实验室单元格。</p>
<div class="links"><a href="four_point_review.html">打开四点手动复核工具</a><a href="measurement_comparison.csv">84 点完整数据 CSV</a><a href="auto_pixels.csv">直线法像素表</a><a href="legacy_pixels.csv">原算法像素表</a><a href="measurements.json">坐标与诊断 JSON</a><a href="validation.json">留一验证完整结果</a></div>''']
    report.append(f'<p>本次原图测量：6 张 × 2 排 × 7 点。直线候选通过内部诊断 {counts["ok"]} 点、待复核 {counts["review"]} 点、未出数 {counts["failed"]} 点。<b>手动四点尚未完成；初始记录为空。</b></p>')
    report.append('''<div class="note">当前直线法是候选方案，不能直接用于尺寸判定。固定中央 121 列分别提取上、下边：保留多个梯度峰，寻找跨列支持的直线分支，再优先选择靠开口一侧的分支；多条边、台阶、弱边或不平行仍需复核。紫色是原算法的 49 列取边，黄色是直线候选，绿色是垂距，红点是弱候选或偏离拟合的点；原始候选全部保存在 JSON 中。较长窗口也可能纳入不应测量的结构，必须查看叠加图。</div>''')
    report.append(f'<p>垂距定义：<b>{DISTANCE_RULE}</b>。也保存反向垂距和夹角；不平行时，两条无限直线没有唯一的恒定间距。上下排仅做平移定位，尚未验证任意旋转或新工装。</p>')
    report.append(f'<p>直接换算使用当前表 P9：{data["meta"]["provisional_scale_mm_per_px"]:.12f} mm/px，是样本拟合比例，<b>不是独立物理标定</b>。下表的方法比较分别在训练产品上重估比例及逐位置偏移，误差 = 预测 − 实验室。原 XLSX 未修改。</p>')
    report.append('<h2>误差比较（mm）</h2><div class="scroll"><table><tr><th>方法</th><th>评估方式</th><th>覆盖</th><th>平均绝对误差</th><th>RMSE</th><th>最大绝对误差</th><th>最大误差位置</th></tr>')
    for method, title in (("legacy", "原49列中位数"), ("auto", "121列直线候选")):
        for stage, label in (("same_batch", "同批拟合回算"), ("leave_one_product_out", "整件留一：5件训练/1件预测")):
            m = data["validation"]["methods"][method][stage]
            worst = m["max_abs_at"] or {}
            report.append(f'<tr><td>{title}</td><td>{label}</td><td>{m["n"]}/{m["expected_count"]}</td><td>{number(m["mae"])}</td><td>{number(m["rmse"])}</td><td>{number(m["max_abs"])}</td><td>{worst.get("image", "")} / {worst.get("name", "")}</td></tr>')
    report.append('</table></div><p class="note">6 张图同时用于开发，整件留一仍只是探索比较，不是独立验收。待复核点也计入统计，没有按实验室误差删点；失败点保持空值，覆盖数量不同的指标不能直接视为精度改善。实验室手动选点差异尚不能解释全部误差。产品公差 ±0.12 mm 不是测量系统的允许误差。</p>')
    for method, title in (("legacy", "原算法"), ("auto", "直线候选")):
        fit = data["validation"]["methods"][method]["same_batch"]
        if not fit.get("training_evaluable", False):
            missing = "、".join(fit.get("missing_training_points", []))
            report.append(f'<p><b>{title}未计算补偿误差：</b>{html.escape(missing)} 没有可参与拟合的读数，无法按预定的14位置模型估计共同比例及全部补偿量。表中的“—”表示不能评估，绝不是误差为零。其余候选像素值仍完整保留。</p>')
    report.append('<h2>逐位置整件留一误差（mm）</h2><div class="scroll"><table><tr><th>位置</th><th>原算法 MAE</th><th>直线候选 MAE</th><th>原算法最大绝对误差</th><th>直线候选最大绝对误差</th></tr>')
    for name in POINT_NAMES:
        old = data["validation"]["methods"]["legacy"]["leave_one_product_out"]["per_point"][name]
        new = data["validation"]["methods"]["auto"]["leave_one_product_out"]["per_point"][name]
        report.append(f'<tr><th>{name}</th><td>{number(old["mae"])}</td><td>{number(new["mae"])}</td><td>{number(old["max_abs"])}</td><td>{number(new["max_abs"])}</td></tr>')
    report.append('</table></div><h2>直线候选原始像素表（未补偿）</h2><div class="scroll"><table><tr><th>图片</th>' + ''.join(f'<th>{n}</th>' for n in POINT_NAMES) + '</tr>')
    for item in data["images"]:
        by_name = {p["name"]: p for p in item["points"]}
        report.append(f'<tr><th>{item["name"]}</th>' + ''.join(f'<td>{number(by_name[n]["auto"]["distance_px"])}</td>' for n in POINT_NAMES) + '</tr>')
    report.append('</table></div>')
    for item in data["images"]:
        filename = html.escape(item["name"])
        report.append(f'<h2>{filename} · 上排 L7→L1，下排 U7→U1</h2><a href="annotated/{filename}">查看完整原尺寸标注</a><p><img src="details/{filename}" alt="{filename} 的14处开口"></p>')
    report.append('<p>复核方法：在交互工具中选位置，放大原图；依次点上边两点、下边两点，并确认记录。自动建议必须人工确认才能计为已复核；导出 JSON 保存坐标，CSV 保存尺寸。程序不替代人工确认实验室采用的实体边界和非平行距离规则。</p><p>首轮仅按最强梯度取边时，3.png 的 L7 下线跳到底板亮边；该轮 JSON、统计及6张局部图保存在 diagnostic_first_strongest 目录，作为防跳边修复的诊断记录。最终结果不能替代新产品、重新放置拍照后的验证。</p></html>')
    (out / "report.html").write_text("\n".join(report), encoding="utf-8")


def write_body_report(out, data):
    counts = Counter(p["auto"]["status"] for im in data["images"] for p in im["points"])
    body = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>按指定主体边优化的开口测量</title>
<style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;background:#f4f8fa;color:#193441;max-width:1480px;margin:24px auto;padding:0 20px}h1{font-size:28px}h2{font-size:21px}a{color:#006d91}img{max-width:100%;height:auto}table{border-collapse:collapse;background:white}th,td{padding:7px 10px;border:1px solid #c7d5df;text-align:right;white-space:nowrap}th{background:#e5edf2}.scroll{overflow:auto}.notice{background:#fff1cf;padding:12px 16px}.links{display:flex;flex-wrap:wrap;gap:20px}</style>
<h1>按指定的主体外轮廓优化取边</h1>
<p>图片上排从左到右 <b>L7 → L1</b>，下排 <b>U7 → U1</b>。共6张图、84个位置；数据表顺序为U1～U7、L1～L7。</p>
<p>自动取边目标是金属唇边朝向开口的外轮廓。以原图中的局部灰度跃迁定位边界；遇到两侧支持不足或混边时，查找具有左右连续支持的同一条轮廓来恢复取点。用左右采样段确定直线，中间突起或凹口保留为诊断点。上下线独立拟合，保留实际斜率；实验室数值不参与选边。</p>
<div class="links"><a href="four_point_review.html">交互查看取边：新版与上版叠加</a><a href="measurement_comparison.csv">84处完整数据 CSV</a><a href="auto_pixels.csv">新版像素表</a><a href="measurements.json">全部原图坐标和诊断</a></div>''']
    body.append(f'<p>本轮：通过内部图像诊断 {counts["ok"]} 处、输出候选并提示复核 {counts["review"]} 处、无有效输出 {counts["failed"]} 处。“通过”仅指图像诊断；还没有独立的毫米精度验证。手动选点是可选检查功能，自动结果无需重新手动点击。</p>')
    body.append('<p>若左右采样段混入不同边界、同一条直线无法解释这些实测点，则只保留诊断图，尺寸留空。不能把偏到背景中的拟合线当成有效测量。</p>')
    body.append('<h2>用户圈出的 2 号图 U3 下边</h2><p>从左到右：无覆盖的原始像素、上一版橙线、本轮结果。上一版已经拒测，但诊断线右端仍偏到了背景。本轮继续修正边缘定位；青色实线表示可供复核的候选，红色虚线表示拒测后的诊断线。三幅使用完全相同的裁剪与倍率。</p><img src="u3_boundary_detail.png" alt="U3下边原始像素与修正前后的同尺度对照">')
    width = data['meta']['fit_width']
    half = (width - 1) // 2
    scale = min(1., (width-1)/140.)
    body.append(f'<p class="notice">本轮保留 <b>{width} 列实际采样（中心左右各 {half} px）</b>，显示线段横向跨度为 {width-1} px。左右定线区仍为距中心 {math.ceil(40*scale)}～{math.floor(65*scale)} px；本次修正混边位置的轮廓选择，采样范围不变。</p>')
    body.append('<h2>2号图：U2、U4、U5 对照检查</h2><p><b>上行橙线：上一版；下行青线：本轮。</b>两行使用相同原图裁剪和相同放大倍率。图中 upper y 是上拟合线在开口中心的原图纵坐标，越小表示越靠上。</p><img src="target_comparison_2.png" alt="2号图三个位置的原像素前后对照">')
    body.append('<div class="scroll"><table><tr><th>位置</th><th>上版夹角</th><th>新版夹角</th><th>新版垂距 px</th><th>状态</th></tr>')
    im = next(i for i in data["images"] if i["name"] == "2.png")
    for name in ("U3", "U2", "U4", "U5"):
        p = next(p for p in im["points"] if p["name"] == name)
        old, new = p.get("previous_auto", {}), p["auto"]
        body.append(f'<tr><th>{name}</th><td>{number(old.get("angle_difference_deg"))}°</td><td>{number(new.get("angle_difference_deg"))}°</td><td>{number(new.get("distance_px"))}</td><td>{new["status"]}</td></tr>')
    body.append('</table></div><p>夹角用于检查取线方向，不是把两条线拉成平行的目标。用户截图用于确定测量对象，未把截图中的手画线当成精确原图坐标或毫米真值。</p>')
    body.append('<h2>6张图的完整取边结果</h2><p>青色实线：主体外轮廓拟合候选；红色虚线：拒测后的诊断线，不代表正确边界；绿色小点：左右定线区内点；红色小点：该区偏离拟合线的候选；绿色连线：垂距。failed位置不输出尺寸。交互页面保留全部候选点和偏差，候选点不等于已确认的实体边。</p>')
    for item in data["images"]:
        filename = html.escape(item["name"])
        body.append(f'<h2>{filename}</h2><a href="annotated/{filename}">完整原尺寸标注</a><p><img src="details/{filename}" alt="{filename} 上下两排的14处取边"></p>')
    body.append('<h2>全部原始像素尺寸</h2><div class="scroll"><table><tr><th>图片</th>' + ''.join(f'<th>{n}</th>' for n in POINT_NAMES) + '</tr>')
    for im in data["images"]:
        by_name = {p["name"]: p for p in im["points"]}
        body.append(f'<tr><th>{im["name"]}</th>' + ''.join(f'<td>{number(by_name[n]["auto"].get("distance_px"))}</td>' for n in POINT_NAMES) + '</tr>')
    body.append('</table></div>')
    body.append(f'<p class="notice">当前垂距规则：{DISTANCE_RULE}。mm/px仍是现有样本拟合比例，并非物理标定。先确认主体边定义与图像重复性，再确定补偿；本次原Excel、原图及上一版报告均未修改。</p>')
    body.append('<details><summary>实验室数值的探索性对照</summary><p>取边算法没有使用实验室数值。以下每种方法分别拟合比例和逐位置偏移；整件留一每次用5件拟合、1件预测，6件已参与算法开发，因此仍不属于独立验收。覆盖不同的结果不能直接解释为精度改善。</p><table><tr><th>方法</th><th>评估方式</th><th>覆盖</th><th>MAE mm</th><th>最大绝对误差 mm</th></tr>')
    for method, title in (("legacy", "原49列法"), ("auto", "主体外轮廓法")):
        for stage, label in (("same_batch", "同批拟合"), ("leave_one_product_out", "整件留一")):
            m = data["validation"]["methods"][method][stage]
            body.append(f'<tr><td>{title}</td><td>{label}</td><td>{m["n"]}/{m["expected_count"]}</td><td>{number(m["mae"])}</td><td>{number(m["max_abs"])}</td></tr>')
    body.append('</table>')
    fit = data['validation']['methods']['auto']['same_batch']
    if not fit.get('training_evaluable', False):
        missing = '、'.join(fit.get('missing_training_points', []))
        body.append(f'<p class="notice">{html.escape(missing)} 缺少有效读数，完整14位置补偿模型无法拟合。0/84表示补偿模型不能评估；其他候选像素仍保留，不表示测量误差为零。</p>')
    body.append('<a href="validation.json">完整对照参数及每件预测</a></details></html>')
    (out / "report.html").write_text("\n".join(body), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/examples/lab_test_opening_gaps.json")
    parser.add_argument("--method", choices=("body", "branch"), default="body")
    parser.add_argument("--fit-width", type=int, help="Actual odd sampling-column count; body default 101, branch default 121")
    parser.add_argument("--previous-report", type=Path, default=ROOT / "artifacts/reports/lab_body_edge_local_transition_20261015/measurements.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/reports/lab_body_edge_coherent_20261016")
    args = parser.parse_args(argv)
    source, out = args.input_dir.resolve(), args.output_dir.resolve()
    workbook = args.workbook.resolve()
    if out == source or source in out.parents:
        raise ValueError("Place report output outside the source-data folder")
    workbook_hash = digest(workbook)
    body_method = args.method == "body"
    fit_width = args.fit_width if args.fit_width is not None else (101 if body_method else 121)
    if fit_width < 21 or fit_width % 2 != 1:
        raise ValueError('fit-width must be an odd column count >= 21')
    anchor_scale = min(1., (fit_width-1)/140.)
    measure = measure_opening_body_edges if body_method else measure_opening_lines
    algorithm = ROOT / "algorithms" / ("measurement_opening_body_edge.py" if body_method else "measurement_opening_lines.py")
    previous = {}
    if body_method and args.previous_report.exists():
        if args.previous_report.resolve().parent == out:
            raise ValueError("Keep previous measurements separate from new output")
        previous_data = json.loads(args.previous_report.read_text(encoding="utf-8"))
        if previous_data["meta"]["mapping"] != MAPPING:
            raise ValueError("Previous report uses a different physical point mapping")
        previous = {item["name"]: item for item in previous_data["images"]}
    lab, provisional_scale, lab_provenance = read_lab_workbook(workbook)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = [source / f"{i}.png" for i in range(1, 7)]
    for subdir in ("originals", "annotated", "details"):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    reference = cv2.cvtColor(read_image(source / config["reference_image"]), cv2.COLOR_BGR2GRAY)
    template_regions = regions_from_params(config)
    identities = [physical_identity(r.name) for r in template_regions]
    if [n for n, _, _ in identities] != MAPPING["top"] + MAPPING["bottom"]:
        raise ValueError("Configuration order must be physical top row followed by bottom row")
    data = {"schema_version": 1, "meta": {
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "body_outline" if body_method else "gradient_branch",
        "mapping": MAPPING, "report_order": list(POINT_NAMES),
        "provisional_scale_mm_per_px": provisional_scale,
        "scale_status": "sample_fitted_not_independent_physical_calibration",
        "distance_rule": DISTANCE_RULE, "input_dir": str(source),
        "workbook": str(workbook), "workbook_sha256": workbook_hash,
        "lab_reference_range": "Sheet1!T2:AG7", "fit_width": fit_width,
        "manual_reviewed_initially": 0, "source_pixels_resampled": False,
        "algorithm_sha256": digest(algorithm),
        "previous_report": str(args.previous_report.resolve()) if previous else None,
        "previous_fit_width": previous_data['meta'].get('fit_width') if previous else None,
        "legacy_algorithm_sha256": digest(ROOT / "algorithms/measurement_opening_gap.py"),
        "runner_sha256": digest(Path(__file__)),
        "notes": ["实验室参考值不参与找边、窗口选择或质量阈值。", f"采样窗口{fit_width}列；没有按实验室误差择优调参。", f"左右距中心{math.ceil(40*anchor_scale)}～{math.floor(65*anchor_scale)}像素的轮廓段确定自由斜率直线，中段保留实测诊断。" if body_method else "按梯度分支选边。", "原图上下排仅平移定位，未验证新姿态。", "当前含义为两条主体边线垂距，不是最窄喉口尺寸。"]},
        "images": [], "validation": {}}
    records = []
    lab_cells = {(p["image"], p["name"]): p["cell"] for p in lab_provenance}
    for path in paths:
        bgr = read_image(path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if list(gray.shape[::-1]) != config["image_size"]:
            raise ValueError(f"Unexpected image size: {path.name}")
        shifts = locate_rows(gray, reference, config["localization"])
        item = {"name": path.name, "url": f"originals/{path.name}", "width": bgr.shape[1], "height": bgr.shape[0],
                "sha256": digest(path), "points": [], "localization": {
                    "top": shifts["U"], "bottom": shifts["L"]}}
        previous_image = previous.get(path.name)
        if previous_image and previous_image["sha256"] != item["sha256"]:
            raise ValueError(f"Previous image differs: {path.name}")
        previous_points = {p["name"]: p for p in previous_image["points"]} if previous_image else {}
        for original in template_regions:
            name, physical_row, physical_col = physical_identity(original.name)
            shift = shifts[original.name[0]]
            region = replace(original, name=name, center_x=original.center_x + shift["dx"],
                             upper_y=tuple(y + shift["dy"] for y in original.upper_y),
                             lower_y=tuple(y + shift["dy"] for y in original.lower_y))
            try:
                old = asdict(measure_opening_gap(gray, region, edge_mode="dark_plateau", include_column_edges=True,
                    **{key: config[key] for key in ("blur_ksize", "min_strength", "min_valid_ratio", "upper_dark_fraction", "lower_dark_fraction")}))
                old["status"] = "measured"
            except RuntimeError as exc:
                # Insufficient image evidence is a missing measurement, not a
                # reason to silently omit the point or stop the whole batch.
                old = {"name": name, "center_x": region.center_x, "distance_px": None,
                       "column_edges": [], "status": "failed", "error": str(exc)}
            auto = measure(gray, region, fit_width=fit_width, blur_ksize=config["blur_ksize"], min_strength=config["min_strength"])
            point = {"name": name, "physical_row": physical_row, "physical_col": physical_col,
                     "legacy_region_name": original.name, "region": asdict(region), "legacy": old, "auto": auto,
                     "lab_mm": lab[(path.name, name)], "lab_cell": lab_cells[(path.name, name)], "comparison": {}}
            if name in previous_points:
                earlier = previous_points[name]
                if (earlier["physical_row"], earlier["physical_col"]) != (physical_row, physical_col):
                    raise ValueError("Previous point identity differs")
                point["previous_auto"] = {key: earlier["auto"].get(key) for key in
                    ("status", "distance_px", "diagnostic_distance_px", "angle_difference_deg", "reasons", "fit_width")}
                point["previous_auto"].update({side: {key: earlier["auto"][side].get(key)
                    for key in ("segment", "coefficients")} for side in ("upper", "lower")})
            item["points"].append(point)
            records.append({"image": path.name, "name": name, "legacy_px": old["distance_px"],
                            "auto_px": auto["distance_px"], "lab_mm": point["lab_mm"], "status": auto["status"]})
        shutil.copy2(path, out / "originals" / path.name)
        save_image(out / "annotated" / path.name, draw_measurements(bgr, item["points"]))
        save_panels(out / "details" / path.name, bgr, item["points"])
        if body_method and path.name == "2.png":
            save_target_comparison(out / "target_comparison_2.png", bgr, item["points"])
            save_boundary_detail(out / "u2_boundary_detail.png", bgr, item["points"], 'U2', 'upper')
            save_boundary_detail(out / "u3_boundary_detail.png", bgr, item["points"], 'U3', 'lower')
        data["images"].append(item)
        print(f"{path.name}: {dict(Counter(p['auto']['status'] for p in item['points']))}", flush=True)
    data["validation"] = evaluate_methods(records)
    predictions = {(method, stage): {(p["image"], p["name"]): p for p in values[stage]["predictions"]}
                   for method, values in data["validation"]["methods"].items()
                   for stage in ("same_batch", "leave_one_product_out")}
    flat = []
    for item in data["images"]:
        points = {p["name"]: p for p in item["points"]}
        for name in POINT_NAMES:
            p = points[name]
            row = {"image": item["name"], "point": name, "physical_row": p["physical_row"], "physical_col_left_to_right": p["physical_col"],
                   "lab_cell": p["lab_cell"], "lab_mm": p["lab_mm"], "legacy_px": p["legacy"]["distance_px"],
                   "legacy_status": p["legacy"]["status"], "legacy_error": p["legacy"].get("error", ""),
                   "auto_px": p["auto"]["distance_px"], "auto_reverse_px": p["auto"]["reverse_distance_px"],
                   "auto_diagnostic_only_px": p["auto"].get("diagnostic_distance_px"),
                   "auto_angle_difference_deg": p["auto"]["angle_difference_deg"], "auto_status": p["auto"]["status"],
                   "auto_reasons": ";".join(p["auto"]["reasons"]), "manual_px": "", "manual_reviewed": False,
                   "distance_rule": DISTANCE_RULE, "provisional_mm_per_px": provisional_scale}
            if body_method:
                row.update({"previous_line_px": p.get("previous_auto", {}).get("distance_px"),
                            "previous_angle_difference_deg": p.get("previous_auto", {}).get("angle_difference_deg")})
                for side in ("upper", "lower"):
                    for key in ("anchor_rmse_px", "left_anchor_support", "right_anchor_support", "middle_max_deviation_px"):
                        row[f"{side}_{key}"] = p["auto"][side].get(key)
            for method in ("legacy", "auto"):
                px = p[method]["distance_px"]
                row[f"{method}_provisional_mm"] = None if px is None else px * provisional_scale
                p["comparison"][method] = {}
                for stage in ("same_batch", "leave_one_product_out"):
                    prediction = predictions[(method, stage)][(item["name"], name)]
                    p["comparison"][method][stage] = prediction
                    row[f"{method}_{stage}_mm"] = prediction["predicted_mm"]
                    row[f"{method}_{stage}_error_mm"] = prediction["error_mm"]
            flat.append(row)
    if digest(workbook) != workbook_hash:
        raise RuntimeError("Source workbook changed while measuring; do not use a mixed snapshot")
    for item in data["images"]:
        if digest(source / item["name"]) != item["sha256"]:
            raise RuntimeError("Source image changed while measuring")
    data["meta"]["source_hashes_verified_after_run"] = True
    write_csv(out / "measurement_comparison.csv", flat, list(flat[0]))
    for method in ("legacy", "auto"):
        wide = [{"image": item["name"], **{p["name"]: p[method]["distance_px"] for p in item["points"]}} for item in data["images"]]
        write_csv(out / f"{method}_pixels.csv", wide, ["image", *POINT_NAMES])
    write_csv(out / "lab_references.csv", lab_provenance, list(lab_provenance[0]))
    json_write(out / "measurements.json", data)
    json_write(out / "validation.json", data["validation"])
    json_write(out / "source_config.json", config)
    template = (ROOT / "tools/templates/lab_line_compare.html").read_text(encoding="utf-8")
    if template.count("__LAB_LINE_DATA_JSON__") != 1:
        raise ValueError("Expected exactly one data placeholder in comparison template")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (out / "four_point_review.html").write_text(template.replace("__LAB_LINE_DATA_JSON__", payload), encoding="utf-8")
    write_report(out, data, flat)
    print(json.dumps({m: {stage: {k: result[k] for k in ("n", "mae", "rmse", "max_abs", "max_abs_at")}
                          for stage, result in values.items() if stage in ("same_batch", "leave_one_product_out")}
                      for m, values in data["validation"]["methods"].items()}, ensure_ascii=False, indent=2))
    print(f"Report: {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
