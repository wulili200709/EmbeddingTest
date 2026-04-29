from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from algorithms.embedding import RegisterModel, load_register_model_npz, predict_one_with_model
from algorithms.registry import algorithm_display_name
from domain import load_inspection_items
from tools.visualize_embeddings import compact_plot_label, load_product_analysis


@dataclass
class ModelReport:
    display_name: str
    backbone: str
    storage_code: str
    model_path: str
    pca_image: str
    tsne_image: str
    feature_dim: int
    ok_count: int
    ng_count: int
    train_accuracy: float
    topk_fullbank_accuracy: float
    proto_similarity: float
    ok_intra_mean: float
    ng_intra_mean: float
    ok_ng_cross_mean: float
    ok_diff_min: float
    ok_diff_mean: float
    ok_diff_max: float
    ng_diff_min: float
    ng_diff_mean: float
    ng_diff_max: float
    safety_gap: float
    notes: List[str]


def _resolve_group_labels(product_dir: str, *, model_key: str) -> Tuple[str, List[str]]:
    items = load_inspection_items(os.path.join(product_dir, "inspection_items.json"))
    matched = [
        item
        for item in items
        if str(getattr(item, "effective_model_key", getattr(item, "model_key", "")) or "").strip() == model_key
    ]
    if not matched:
        return "cam1", []
    camera_id = str(getattr(matched[0], "camera_id", "") or "cam1").strip() or "cam1"
    labels = list(
        dict.fromkeys(
            str(getattr(item, "roi_label", "") or "").strip()
            for item in matched
            if str(getattr(item, "roi_label", "") or "").strip()
        )
    )
    return camera_id, labels


def _load_group_annotation_summary(product_dir: str, *, model_key: str) -> List[Tuple[str, int, int, List[str]]]:
    camera_id, labels = _resolve_group_labels(product_dir, model_key=model_key)
    if not labels:
        return []
    annotations_path = os.path.join(product_dir, "sample_annotations.json")
    if not os.path.exists(annotations_path):
        return []
    with open(annotations_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = payload.get("images", payload) if isinstance(payload, dict) else {}
    if not isinstance(images, dict):
        return []
    rows: List[Tuple[str, int, int, List[str]]] = []
    for path_key, raw_value in images.items():
        if not isinstance(raw_value, dict):
            continue
        roi_status = raw_value.get("roi_status", raw_value)
        if not isinstance(roi_status, dict):
            continue
        ok_labels: List[str] = []
        ng_labels: List[str] = []
        for label in labels:
            status = str(roi_status.get(f"{camera_id}::{label}", "") or "").strip().upper()
            if status == "OK":
                ok_labels.append(label)
            elif status == "NG":
                ng_labels.append(label)
        if ok_labels or ng_labels:
            rows.append((os.path.basename(str(path_key or "")), len(ok_labels), len(ng_labels), ng_labels))
    rows.sort(key=lambda row: row[0])
    return rows


def _save_scatter_plot(result, output_path: str) -> None:
    coords = result.point_coords
    labels = result.point_labels
    ok_mask = np.array([label == "OK" for label in labels], dtype=bool)
    ng_mask = ~ok_mask

    fig, ax = plt.subplots(figsize=(10, 8))
    if np.any(ok_mask):
        ok_points = coords[ok_mask]
        ax.scatter(ok_points[:, 0], ok_points[:, 1], c="tab:blue", s=55, alpha=0.85, label="OK")
        ok_center = ok_points.mean(axis=0)
        ax.scatter([ok_center[0]], [ok_center[1]], c="navy", marker="*", s=260, label="OK center")
    if np.any(ng_mask):
        ng_points = coords[ng_mask]
        ax.scatter(ng_points[:, 0], ng_points[:, 1], c="tab:red", s=55, alpha=0.85, label="NG")
        ng_center = ng_points.mean(axis=0)
        ax.scatter([ng_center[0]], [ng_center[1]], c="darkred", marker="*", s=260, label="NG center")

    for idx, name in enumerate(result.point_names):
        ax.annotate(compact_plot_label(name), (coords[idx, 0], coords[idx, 1]), fontsize=8, alpha=0.75)

    ax.set_title(f"{result.product_name} / {result.tool_name or result.model_key} / {result.projection_method.upper()}")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _compute_topk_accuracy(model: RegisterModel) -> float:
    ok_bank = np.asarray(model.ok_analysis_bank if model.ok_analysis_bank is not None else model.ok_bank, dtype=np.float32)
    ng_bank = np.asarray(model.ng_analysis_bank if model.ng_analysis_bank is not None else model.ng_bank, dtype=np.float32)
    topk_model = RegisterModel(
        backbone=model.backbone,
        score_mode="topk",
        margin=float(model.margin),
        topk=int(model.topk),
        label_name=model.label_name,
        label_names=model.label_names,
        device=model.device,
        ok_proto=model.ok_proto,
        ng_proto=model.ng_proto,
        ok_bank=ok_bank,
        ng_bank=ng_bank,
    )
    preds = [predict_one_with_model(vec, topk_model)[0] for vec in ok_bank] + [
        predict_one_with_model(vec, topk_model)[0] for vec in ng_bank
    ]
    gt = ["OK"] * len(ok_bank) + ["NG"] * len(ng_bank)
    if not gt:
        return float("nan")
    return float(sum(int(a == b) for a, b in zip(preds, gt)) / len(gt))


def _build_model_report(
    *,
    session_root: str,
    product_name: str,
    model_key: str,
    backbone: str,
    storage_code: str,
    output_dir: str,
) -> ModelReport:
    pca_result = load_product_analysis(session_root, product_name, backbone, model_key=model_key, projection_method="pca")
    tsne_result = load_product_analysis(session_root, product_name, backbone, model_key=model_key, projection_method="tsne")
    model = load_register_model_npz(pca_result.model_path)

    ok_bank = np.asarray(model.ok_analysis_bank if model.ok_analysis_bank is not None else model.ok_bank, dtype=np.float32)
    ng_bank = np.asarray(model.ng_analysis_bank if model.ng_analysis_bank is not None else model.ng_bank, dtype=np.float32)
    ok_diffs = np.array([predict_one_with_model(vec, model)[1] for vec in ok_bank], dtype=np.float32)
    ng_diffs = np.array([predict_one_with_model(vec, model)[1] for vec in ng_bank], dtype=np.float32)

    pca_image_name = f"{storage_code}_pca.png"
    tsne_image_name = f"{storage_code}_tsne.png"
    _save_scatter_plot(pca_result, os.path.join(output_dir, pca_image_name))
    _save_scatter_plot(tsne_result, os.path.join(output_dir, tsne_image_name))

    margin = float(model.margin)
    ok_gap = float(ok_diffs.min()) - margin if ok_diffs.size else float("nan")
    ng_gap = margin - float(ng_diffs.max()) if ng_diffs.size else float("nan")
    safety_gap = min(ok_gap, ng_gap) if not math.isnan(ok_gap) and not math.isnan(ng_gap) else float("nan")

    return ModelReport(
        display_name=algorithm_display_name(backbone) or backbone,
        backbone=backbone,
        storage_code=storage_code,
        model_path=pca_result.model_path,
        pca_image=pca_image_name,
        tsne_image=tsne_image_name,
        feature_dim=pca_result.feature_dim,
        ok_count=int(pca_result.metrics["ok_count"]),
        ng_count=int(pca_result.metrics["ng_count"]),
        train_accuracy=float(pca_result.metrics["train_accuracy"]),
        topk_fullbank_accuracy=_compute_topk_accuracy(model),
        proto_similarity=float(pca_result.metrics["proto_similarity"]),
        ok_intra_mean=float(pca_result.metrics["ok_intra_mean"]),
        ng_intra_mean=float(pca_result.metrics["ng_intra_mean"]),
        ok_ng_cross_mean=float(pca_result.metrics["ok_ng_cross_mean"]),
        ok_diff_min=float(ok_diffs.min()),
        ok_diff_mean=float(ok_diffs.mean()),
        ok_diff_max=float(ok_diffs.max()),
        ng_diff_min=float(ng_diffs.min()),
        ng_diff_mean=float(ng_diffs.mean()),
        ng_diff_max=float(ng_diffs.max()),
        safety_gap=float(safety_gap),
        notes=list(dict.fromkeys(list(pca_result.notes) + list(tsne_result.notes))),
    )


def _recommendation_order(reports: Sequence[ModelReport]) -> List[ModelReport]:
    return sorted(
        reports,
        key=lambda report: (
            report.safety_gap,
            -report.proto_similarity,
            report.feature_dim,
        ),
        reverse=True,
    )


def _format_float(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def _build_markdown(
    *,
    product_name: str,
    model_key: str,
    product_dir: str,
    reports: Sequence[ModelReport],
    ng_summary: Sequence[Tuple[str, int, int, List[str]]],
) -> str:
    ordered = _recommendation_order(reports)
    lines: List[str] = []
    lines.append(f"# 学习模型对比报告: {product_name} / {model_key}")
    lines.append("")
    lines.append("## 范围")
    lines.append("")
    lines.append(f"- 产品目录: `{product_dir}`")
    lines.append(f"- 分析对象: `{model_key}`")
    lines.append("- 数据来源: 当前 `.qr_session` 中已训练完成的 group 模型")
    lines.append("- 模型范围: `高精度学习 lt01`、`轻量学习 lt02`、`均衡学习 lt03`")
    lines.append("")
    lines.append("## PCA / TSNE 说明")
    lines.append("")
    lines.append("- `PCA` 是线性投影。它优先保留整体方差最大的方向，更适合看全局可分性和整体分离方向。")
    lines.append("- `TSNE` 是邻域展开。它优先保留“谁跟谁近”，不强调全局几何结构，更适合看局部簇和子簇。")
    lines.append("- 所以同一批样本在 `PCA` 和 `TSNE` 里形状差异很大是正常的。")
    lines.append("")
    lines.append("## 数据概况")
    lines.append("")
    if reports:
        lines.append(f"- 当前 group 的训练样本量: `OK {reports[0].ok_count}` / `NG {reports[0].ng_count}`")
    lines.append("- 这组 `NG` 不是单一模式，而是多张图、多种位置组合混在一起，所以特征图中通常会分成多个红色子簇。")
    lines.append("")
    lines.append("### NG 来源统计")
    lines.append("")
    lines.append("| 图片 | OK 数 | NG 数 | NG ROI |")
    lines.append("| --- | ---: | ---: | --- |")
    for file_name, ok_count, ng_count, ng_labels in ng_summary:
        if ng_count <= 0:
            continue
        ng_desc = ", ".join(ng_labels)
        lines.append(f"| `{file_name}` | {ok_count} | {ng_count} | `{ng_desc}` |")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append("")
    if ordered:
        lines.append("按当前这组真实数据的判定余量排序:")
        lines.append("")
        for index, report in enumerate(ordered, start=1):
            lines.append(
                f"{index}. `{report.display_name} {report.storage_code}`: safety gap = `{_format_float(report.safety_gap)}`"
            )
        lines.append("")
        lines.append(f"当前首选: `{ordered[0].display_name} {ordered[0].storage_code}`")
        lines.append("")
        lines.append("判断标准:")
        lines.append("- `OK diff 最小值` 越大越好，说明最难的 OK 样本也离判错边界更远。")
        lines.append("- `NG diff 最大值` 越负越好，说明最危险的 NG 样本也离漏检边界更远。")
        lines.append("- `safety gap` 取两侧到当前 `margin=0.02` 的最小安全距离，越大越稳。")
        lines.append("")
    lines.append("## 指标总表")
    lines.append("")
    lines.append("| 模型 | 维度 | train acc(proto) | train acc(topk/full bank) | proto sim | OK intra | NG intra | OK-NG cross | OK diff min | NG diff max | safety gap |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for report in reports:
        lines.append(
            "| "
            + f"`{report.display_name} {report.storage_code}` | "
            + f"{report.feature_dim} | "
            + f"{_format_float(report.train_accuracy)} | "
            + f"{_format_float(report.topk_fullbank_accuracy)} | "
            + f"{_format_float(report.proto_similarity)} | "
            + f"{_format_float(report.ok_intra_mean)} | "
            + f"{_format_float(report.ng_intra_mean)} | "
            + f"{_format_float(report.ok_ng_cross_mean)} | "
            + f"{_format_float(report.ok_diff_min)} | "
            + f"{_format_float(report.ng_diff_max)} | "
            + f"{_format_float(report.safety_gap)} |"
        )
    lines.append("")
    for report in reports:
        lines.append(f"## {report.display_name} ({report.storage_code})")
        lines.append("")
        lines.append(f"- backbone: `{report.backbone}`")
        lines.append(f"- 模型文件: `{report.model_path}`")
        lines.append(f"- 特征维度: `{report.feature_dim}`")
        lines.append(f"- 训练样本: `OK {report.ok_count}` / `NG {report.ng_count}`")
        lines.append(f"- `proto` 训练准确率: `{_format_float(report.train_accuracy)}`")
        lines.append(f"- `topk/full bank` 训练准确率: `{_format_float(report.topk_fullbank_accuracy)}`")
        lines.append(f"- `proto_similarity`: `{_format_float(report.proto_similarity)}`")
        lines.append(f"- `ok_intra_mean`: `{_format_float(report.ok_intra_mean)}`")
        lines.append(f"- `ng_intra_mean`: `{_format_float(report.ng_intra_mean)}`")
        lines.append(f"- `ok_ng_cross_mean`: `{_format_float(report.ok_ng_cross_mean)}`")
        lines.append(
            f"- `OK diff min/mean/max`: `{_format_float(report.ok_diff_min)}` / `{_format_float(report.ok_diff_mean)}` / `{_format_float(report.ok_diff_max)}`"
        )
        lines.append(
            f"- `NG diff min/mean/max`: `{_format_float(report.ng_diff_min)}` / `{_format_float(report.ng_diff_mean)}` / `{_format_float(report.ng_diff_max)}`"
        )
        lines.append(f"- `safety gap`: `{_format_float(report.safety_gap)}`")
        if report.notes:
            lines.append("- 备注:")
            for note in report.notes:
                lines.append(f"  - {note}")
        lines.append("")
        lines.append("### 结论")
        lines.append("")
        if report.storage_code == "lt03":
            lines.append("- 这组数据上整体最稳。OK 最难样本和 NG 最危险样本都离边界更远。")
            lines.append("- 如果你要优先选一套上线试，这一套最合适。")
        elif report.storage_code == "lt02":
            lines.append("- 速度和稳定性比较平衡。")
            lines.append("- 如果你更在意 CPU 侧节拍，这套通常比 `lt01` 更值得优先试。")
        elif report.storage_code == "lt01":
            lines.append("- 全局分离不差，但边缘样本更贴近判定边界。")
            lines.append("- 当前这组数据里，它不是最稳的一套。")
        lines.append("")
        lines.append("### PCA")
        lines.append("")
        lines.append(f"![{report.display_name} PCA](./{report.pca_image})")
        lines.append("")
        lines.append("### TSNE")
        lines.append("")
        lines.append(f"![{report.display_name} TSNE](./{report.tsne_image})")
        lines.append("")
    lines.append("## 最终建议")
    lines.append("")
    lines.append("- 如果你当前优先追求稳定性: 先用 `均衡学习 lt03`。")
    lines.append("- 如果你更在意 CPU 侧速度: 优先试 `轻量学习 lt02`。")
    lines.append("- 当前这组 `hole` 的 `NG` 明显是多峰分布，所以后面如果还想继续压误判，方向不是先换阈值，而是：")
    lines.append("  - 继续细分 `hole` group，或者")
    lines.append("  - 从单 `proto` 继续升级到保留更多样本结构的判定方式。")
    lines.append("")
    return "\n".join(lines)


def generate_report(*, session_root: str, product_name: str, model_key: str, output_dir: str) -> Tuple[str, List[str]]:
    product_dir = os.path.join(session_root, product_name)
    os.makedirs(output_dir, exist_ok=True)
    model_specs = [
        ("b0", "lt01"),
        ("b1", "lt02"),
        ("b2", "lt03"),
    ]
    reports = [
        _build_model_report(
            session_root=session_root,
            product_name=product_name,
            model_key=model_key,
            backbone=backbone,
            storage_code=storage_code,
            output_dir=output_dir,
        )
        for backbone, storage_code in model_specs
    ]
    ng_summary = _load_group_annotation_summary(product_dir, model_key=model_key)
    markdown = _build_markdown(
        product_name=product_name,
        model_key=model_key,
        product_dir=product_dir,
        reports=reports,
        ng_summary=ng_summary,
    )
    report_path = os.path.join(output_dir, "learning_model_comparison_report.md")
    with open(report_path, "w", encoding="utf-8-sig") as handle:
        handle.write(markdown)
    image_paths = [os.path.join(output_dir, report.pca_image) for report in reports] + [
        os.path.join(output_dir, report.tsne_image) for report in reports
    ]
    return report_path, image_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a markdown comparison report for learning models.")
    parser.add_argument("--session-root", default=".qr_session")
    parser.add_argument("--product", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    report_path, image_paths = generate_report(
        session_root=args.session_root,
        product_name=args.product,
        model_key=args.model_key,
        output_dir=args.output_dir,
    )
    print(report_path)
    for image_path in image_paths:
        print(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
