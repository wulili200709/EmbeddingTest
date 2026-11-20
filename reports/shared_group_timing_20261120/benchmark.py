"""Read-only comparison of the current test order's independent/group ROI inference.

Only report files are written. Camera capture, alignment, UI and production
recipes are not changed. Both variants use the same image, crops and CPU ORT.
"""
from __future__ import annotations

import copy
import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnxruntime as ort
import torch
import algorithms.lazy_api as qr_core
from algorithms.embedding import OrtBackboneRunner, describe_backbone_runner
from application.algorithm_controller import AlgorithmController
from application.runtime.preview_frame import RuntimePreviewShape
from application.runtime_context import _predict_learning_items_batch_rows_from_frame
from common.algorithm_codes import learning_backbone_storage_code
from domain import load_inspection_items


def median(values):
    return float(statistics.median(values))


def measure(controller, runner, items, image, shapes):
    timing = {}
    started = time.perf_counter()
    rows = _predict_learning_items_batch_rows_from_frame(
        image_bgr=image, roi_shapes=shapes, items=items, match_ms=0.,
        algo=controller, feat_net=runner,
        load_embedding_model=lambda algorithm, model_key: controller.load_model_for_algorithm(
            algorithm, str(ORDER), model_key=model_key),
        timing_breakdown=timing,
    )
    timing["total_ms"] = (time.perf_counter() - started) * 1000.
    timing["npz_cache_entries"] = len(controller._embedding_model_cache)
    timing["predictions"] = {key: row["pred"] for key, row in rows.items()}
    return timing


def run_case(mode, image, shapes, items, backbone):
    controller = AlgorithmController()
    controller.load_params(str(ORDER / "product_params.json"))
    model_path = ROOT / ".cache" / "ort_backbones" / f"{learning_backbone_storage_code(backbone)}_cpu_v1.ort"
    started = time.perf_counter()
    # Use an existing asset directly, avoiding model export, download or cache writes.
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    runner = OrtBackboneRunner(
        session=session, input_name=session.get_inputs()[0].name,
        output_name=session.get_outputs()[0].name,
        input_shape=tuple(session.get_inputs()[0].shape), providers=("CPUExecutionProvider",),
        device="cpu", model_path=str(model_path), cpu_inference_chunk_size=2,
    )
    initialization_ms = (time.perf_counter() - started) * 1000.
    first = measure(controller, runner, items, image, shapes)
    warm = [measure(controller, runner, items, image, shapes) for _ in range(10)]
    # Training/session reload clears NPZ cache, but keeps an existing backbone alive.
    controller.load_params(str(ORDER / "product_params.json"))
    after_parameter_reload = measure(controller, runner, items, image, shapes)
    after_reload_warm = [measure(controller, runner, items, image, shapes) for _ in range(5)]
    return dict(mode=mode, initialization_ms=initialization_ms,
                runner=describe_backbone_runner(runner), input_shape=runner.input_shape,
                first=first, warm=warm, after_parameter_reload=after_parameter_reload,
                after_reload_warm=after_reload_warm)


ORDER = ROOT / ".qr_session" / "test"


def main():
    data = json.loads((ORDER / "session.json").read_text(encoding="utf-8-sig"))
    # This image already has both ROI geometries, avoiding writes from localization.
    image_path = ORDER / data["train_files"][0]
    annotation = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8-sig"))
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be decoded")
    items = [item for item in load_inspection_items(str(ORDER / "inspection_items.json"))
             if item.enabled and item.camera_id == "cam1"]
    if len(items) != 2 or any(item.model_source != "group" for item in items):
        raise ValueError("Expected the current two-ROI shared test order")
    shapes = tuple(RuntimePreviewShape(
        label=shape["label"], shape_type=shape.get("shape_type", "polygon"),
        points=tuple((float(x), float(y)) for x, y in shape["points"]))
        for shape in annotation["shapes"] if shape["label"] in {item.roi_label for item in items})
    independent = copy.deepcopy(items)
    for item in independent:
        item.model_source = "independent"
        item.task_group = ""
    controller = AlgorithmController()
    controller.load_params(str(ORDER / "product_params.json"))
    backbone = controller.current_learning_backbone("cam1")
    # Resolve lazy imports before either variant's timers start.
    _ = qr_core.predict_one_with_model
    _ = qr_core.load_register_model_npz
    metadata = {}
    for mode, selected in (("independent", independent), ("group", items)):
        models = []
        for item in selected:
            path = Path(controller._local_embedding_file(backbone, str(ORDER), item.effective_model_key))
            with np.load(path, allow_pickle=False) as model:
                models.append(dict(roi=item.roi_label, file=path.name, bytes=path.stat().st_size,
                                   ok_bank_shape=list(model["ok_bank"].shape),
                                   ng_bank_shape=list(model["ng_bank"].shape)))
        metadata[mode] = models
    runs = []
    for repetition in range(3):
        modes = [("independent", independent), ("group", items)]
        if repetition % 2:
            modes.reverse()
        for mode, selected in modes:
            result = run_case(mode, image, shapes, selected, backbone)
            result["repetition"] = repetition
            runs.append(result)
            print(json.dumps({"mode": mode, "repetition": repetition,
                              "init_ms": result["initialization_ms"],
                              "first_ms": result["first"]["total_ms"],
                              "warm_ms": median([row["total_ms"] for row in result["warm"]])}), flush=True)
            gc.collect()
    summary = {}
    numeric_stages = ("total_ms", "model_load_ms", "roi_preprocess_ms", "backbone_ms", "classify_ms")
    for mode in ("independent", "group"):
        selected = [run for run in runs if run["mode"] == mode]
        summary[mode] = {
            "initialization_ms": median([run["initialization_ms"] for run in selected]),
            "first": {key: median([run["first"][key] for run in selected]) for key in numeric_stages},
            "warm": {key: median([row[key] for run in selected for row in run["warm"]]) for key in numeric_stages},
            "after_parameter_reload": {key: median([run["after_parameter_reload"][key] for run in selected]) for key in numeric_stages},
        }
    report = dict(scope="ROI inference only; excludes image decoding, alignment, camera, UI, IO",
                  order="test", image=image_path.name, roi_count=len(items), backbone=backbone,
                  score_mode=controller.product_params.score_mode, cpu_torch_threads=torch.get_num_threads(),
                  gpu_available=torch.cuda.is_available(), models=metadata, summary=summary, runs=runs)
    output = Path(__file__).with_name("report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
