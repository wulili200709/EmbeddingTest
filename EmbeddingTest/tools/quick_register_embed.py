"""
quick_register_embed.py

原本的单文件脚本保留为 CLI 示例，但核心逻辑已抽到 `qr_core.py`，
后续 GUI 也会复用这套核心实现。
"""

import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    root_str = str(Path(__file__).resolve().parents[1])
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from algorithms.api import (
    embed_one,
    load_backbone,
    load_images,
    predict_one,
    score_topk,
)

print("PYTHON =", sys.executable)

DEVICE = "cuda"  # 这里仅用于打印；真正 device 在 qr_core 内部自适应

# 选择 backbone：efficientnet_b0 / mobilenet_v3_small / mobilenet_v3_large
BACKBONE = "efficientnet_b0"
# Scoring: "proto" or "topk"
SCORE_MODE = "proto"
MARGIN = 0.02
TOPK = 3

def main():
    feat_net, _ = load_backbone(BACKBONE)

    ok_files = load_images("OK")
    ng_files = load_images("NG")
    if not ok_files or not ng_files:
        raise RuntimeError("OK/NG 文件夹里都要有图片")

    # 每张图用自己的 labelme ROI 提 embedding
    ok_emb = np.stack([embed_one(p, feat_net, label_name="roi") for p in ok_files])
    ng_emb = np.stack([embed_one(p, feat_net, label_name="roi") for p in ng_files])

    # 注册：OK/NG 原型
    ok_proto = ok_emb.mean(axis=0, keepdims=True)
    ok_proto = ok_proto / np.linalg.norm(ok_proto, axis=1, keepdims=True)
    ng_proto = ng_emb.mean(axis=0, keepdims=True)
    ng_proto = ng_proto / np.linalg.norm(ng_proto, axis=1, keepdims=True)

    ok_bank = ok_emb
    ng_bank = ng_emb

    if SCORE_MODE == "proto":
        ok_diff = (ok_emb @ ok_proto.T).ravel() - (ok_emb @ ng_proto.T).ravel()
        ng_diff = (ng_emb @ ok_proto.T).ravel() - (ng_emb @ ng_proto.T).ravel()
    elif SCORE_MODE == "topk":
        k_ok = min(TOPK, len(ok_bank))
        k_ng = min(TOPK, len(ng_bank))
        ok_diff = np.array([score_topk(e, ok_bank, k=k_ok) - score_topk(e, ng_bank, k=k_ng) for e in ok_emb])
        ng_diff = np.array([score_topk(e, ok_bank, k=k_ok) - score_topk(e, ng_bank, k=k_ng) for e in ng_emb])
    else:
        raise ValueError("Unknown score mode")

    print(f"[Backbone] {BACKBONE}")
    print(f"[Score] mode={SCORE_MODE} margin={MARGIN:.4f}")
    if SCORE_MODE == "topk":
        print(f"[TopK] ok_k={k_ok} ng_k={k_ng}")
    tp = int(np.sum(ok_diff >= MARGIN))
    fn = int(ok_diff.size - tp)
    tn = int(np.sum(ng_diff < MARGIN))
    fp = int(ng_diff.size - tn)
    print(f"TP(OK->OK)={tp}  FN(OK->NG)={fn}")
    print(f"TN(NG->NG)={tn}  FP(NG->OK)={fp}")

    # 可选：对 test 输出（test 也需要同名 json）
    # 兼容 TEST / test
    test_dir = "TEST" if os.path.isdir("TEST") else "test"
    test_files = load_images(test_dir)
    if test_files:
        print("\n[Test Predictions]")
        for p in test_files:
            e = embed_one(p, feat_net, label_name="roi")
            pred, diff, sim_ok, sim_ng = predict_one(
                e, ok_proto, ng_proto, ok_bank, ng_bank,
                mode=SCORE_MODE, margin=MARGIN, topk=TOPK,
            )
            print(f"{pred}\t diff={diff:.4f}\t ok={sim_ok:.4f}\t ng={sim_ng:.4f}\t {os.path.basename(p)}")

if __name__ == "__main__":
    main()
