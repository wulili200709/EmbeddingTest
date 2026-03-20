"""
test_hist_embed.py

用颜色直方图（HSV）替代 backbone embedding，测试密封圈有无检测。
不依赖 torch，只需 OpenCV。

用法：
  直接修改下面的 OK_FOLDER / NG_FOLDER / ROI_LABEL，
  然后运行：python test_hist_embed.py
"""

import glob
import os
import cv2
import numpy as np
import json

# ============================================================
# ★ 修改这里：指定你的 OK / NG 图片目录 和 ROI label 名
# ============================================================
PRODUCT_DIR = r"C:\Users\goney\Desktop\MatchTemplate2\EmbeddingTest\6"
ROI_LABEL   = "roi1"          # labelme json 里的 label 名
TEST_DIR    = r"C:\Users\goney\Desktop\MatchTemplate2\EmbeddingTest\6\test"  # 测试文件夹，设为 None 则对训练集本身评估
# ============================================================


def load_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)


def read_roi_xywh(img_path: str, label: str):
    """从同名 labelme json 读取 ROI"""
    base, _ = os.path.splitext(img_path)
    jpath = base + ".json"
    if not os.path.exists(jpath):
        raise FileNotFoundError(f"缺少 json: {jpath}")
    with open(jpath, encoding="utf-8") as f:
        data = json.load(f)
    for s in data.get("shapes", []):
        if s.get("label") != label:
            continue
        pts = np.array(s["points"], dtype=np.float32)
        x = int(round(float(pts[:, 0].min())))
        y = int(round(float(pts[:, 1].min())))
        w = int(round(float(pts[:, 0].max() - pts[:, 0].min())))
        h = int(round(float(pts[:, 1].max() - pts[:, 1].min())))
        return x, y, max(1, w), max(1, h)
    raise RuntimeError(f"未找到 label={label} 的 ROI: {jpath}")


def hist_embed(img_path: str, roi_xywh) -> np.ndarray:
    """提取 ROI 的 HSV 饱和度(S)直方图作为 64 维 embedding
    仅用 S 通道：灰色S≈0~30，橙色S≈130~200，对光照/角度变化不敏感
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    x, y, w, h = roi_xywh
    H_img, W_img = img.shape[:2]
    x = max(0, min(x, W_img - 1))
    y = max(0, min(y, H_img - 1))
    w = max(1, min(w, W_img - x))
    h = max(1, min(h, H_img - y))

    roi = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 只用 S（饱和度）：64 桶，抗光照/角度变化
    hist_s = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()

    feat = hist_s / (hist_s.sum() + 1e-6)   # 归一化，与ROI大小无关
    return feat.astype(np.float32)


def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    """巴氏距离：越小越相似（0=完全相同）"""
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def main():
    ok_dir = os.path.join(PRODUCT_DIR, "OK")
    ng_dir = os.path.join(PRODUCT_DIR, "NG")

    ok_files = load_images(ok_dir)
    ng_files = load_images(ng_dir)

    if not ok_files:
        print(f"[错误] OK 目录没有图片: {ok_dir}")
        return
    if not ng_files:
        print(f"[错误] NG 目录没有图片: {ng_dir}")
        return

    print(f"OK: {len(ok_files)} 张   NG: {len(ng_files)} 张\n")

    # ---------- 训练：计算 OK / NG 原型直方图 ----------
    ok_feats, ng_feats = [], []

    for p in ok_files:
        try:
            roi = read_roi_xywh(p, ROI_LABEL)
            ok_feats.append(hist_embed(p, roi))
        except Exception as e:
            print(f"  [跳过 OK] {os.path.basename(p)}: {e}")

    for p in ng_files:
        try:
            roi = read_roi_xywh(p, ROI_LABEL)
            ng_feats.append(hist_embed(p, roi))
        except Exception as e:
            print(f"  [跳过 NG] {os.path.basename(p)}: {e}")

    if not ok_feats or not ng_feats:
        print("[错误] OK/NG 都需要至少1张有效样本")
        return

    ok_proto = np.mean(ok_feats, axis=0).astype(np.float32)
    ng_proto = np.mean(ng_feats, axis=0).astype(np.float32)

    # ---------- 评估目标 ----------
    if TEST_DIR:
        # 对 test 文件夹中的所有图片做预测（无 GT，只输出结果）
        test_files = load_images(TEST_DIR)
        print(f"\n测试集：{len(test_files)} 张\n")
        print(f"{'文件':<45} {'Pred':<5} {'dist_ok':>8} {'dist_ng':>8}")
        print("-" * 72)
        for img_path in test_files:
            try:
                roi = read_roi_xywh(img_path, ROI_LABEL)
                feat = hist_embed(img_path, roi)
                d_ok = bhattacharyya(feat, ok_proto)
                d_ng = bhattacharyya(feat, ng_proto)
                pred = "OK" if d_ok < d_ng else "NG"
                name = os.path.basename(img_path)
                print(f"{name:<45} {pred:<5} {d_ok:>8.4f} {d_ng:>8.4f}")
            except Exception as e:
                name = os.path.basename(img_path)
                print(f"{name:<45} [跳过: {e}]")
    else:
        # 对训练集本身评估准确率
        all_files = [(p, "OK") for p in ok_files] + [(p, "NG") for p in ng_files]
        correct = 0
        print(f"{'文件':<40} {'GT':<4} {'Pred':<4} {'dist_ok':>8} {'dist_ng':>8}")
        print("-" * 70)
        for img_path, gt in all_files:
            try:
                roi = read_roi_xywh(img_path, ROI_LABEL)
                feat = hist_embed(img_path, roi)
                d_ok = bhattacharyya(feat, ok_proto)
                d_ng = bhattacharyya(feat, ng_proto)
                pred = "OK" if d_ok < d_ng else "NG"
                ok_mark = "✓" if pred == gt else "✗"
                if pred == gt:
                    correct += 1
                name = os.path.basename(img_path)
                print(f"{name:<40} {gt:<4} {pred:<4} {d_ok:>8.4f} {d_ng:>8.4f}  {ok_mark}")
            except Exception as e:
                print(f"{os.path.basename(img_path):<40} [跳过]: {e}")
        total = len(all_files)
        print("-" * 70)
        print(f"\n准确率: {correct}/{total} = {correct/total*100:.1f}%")
        print("\n说明：dist_ok/dist_ng 越小=越相似。OK图应 dist_ok < dist_ng")


if __name__ == "__main__":
    main()
