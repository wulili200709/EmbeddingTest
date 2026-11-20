"""Plot exploratory correspondence checks without changing the source workbook."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "comparison_statistics.json").read_text(encoding="utf-8"))
font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams["font.family"] = font.get_name()
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(2, 1, figsize=(12, 8.4), gridspec_kw={"height_ratios": [1.1, 1]})
fig.patch.set_facecolor("#f8fafc")
for ax in axes:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)

x = np.arange(14)
axes[0].bar(x-.18, data["models"]["identity"]["point_r"], .36, color="#9aabbc", label="表中同名点对应")
axes[0].bar(x+.18, data["models"]["rotate_180"]["point_r"], .36, color="#087f8c", label="假设相机朝向旋转180°后对应")
axes[0].axhline(0, color="#718096", linewidth=.7)
axes[0].set_xticks(x, data["regions"])
axes[0].set_ylim(-1, 1.15)
axes[0].set_ylabel("相关系数 r（每个点位仅6件）")
axes[0].set_xlabel("实验室点位编号")
axes[0].legend(loc="lower right", frameon=False)
axes[0].set_title("对应关系值得核对：部分点明显改善，仍有点位相关性很差", loc="left", fontsize=13, pad=14)

camera = np.asarray(data["camera"])
lab = np.asarray(data["lab1"])
series = [(lab[:,6], "实验室 U7", "#c87927"), (camera[:,6], "相机 U7（同名对应）", "#9aabbc"),
          (camera[:,7], "相机 L1（180°对应假设）", "#087f8c")]
for values, label, color in series:
    axes[1].plot(np.arange(1,7), values-values.mean(), "o-", linewidth=2, color=color, label=label)
axes[1].set_xticks(np.arange(1,7), [f"{i}号" for i in range(1,7)])
axes[1].set_ylabel("相对各自6件平均值的偏差（mm）")
axes[1].set_xlabel("产品编号（沿用工作表，独立性待确认）")
axes[1].legend(loc="lower left", frameon=False)
axes[1].set_title("第6件的低值出现在实验室 U7、相机 L1；此图仅比较变化趋势", loc="left", fontsize=13, pad=14)
fig.suptitle("相机与实验室：点位对应排查", x=.07, ha="left", fontsize=19, fontweight="bold")
fig.text(.07, .927, "0.0075 mm/px；实验室第一次数据；180°对应是待确认假设，未改动原表或测量算法。", fontsize=10, color="#435466")
fig.text(.07, .025, "各点只有6个配对观测；相关系数不能代替尺寸一致性、重复性或公差能力评价。", fontsize=10, color="#435466")
fig.subplots_adjust(left=.09, right=.97, top=.86, bottom=.105, hspace=.5)
fig.savefig(ROOT / "camera_lab_correspondence.png", dpi=160)
