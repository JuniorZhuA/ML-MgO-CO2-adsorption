"""纯绘图脚本: Train/Test 同轴重叠箱线图。

从 50_splits_raw_results.csv 读取数据, 绝不执行任何模型训练。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.config import TABLES, FIGURES

# ── 1. 读取数据 ────────────────────────────────────────────────────────
RAW_CSV = TABLES / "50_splits_raw_results.csv"
try:
    df = pd.read_csv(RAW_CSV)
except FileNotFoundError:
    print(f"错误: 找不到 {RAW_CSV} 数据文件。")
    exit()

# ── 2. 配色与画布 ──────────────────────────────────────────────────────
COLOR_TRAIN = "#5b84b1"
COLOR_TEST = "#fc766a"
FACE_TRAIN = "#a8c5e0"
FACE_TEST = "#fdc8c0"

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
metrics = ["R2", "RMSE", "MAE"]
model_order = df["Model"].unique().tolist()
n_models = len(model_order)
x_pos = np.arange(n_models)

# 统一的线宽参数
LW = 1.2
MW = 1.8

for ax_i, metric in enumerate(metrics):
    ax = axes[ax_i]
    subset = df[df["Metric"] == metric]

    for i, model in enumerate(model_order):
        train_data = subset[(subset["Model"] == model) & (subset["Dataset"] == "Train")]["Value"].values
        test_data = subset[(subset["Model"] == model) & (subset["Dataset"] == "Test")]["Value"].values

        # Train 箱 (宽, 半透明, 后层)
        bp = ax.boxplot(
            [train_data], positions=[x_pos[i]], widths=0.55,
            patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor=FACE_TRAIN, color=COLOR_TRAIN, linewidth=LW, alpha=0.55),
            whiskerprops=dict(color=COLOR_TRAIN, linewidth=LW),
            capprops=dict(color=COLOR_TRAIN, linewidth=LW),
            medianprops=dict(color="#2d5a7a", linewidth=MW),
            flierprops=dict(marker="o", markerfacecolor=COLOR_TRAIN, markersize=2,
                            markeredgecolor="none", alpha=0.3),
        )

        # Test 箱 (窄, 不透明, 前层)
        ax.boxplot(
            [test_data], positions=[x_pos[i]], widths=0.30,
            patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor=FACE_TEST, color=COLOR_TEST, linewidth=LW, alpha=0.85),
            whiskerprops=dict(color=COLOR_TEST, linewidth=LW),
            capprops=dict(color=COLOR_TEST, linewidth=LW),
            medianprops=dict(color="#7a2a1a", linewidth=MW),
            flierprops=dict(marker="o", markerfacecolor=COLOR_TEST, markersize=2,
                            markeredgecolor="none", alpha=0.3),
        )

    # 横坐标
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, rotation=0, ha="center", fontsize=10)
    ax.set_xlabel("")

    # Y轴
    if metric == "R2":
        all_vals = subset["Value"].values
        ax.set_ylim(max(0, all_vals.min() - 0.05), 1.0)
        ax.set_ylabel("R²", fontsize=12)
    elif metric == "RMSE":
        ax.set_ylim(bottom=0)
        ax.set_ylabel("RMSE", fontsize=12)
    else:
        ax.set_ylim(bottom=0)
        ax.set_ylabel("MAE", fontsize=12)

    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

# ── 3. 全局图例 ────────────────────────────────────────────────────────
legend_elements = [
    Patch(facecolor=FACE_TRAIN, edgecolor=COLOR_TRAIN, linewidth=LW,
          alpha=0.55, label="Train"),
    Patch(facecolor=FACE_TEST, edgecolor=COLOR_TEST, linewidth=LW,
          alpha=0.85, label="Test"),
]
fig.legend(
    handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.06),
    ncol=2, frameon=False, fontsize=14,
)

plt.tight_layout()
out_path_png = FIGURES / "model_performance_boxplot_final.png"
out_path_svg = FIGURES / "model_performance_boxplot_final.svg"
plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
plt.savefig(out_path_svg, bbox_inches="tight")
plt.close()
print(f"同轴箱线图完成! 已保存: {out_path_png}")
print(f"SVG 已保存: {out_path_svg}")
