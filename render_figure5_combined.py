#!/usr/bin/env python
"""
Figure 5: SHAP Beeswarm — shap 原生引擎, 15 特征, 黑色加粗字体

输出: outputs/figures/figure_5_beeswarm.png
"""

import sys, io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import shap

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]

N_TOP = 15

# ── 加载缓存 ─────────────────────────────────────────────────────────
print(f"[1/3] 加载缓存...")
X_t  = pd.read_csv(TABLES / "X_transformed_gbdt.csv")
sv   = pd.read_csv(TABLES / "shap_values_gbdt.csv")
feat = list(X_t.columns)
X_arr, sv_arr = X_t.values, sv.values
print(f"    特征: {len(feat)}, 样本: {len(X_t)}")

# ── Top 15 子集 ─────────────────────────────────────────────────────
mean_abs = np.abs(sv_arr).mean(axis=0)
top_idx  = np.argsort(mean_abs)[::-1][:N_TOP]
top_names = [feat[i] for i in top_idx]
print(f"[2/3] Top {N_TOP}: {', '.join(top_names[:4])} ...")

explanation = shap.Explanation(
    values=sv_arr[:, top_idx],
    data=X_arr[:, top_idx],
    feature_names=top_names,
)

# ── 原生 beeswarm ────────────────────────────────────────────────────
print(f"[3/3] 渲染 + 保存...")
shap.plots.beeswarm(explanation, max_display=N_TOP, show=False)

fig = plt.gcf()
ax  = plt.gca()

# ── 审美: 全部黑色加粗 ───────────────────────────────────────────────
ax.set_xlabel("SHAP value (impact on model output)",
              fontsize=13, fontweight="bold", color="black", labelpad=4)
ax.tick_params(axis="x", labelsize=10, colors="black")
ax.tick_params(axis="y", labelsize=11.5, colors="black", length=0, pad=10)

# Y 轴特征名
labels = [t.get_text() for t in ax.get_yticklabels()]
ax.set_yticklabels(labels, fontweight="bold", color="black")

# X 轴刻度
for lbl in ax.get_xticklabels():
    lbl.set_fontweight("bold")
    lbl.set_color("black")

# 边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("black")
ax.spines["bottom"].set_linewidth(0.8)

# 水平网格
ax.set_axisbelow(True)
ax.xaxis.grid(True, alpha=0.12, linestyle="--", linewidth=0.5, color="#cccccc")
ax.yaxis.grid(False)

# 标题
ax.set_title(ax.get_title(), fontsize=14, fontweight="bold", color="black", pad=8)

# 颜色条标签
for cax in fig.axes:
    if cax is not ax:
        cax.tick_params(labelsize=9, colors="black")
        for cl in cax.get_yticklabels():
            cl.set_fontweight("bold")
            cl.set_color("black")
        for lbl_attr in [cax.get_xlabel(), cax.get_ylabel()]:
            if lbl_attr:
                cax.set_ylabel(lbl_attr, fontweight="bold", color="black", fontsize=10)

# ── 保存 ─────────────────────────────────────────────────────────────
fig.set_size_inches(11, 7.5)
plt.subplots_adjust(left=0.23, right=0.90, top=0.96, bottom=0.08, hspace=0)
out = FIGURES / "figure_5_beeswarm.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.close("all")
print(f"    ✓ {out}")
