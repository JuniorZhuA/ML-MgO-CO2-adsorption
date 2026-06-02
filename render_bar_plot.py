#!/usr/bin/env python
"""纯渲染 SHAP Bar Plot — 仅读取缓存，零模型计算。

从 outputs/tables/ 加载缓存，调用原生 shap.plots.bar()，然后接管 matplotlib
进行顶刊审美定制：莫兰迪配色 + 无边框 + 虚线网格 + 清晰字体。
"""
import sys, io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── 路径 ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(ROOT, "outputs", "tables")
FIGURES = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIGURES, exist_ok=True)

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════
# 1. 纯缓存加载 → 构建 shap.Explanation
# ═══════════════════════════════════════════════════════════════════════
print("加载缓存...")
X_t = pd.read_csv(os.path.join(TABLES, "X_transformed_gbdt.csv"))
sv = pd.read_csv(os.path.join(TABLES, "shap_values_gbdt.csv"))
feature_names = list(X_t.columns)
print(f"  X_transformed: {X_t.shape}")
print(f"  shap_values:   {sv.shape}")

explanation = shap.Explanation(
    values=sv.values,
    data=X_t.values,
    feature_names=feature_names,
)
print("  shap.Explanation 构建完成")

# ═══════════════════════════════════════════════════════════════════════
# 2. 原生 bar plot → 接管审美
# ═══════════════════════════════════════════════════════════════════════
print("\n渲染原生 bar plot (max_display=15)...")
shap.plots.bar(explanation, max_display=15, show=False)

# 接管当前坐标轴
ax = plt.gca()
fig = plt.gcf()

# ── 审美定制 ───────────────────────────────────────────────────────────
MORANDI_SLATE = "#5b84b1"   # 莫兰迪灰蓝
GRID_COLOR = "#c0c0c0"      # 网格线浅灰

# (a) 替换条形颜色
for patch in ax.patches:
    patch.set_facecolor(MORANDI_SLATE)
    patch.set_edgecolor("white")
    patch.set_linewidth(0.3)
    patch.set_alpha(0.95)

# (b) 移除右侧/顶部/底部边框，仅保留左侧
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["left"].set_color("#cccccc")

# (c) X 轴虚线网格（置于底层）
ax.xaxis.grid(True, alpha=0.5, linestyle="--", linewidth=0.6, color=GRID_COLOR, zorder=0)
ax.set_axisbelow(True)  # 网格在条形下方

# (d) Y 轴特征名字体
ax.tick_params(axis="y", labelsize=12, pad=8)

# (e) X 轴标签
ax.set_xlabel("mean(|SHAP value|)", fontsize=13, fontweight="bold", labelpad=8)
ax.tick_params(axis="x", labelsize=9)

# (f) 标题
ax.set_title("GBDT SHAP Feature Importance (Global)", fontsize=14,
             fontweight="bold", pad=14)

# (g) 调整图形尺寸
fig.set_size_inches(9, 0.38 * 15 + 1.8)

# ═══════════════════════════════════════════════════════════════════════
# 3. 保存
# ═══════════════════════════════════════════════════════════════════════
output_path = os.path.join(FIGURES, "figure_7_shap_bar.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close("all")

print(f"\n[OK] 已保存: {output_path}")
print("    零模型计算，纯缓存渲染。")
