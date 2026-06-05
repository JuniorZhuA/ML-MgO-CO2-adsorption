#!/usr/bin/env python
"""纯渲染脚本 — 从 outputs/tables/ 缓存读取，零模型计算。

生成:
  1. Figure 5 Native Beeswarm  — shap.plots.beeswarm (原生)
  2. Model Consistency Scatter  — TabPFN vs GBDT 排名散点图
"""
import sys, io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import shap

# ── 路径 ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(ROOT, "outputs", "tables")
FIGURES = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIGURES, exist_ok=True)

# UTF-8
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── 加载缓存 ────────────────────────────────────────────────────────────
print("加载缓存数据...")
X_t = pd.read_csv(os.path.join(TABLES, "X_transformed_gbdt.csv"))
sv = pd.read_csv(os.path.join(TABLES, "shap_values_gbdt.csv"))
cons = pd.read_csv(os.path.join(TABLES, "consistency_comparison.csv"))

feature_names = list(X_t.columns)
print(f"  X_transformed: {X_t.shape}")
print(f"  shap_values:   {sv.shape}")
print(f"  consistency:   {cons.shape[0]} rows")

# ── 加载 ρ 值（从 consistency.csv 独立计算，确保准确）──────────────────
valid = cons[cons["tabpfn_rank"].notna()].copy()
rho, pval = spearmanr(valid["importance_mean"], valid["shap_importance_mean"])
print(f"  Spearman ρ = {rho:.4f} (p = {pval:.4f}), n = {len(valid)}")

# ========================================================================
# Figure 1: 原生 SHAP Beeswarm
# ========================================================================
print("\n" + "=" * 60)
print("Figure 1: Native SHAP Beeswarm (shap.plots.beeswarm)")
print("=" * 60)

# 构建 shap.Explanation 对象
explanation = shap.Explanation(
    values=sv.values,                # SHAP values (341, 27)
    data=X_t.values,                 # feature values (341, 27)
    feature_names=feature_names,     # feature names
)

# 使用原生 beeswarm，max_display=12（默认红蓝发散色带 = coolwarm 效果）
print("  渲染中 (max_display=12, native SHAP coloring)...")
shap.plots.beeswarm(
    explanation,
    max_display=12,
    show=False,
)

# 保存
output_path = os.path.join(FIGURES, "figure_5_native_beeswarm.png")
plt.gcf().set_size_inches(10, 5.5)  # 调整尺寸
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close("all")
print(f"  [OK] 已保存: {output_path}")

# ========================================================================
# Figure 2: 多模型一致性散点图
# ========================================================================
print("\n" + "=" * 60)
print("Figure 2: TabPFN vs GBDT Feature Rank Consistency")
print("=" * 60)

MORANDI_BLUE = "#5b84b1"   # 莫兰迪蓝
MORANDI_RED = "#bc6c6c"    # 柔和红色（ρ 标注）

fig, ax = plt.subplots(figsize=(6.5, 6))

# 散点 — 仅有效特征（22个）
ax.scatter(
    valid["gbdt_rank"], valid["tabpfn_rank"],
    c=MORANDI_BLUE, s=70, edgecolors="white", linewidth=0.6,
    alpha=0.85, zorder=3,
)

# 对角虚线 y=x
lims = [0.5, max(valid["gbdt_rank"].max(), valid["tabpfn_rank"].max()) + 1.5]
ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1.2, zorder=1,
        label="y = x (perfect agreement)")

# Spearman ρ 文本框
text_str = f"Spearman $\\rho$ = {rho:.4f}\n($p$ = {pval:.2e})"
ax.text(
    0.95, 0.08, text_str,
    transform=ax.transAxes, fontsize=12,
    verticalalignment="bottom", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray",
              linewidth=0.8),
    color=MORANDI_RED, fontweight="bold",
)

ax.set_xlabel("GBDT SHAP Feature Rank", fontsize=12, fontweight="bold")
ax.set_ylabel("TabPFN Permutation Feature Rank", fontsize=12, fontweight="bold")
ax.set_title("Feature Importance Rank Agreement:\nTabPFN vs GBDT", fontsize=13,
             fontweight="bold", pad=10)

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=9, framealpha=0.8)

# 移除顶部和右侧边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

# 保存
output_path = os.path.join(FIGURES, "model_consistency_scatter.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close("all")
print(f"  [OK] 已保存: {output_path}")

# ========================================================================
# 汇总
# ========================================================================
print("\n" + "=" * 60)
print("渲染完成。两张图均从缓存读取，零模型计算。")
print("=" * 60)
print(f"  1. {FIGURES}/figure_5_native_beeswarm.png")
print(f"  2. {FIGURES}/model_consistency_scatter.png")
