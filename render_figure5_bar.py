#!/usr/bin/env python
"""
Figure 5 补充: 四分类 SHAP 柱状图 (4 子图纵向排列)

四类:
  1. 吸附操作条件 (4)
  2. 合成过程（前驱体+碳化+活化） (13)
  3. MgO负载与后处理 (5)
  4. 最终孔结构特征 (5)

零模型计算，纯缓存渲染。
输出: outputs/figures/figure_5_bar_groups.png
"""

import sys, io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════════════════════════════════
GROUPS = [
    ("1. Adsorption Conditions\n   吸附操作条件", [
        "temperature_C", "pressure_bar", "T_lnP", "inv_T_K",
    ]),
    ("2. Synthesis Process\n   合成过程（前驱体+碳化+活化）", [
        "carbon_precursors", "carb1_temp_C", "carb1_duration_h", "carb1_type",
        "carb2_temp_C", "carb2_duration_h", "carb2_type",
        "act1_temp_C", "act1_duration_h", "act1_type",
        "act2_temp_C", "act2_duration_h", "act2_type",
    ]),
    ("3. MgO Loading & Post-treatment\n   MgO负载与后处理", [
        "MgO_precursors", "Mg_loading_method", "MgO_mass_ratio",
        "MgO_surface_density", "post_treatment_type",
    ]),
    ("4. Final Pore Structure\n   最终孔结构特征", [
        "SBET_m2_g", "Vmicro_cm3_g",
        "Vmeso_cm3_g", "microporosity",
    ]),
]

MORANDI_SLATE = "#5b84b1"
BAR_COLORS = ["#5b84b1", "#7aa6c2", "#4a7a9b", "#6b94b5"]  # 每子图不同色调

# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Figure 5: 四分类 SHAP 柱状图")
print("=" * 60)

# 1. 加载
print("\n[1/3] 加载缓存...")
X_t  = pd.read_csv(TABLES / "X_transformed_gbdt.csv")
sv   = pd.read_csv(TABLES / "shap_values_gbdt.csv")
feat_all = list(X_t.columns)
X_arr, sv_arr = X_t.values, sv.values
mean_abs = np.abs(sv_arr).mean(axis=0)
name_to_mean = dict(zip(feat_all, mean_abs))

# 2. 构建分组数据
print("[2/3] 分组计算...")
group_results = []
for g_title, g_feats in GROUPS:
    g_order = sorted(g_feats, key=lambda f: name_to_mean.get(f, 0), reverse=True)
    vals = [name_to_mean[f] for f in g_order]
    group_results.append((g_title, g_order, vals))
    print(f"    {g_title.split(chr(10))[0]}: {len(g_order)} 特征, "
          f"mean|SHAP| = [{min(vals):.2f}, {max(vals):.2f}]")

# 3. 绘制 4×1 子图
print("[3/3] 渲染...")

fig, axes = plt.subplots(4, 1, figsize=(9, 16),
                          gridspec_kw={"hspace": 0.55})
fig.patch.set_facecolor("white")

for ax, (g_title, g_feats, g_vals), color in zip(axes, group_results, BAR_COLORS):
    n = len(g_feats)
    y_pos = range(n - 1, -1, -1)  # 最大值在顶部

    bars = ax.barh(
        y=y_pos,
        width=g_vals,
        height=0.65,
        color=color,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.92,
        zorder=3,
    )

    # 数值标注
    for i, (x, y) in enumerate(zip(g_vals, y_pos)):
        ax.text(x + max(g_vals) * 0.02, y, f"{x:.2f}",
                fontsize=8.5, fontweight="bold", color="black",
                verticalalignment="center")

    # ── 审美 ───────────────────────────────────────────────────────
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(g_feats, fontsize=10, fontweight="bold", color="black")
    ax.tick_params(axis="y", length=0, pad=8)

    ax.tick_params(axis="x", labelsize=9, colors="black")
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
        lbl.set_color("black")

    ax.set_xlabel("mean(|SHAP value|)", fontsize=11, fontweight="bold",
                  color="black", labelpad=4)

    ax.set_title(g_title, fontsize=12, fontweight="bold", color="black",
                 pad=10, loc="left")

    # 边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(0.7)

    # 网格
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, alpha=0.12, linestyle="--", linewidth=0.5, color="#cccccc")
    ax.yaxis.grid(False)

    # X 轴从 0 开始
    ax.set_xlim(0, max(g_vals) * 1.18)

# ── 总标题 ─────────────────────────────────────────────────────────────
fig.suptitle("GBDT SHAP Feature Importance by Category",
             fontsize=15, fontweight="bold", color="black", y=0.995)

# ── 保存 ─────────────────────────────────────────────────────────────
plt.subplots_adjust(left=0.28, right=0.94, top=0.96, bottom=0.03)
out = FIGURES / "figure_5_bar_groups.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.close("all")
print(f"\n    ✓ {out}")
print("=" * 60)
