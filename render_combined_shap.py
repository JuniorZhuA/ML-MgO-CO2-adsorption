#!/usr/bin/env python
"""
顶刊风格 SHAP 组合图: 上方 Beeswarm + 下方 Bar Plot (共享 X 轴)

完全从 outputs/tables/ 缓存读取，零模型计算。
使用当前最佳模型 (GBDT) 的 SHAP 值 (TreeExplainer)。
"""

import sys, io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ── 路径 & 输出编码 ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]

# ═══════════════════════════════════════════════════════════════════════════
# 1. 加载缓存 — 注意: 不能 index_col=0, 否则 carbon_precursors 列丢失
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SHAP 组合图渲染 (Beeswarm + Bar)")
print("=" * 70)

print("\n[1/4] 加载缓存数据...")
X_t = pd.read_csv(TABLES / "X_transformed_gbdt.csv")
sv_df = pd.read_csv(TABLES / "shap_values_gbdt.csv")
feature_names = list(X_t.columns)  # 27 个特征
X_arr = X_t.values                 # (341, 27)
sv_arr = sv_df.values              # (341, 27)

print(f"    特征数: {len(feature_names)}")
print(f"    样本数: {len(X_t)}")
print(f"    SHAP 值范围: [{sv_arr.min():.1f}, {sv_arr.max():.1f}]")

# ── 计算 Top 15 特征 (按 mean(|SHAP|) 降序) ───────────────────────────────
mean_abs_shap = np.abs(sv_arr).mean(axis=0)               # (27,)
top15_order = np.argsort(mean_abs_shap)[::-1][:15]        # 降序索引
top15_features = [feature_names[i] for i in top15_order]
top15_mean_abs = mean_abs_shap[top15_order]

print(f"    Top 15 特征 (按 mean|SHAP|):")
for rank, (feat, val) in enumerate(zip(top15_features, top15_mean_abs), 1):
    print(f"      {rank:>2}. {feat:<28s} {val:8.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. 颜色归一化 — 基于 SHAP 值大小映射到 RdBu_r 色带
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/4] 构建颜色映射...")

# 使用全体 27 特征值作为颜色参考范围（百分位裁剪以抑制极端值）
all_feat_vals = X_arr[:, top15_order]  # 仅 Top 15
vmin = np.percentile(all_feat_vals, 2)
vmax = np.percentile(all_feat_vals, 98)
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.RdBu_r

print(f"    特征值颜色范围: [{vmin:.2f}, {vmax:.2f}] (2–98 百分位)")

# ═══════════════════════════════════════════════════════════════════════════
# 3. 蜜蜂群图 y 偏移算法
# ═══════════════════════════════════════════════════════════════════════════
def beeswarm_offsets(values, spread=0.35, n_bins=60):
    """为给定 SHAP 值计算蜜蜂群 y 偏移。

    点按值大小排序后分箱，同一箱内交替方向堆叠，
    产生类似 shap.plots.beeswarm 的紧凑分布。
    """
    n = len(values)
    offsets = np.zeros(n)
    order = np.argsort(values)

    vmin_v, vmax_v = values.min(), values.max()
    if np.isclose(vmin_v, vmax_v):
        return np.random.RandomState(0).uniform(-spread, spread, n)

    bin_edges = np.linspace(vmin_v, vmax_v, n_bins + 1)
    bin_counts = {}
    for idx in order:
        val = values[idx]
        b = min(int((val - vmin_v) / (vmax_v - vmin_v) * n_bins), n_bins - 1)
        cnt = bin_counts.get(b, 0)
        direction = 1 if cnt % 2 == 0 else -1
        layer = (cnt + 1) // 2
        step = spread / max(1, n / 15)  # 步长自适应样本密度
        offsets[idx] = direction * layer * step
        bin_counts[b] = cnt + 1

    return offsets

# ═══════════════════════════════════════════════════════════════════════════
# 4. 创建组合图
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/4] 绘制组合图...")

# ── 4a. 创建画布 ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(11, 10.5),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1.05], "hspace": 0.04},
)
fig.patch.set_facecolor("white")

# ── 4b. 上方: Beeswarm ──────────────────────────────────────────────────
scatter_handles = []
for j, feat_idx in enumerate(top15_order):  # top15_order[0] = #1 特征
    y_center = 14 - j  # #1 在最上方 (y=14), #15 在最下方 (y=0)

    f_vals = X_arr[:, feat_idx]
    s_vals = sv_arr[:, feat_idx]
    y_off = beeswarm_offsets(s_vals, spread=0.38)
    y_positions = y_center + y_off

    colors = cmap(norm(f_vals))

    ax1.scatter(
        s_vals, y_positions,
        c=colors, s=4.5, alpha=0.72,
        edgecolors="none", linewidth=0,
        rasterized=True,
    )

# ── 4c. 下方: Bar Plot ──────────────────────────────────────────────────
bar_values = mean_abs_shap[top15_order][::-1]  # 从 #15 → #1 (自下向上)
bar_colors_list = [cmap(norm(X_arr[:, idx].mean())) for idx in top15_order[::-1]]

ax2.barh(
    y=range(15),
    width=bar_values,
    height=0.65,
    color="#5b84b1",          # 莫兰迪灰蓝
    edgecolor="white",
    linewidth=0.4,
    alpha=0.92,
    zorder=3,
)

# ═══════════════════════════════════════════════════════════════════════════
# 5. 审美定制
# ═══════════════════════════════════════════════════════════════════════════
print("[4/4] 应用顶刊审美...")

MORANDI_SLATE = "#5b84b1"
GRID_GRAY = "#d8d8d8"
DASH_GRAY = "#999999"

# ── 5a. 两子图: 隐藏上/右边框 ──────────────────────────────────────────
for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)    # Y 轴不需要脊线
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("#555555")

# ── 5b. Y 轴: 特征名加粗 ──────────────────────────────────────────────
# 上方 beeswarm — 特征名在最左侧（ax1 的 yticks）
ax1.set_yticks(range(15))
ylabels_top = [top15_features[i] for i in range(14, -1, -1)]  # 从 #15 到 #1
ax1.set_yticklabels(ylabels_top, fontsize=10.5, fontweight="bold")
ax1.tick_params(axis="y", length=0, pad=6)

# 下方 bar — 相同特征名
ax2.set_yticks(range(15))
ylabels_bot = [top15_features[i] for i in range(14, -1, -1)]
ax2.set_yticklabels(ylabels_bot, fontsize=10.5, fontweight="bold")
ax2.tick_params(axis="y", length=0, pad=6)

# ── 5c. X 轴 ───────────────────────────────────────────────────────────
# ax1: 隐藏刻度标签 (shared x-axis)
ax1.tick_params(axis="x", labelbottom=False, length=0)
ax1.set_xlabel("")

# ax2: 显示 X 轴
ax2.set_xlabel(
    "Negative impact  <--  (CO2 uptake, mmol/g)  SHAP value"
    "  (Impact on model output)  -->  Positive impact",
    fontsize=11.5,
    fontweight="bold",
    labelpad=10,
    color="#333333",
)
ax2.tick_params(axis="x", labelsize=8.5)

# ── 5d. 水平虚线网格 (alpha=0.1) ──────────────────────────────────────
for ax in (ax1, ax2):
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, alpha=0.15, linestyle="--", linewidth=0.5,
                  color=GRID_GRAY, zorder=0)
    ax.yaxis.grid(False)

# ── 5e. 垂直分界虚线 (x=0) 贯穿上下子图 ────────────────────────────────
for ax in (ax1, ax2):
    ax.axvline(x=0, color=DASH_GRAY, linestyle="--", linewidth=0.9,
               alpha=0.75, zorder=2)

# ── 5f. X 轴范围 ───────────────────────────────────────────────────────
# 基于 SHAP 极值自动扩展 5%
x_min = sv_arr[:, top15_order].min()
x_max = max(sv_arr[:, top15_order].max(), bar_values.max())
x_margin = (x_max - x_min) * 0.04
ax1.set_xlim(x_min - x_margin, x_max + x_margin)

# ── 5g. Y 轴范围（紧凑，但留呼吸空间）────────────────────────────────
ax1.set_ylim(-0.9, 14.9)
ax2.set_ylim(-0.6, 14.6)

# ── 5h. 颜色条 (基于特征值) ────────────────────────────────────────────
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(
    sm, ax=[ax1, ax2],
    orientation="vertical",
    fraction=0.022,
    pad=0.025,
    aspect=35,
    shrink=0.55,
)
cbar.set_label("Feature value", fontsize=9.5, fontweight="bold",
               color="#444444", labelpad=6)
cbar.ax.tick_params(labelsize=7.5)
cbar.outline.set_visible(False)

# ── 5i. 标题 ─────────────────────────────────────────────────────────────
fig.suptitle(
    "GBDT SHAP Analysis — Top 15 Feature Impacts on CO₂ Adsorption",
    fontsize=13.5, fontweight="bold", y=0.985, color="#222222",
)

# ── 5j. 调整边距，确保长特征名不被截断 ─────────────────────────────────
plt.subplots_adjust(
    left=0.22,    # 为 MgO_surface_density 等长名留空间
    right=0.915,
    top=0.94,
    bottom=0.08,
)

# ═══════════════════════════════════════════════════════════════════════════
# 6. 保存
# ═══════════════════════════════════════════════════════════════════════════
output_path = FIGURES / "shap_combined_beeswarm_bar.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white",
            edgecolor="none", pad_inches=0.15)
plt.close("all")

print(f"\n{'=' * 70}")
print(f"  ✓ 已保存: {output_path}")
print(f"  ✓ 零模型计算，纯缓存渲染")
print(f"{'=' * 70}")
