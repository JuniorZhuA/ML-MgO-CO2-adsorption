#!/usr/bin/env python
"""
Figure 7: Grouped SHAP Bar Plot by Feature Category (English only)
Pure cache rendering — zero model computation.

Four categories:
  1. Adsorption Conditions
  2. Synthesis Process (Precursor + Carbonization + Activation)
  3. MgO Loading & Post-treatment
  4. Final Pore Structure

Output: outputs/figures/figure_7_grouped_shap_bar.png
"""

import sys, io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

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
# Feature categories (all English, ordered as defined)
# ═══════════════════════════════════════════════════════════════════════════
CATEGORIES = [
    ("1. Adsorption Conditions",
     ["temperature_C", "pressure_bar", "T_lnP", "inv_T_K"]),
    ("2. Synthesis Process\n   (Precursor + Carbonization + Activation + Post-treatment)",
     ["carbon_precursors", "carb1_temp_C", "carb1_duration_h", "carb1_type",
      "carb2_temp_C", "carb2_duration_h", "carb2_type",
      "act1_temp_C", "act1_duration_h", "act1_type",
      "act2_temp_C", "act2_duration_h", "act2_type",
      "post_treatment_type"]),
    ("3. MgO Loading",
     ["MgO_precursors", "Mg_loading_method", "MgO_mass_ratio",
      "MgO_surface_density"]),
    ("4. Pore Structure",
     ["SBET_m2_g", "Vmicro_cm3_g",
      "Vmeso_cm3_g", "microporosity"]),
]

# Academic Morandi palette
CAT_COLORS = {
    0: "#5b84b1",  # slate blue
    1: "#fc766a",  # brick red
    2: "#76b5c5",  # lake blue
    3: "#9e9e9e",  # gray
}

# Validate
all_cat_feats = []
for _, feats in CATEGORIES:
    all_cat_feats.extend(feats)
assert len(all_cat_feats) == 26, f"Expected 27 features, got {len(all_cat_feats)}"
assert len(set(all_cat_feats)) == 26, "Duplicate feature names detected!"

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load cache
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Figure 7: Grouped SHAP Bar Plot")
print("=" * 60)

print("\n[1/3] Loading cached data ...")
X_t  = pd.read_csv(TABLES / "X_transformed_gbdt.csv")
sv   = pd.read_csv(TABLES / "shap_values_gbdt.csv")
feat_all = list(X_t.columns)
X_arr, sv_arr = X_t.values, sv.values
mean_abs = np.abs(sv_arr).mean(axis=0)
name_to_mean = dict(zip(feat_all, mean_abs))
print(f"    Features: {len(feat_all)}, Samples: {len(X_t)}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Build ordered feature list: by category group, within each group
#    sorted by mean|SHAP| descending
# ═══════════════════════════════════════════════════════════════════════════
print("[2/3] Categorizing & ordering features ...")

ordered_names = []
ordered_values = []
ordered_cat_idx = []  # which category each bar belongs to

for cat_idx, (cat_name, cat_feats) in enumerate(CATEGORIES):
    # Sort within category by mean|SHAP| descending
    g_order = sorted(cat_feats, key=lambda f: name_to_mean.get(f, 0), reverse=True)
    for f in g_order:
        ordered_names.append(f)
        ordered_values.append(name_to_mean[f])
        ordered_cat_idx.append(cat_idx)
    n = len(g_order)
    top_f = g_order[0]
    print(f"    {cat_name.split(chr(10))[0]}: {n} features, "
          f"top = {top_f} ({name_to_mean[top_f]:.2f})")

n_total = len(ordered_names)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Plot grouped horizontal bar chart
# ═══════════════════════════════════════════════════════════════════════════
print("[3/3] Rendering ...")

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y_positions = list(range(n_total - 1, -1, -1))  # top to bottom

# Draw bars with category colors
bar_colors = [CAT_COLORS[ci] for ci in ordered_cat_idx]
bars = ax.barh(
    y=y_positions,
    width=ordered_values,
    height=0.7,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.5,
    alpha=0.92,
    zorder=3,
)

# ── Category separator lines ────────────────────────────────────────────
current_y = n_total
for cat_idx in range(len(CATEGORIES) - 1):
    # Find where this category ends (looking from top to bottom)
    cat_size = len(CATEGORIES[cat_idx][1])
    current_y -= cat_size
    ax.axhline(y=current_y - 0.5, color="#666666", linestyle="--",
               linewidth=1.2, alpha=0.5, zorder=5, dashes=(8, 4))

# ── Category labels (right side of plot) ─────────────────────────────────
current_y = n_total
for cat_idx, (cat_name, cat_feats) in enumerate(CATEGORIES):
    cat_size = len(cat_feats)
    mid_y = current_y - cat_size / 2
    short_label = cat_name.replace("\n   ", ": ")
    ax.text(
        1.015, mid_y / n_total,
        cat_name.split("\n")[0].replace("1. ", "").replace("2. ", "")
        .replace("3. ", "").replace("4. ", ""),
        transform=ax.transAxes,
        fontsize=9.5, fontweight="bold", color="#555555",
        verticalalignment="center", rotation=0,
    )
    current_y -= cat_size

# ── Styling ──────────────────────────────────────────────────────────────
ax.set_yticks(list(y_positions))
ax.set_yticklabels(ordered_names, fontsize=12, fontweight="bold", color="black")
ax.tick_params(axis="y", length=0, pad=10)

ax.tick_params(axis="x", labelsize=11, colors="black")
for lbl in ax.get_xticklabels():
    lbl.set_fontweight("bold")
    lbl.set_color("black")

ax.set_xlabel("mean(|SHAP value|)  (Feature Importance)",
              fontsize=14, fontweight="bold", color="black", labelpad=6)
ax.set_title("GBDT SHAP Feature Importance by Category",
             fontsize=15, fontweight="bold", color="black", pad=12)

# Spines: remove top/right/left, keep bottom with ticks
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("black")
ax.spines["bottom"].set_linewidth(0.8)

# Vertical dashed grid
ax.set_axisbelow(True)
ax.xaxis.grid(True, alpha=0.15, linestyle="--", linewidth=0.6,
              color="#bbbbbb", zorder=0)
ax.yaxis.grid(False)

# X-axis from 0
ax.set_xlim(0, max(ordered_values) * 1.10)

# ── Legend ────────────────────────────────────────────────────────────────
legend_patches = []
for cat_idx, (cat_name, _) in enumerate(CATEGORIES):
    short = cat_name.split("\n")[0].split(". ", 1)[1]
    legend_patches.append(mpatches.Patch(
        color=CAT_COLORS[cat_idx], alpha=0.92,
        label=f"{cat_idx + 1}. {short}"
    ))

legend = ax.legend(
    handles=legend_patches,
    loc="lower right",
    fontsize=10.5,
    frameon=True,
    framealpha=0.92,
    edgecolor="#cccccc",
    facecolor="white",
    ncol=1,
    title="Feature Categories",
    title_fontsize=11,
)
legend.get_title().set_fontweight("bold")

# ── Save ──────────────────────────────────────────────────────────────────
fig.set_size_inches(13, 10.5)
plt.subplots_adjust(left=0.30, right=0.93, top=0.96, bottom=0.07)
out = FIGURES / "figure_7_grouped_shap_bar.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.close("all")

print(f"\n    Saved: {out}")
print(f"    Categories: {len(CATEGORIES)}, Features: {n_total}")
print("=" * 60)
