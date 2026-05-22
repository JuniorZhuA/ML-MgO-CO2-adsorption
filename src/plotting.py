"""步骤7: 论文图表绘制 — 逐图精细打磨

Figure 1: 数值特征 Spearman 相关矩阵热力图 (Ward 聚类排序, 无树状图)
"""

import sys
import io
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from src.config import ROOT, TARGET, FIGURES
from src.models import _get_column_lists

warnings.filterwarnings("ignore")


def figure_1_clustermap() -> plt.Figure:
    """Figure 1: Spearman 相关矩阵热力图 (300 DPI)

    特征: 17 个有效数值特征 (排除零方差)
    排序: Spearman ρ → 1−|ρ| 距离 → Ward 层次聚类 (仅排序, 不画树状图)
    色带: RdBu_r (柔和红蓝发散型, 中心=0 白色)
    """
    # ── 1. 加载预处理数据 ──────────────────────────────────────────────
    from src.data_loader import load_and_clean
    from src.preprocessing import MissingValueImputer
    from src.feature_engineering import FeatureEngineer
    from sklearn.impute import SimpleImputer

    print("加载预处理数据...")
    df = load_and_clean()
    df = MissingValueImputer().fit_transform(df)
    df = FeatureEngineer().fit_transform(df)

    X = df.drop(columns=[TARGET])
    _, num_cols = _get_column_lists(X)
    X_num = X[num_cols]

    X_num_imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_num),
        columns=X_num.columns,
        index=X_num.index,
    )

    stds = X_num_imputed.std()
    zero_var_feats = stds[stds < 1e-10].index.tolist()
    if zero_var_feats:
        print(f"  排除零方差特征: {zero_var_feats}")
        X_num_imputed = X_num_imputed.drop(columns=zero_var_feats)

    feats = X_num_imputed.columns.tolist()
    n_feat = len(feats)
    print(f"  有效数值特征 ({n_feat}): {feats}")

    # ── 2. Spearman 相关 + 距离 + Ward 聚类排序 ────────────────────
    corr = X_num_imputed.corr(method="spearman")
    if corr.isna().any().any():
        print(f"  [注意] 填充 {corr.isna().sum().sum()} 个 NaN → 0")
        corr = corr.fillna(0.0)
    corr = (corr + corr.T) / 2

    dist = 1 - np.abs(corr.values)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    order = leaves_list(Z)  # Ward 聚类后的最优叶序

    corr_ordered = corr.iloc[order, order]

    # ── 3. 绘图 ─────────────────────────────────────────────────────
    fig_w, fig_h = 18, 17
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    print(f"  figsize: {fig_w} × {fig_h}")

    soft_cmap = sns.diverging_palette(250, 10, s=60, l=55, center="light", as_cmap=True)

    hm = sns.heatmap(
        corr_ordered,
        cmap=soft_cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.6,
        linecolor="white",
        square=True,
        cbar_kws={
            "shrink": 0.5,
        },
        ax=ax,
    )

    # ── 4. 美化 ─────────────────────────────────────────────────────
    font_base = 12
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45, ha="right", fontsize=font_base, fontweight="bold",
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0, fontsize=font_base, fontweight="bold",
    )

    for t in hm.texts:
        t.set_fontsize(font_base)
        t.set_fontweight("bold")

    # 颜色条: 标签加粗加大，刻度加大
    cbar = hm.collections[0].colorbar
    cbar.ax.set_ylabel("Spearman ρ", fontsize=font_base + 1, fontweight="bold", labelpad=10)
    cbar.ax.yaxis.label.set_fontweight("bold")
    cbar.ax.tick_params(labelsize=font_base)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    # ── 5. 保存 ─────────────────────────────────────────────────────
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / "Figure_1_Clustermap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {output_path}")
    print(f"     分辨率: 300 DPI, 画布: {fig_w}×{fig_h} 英寸")
    print(f"     特征数: {n_feat}, 全局字号: {font_base}pt bold")

    plt.close("all")
    return fig


def figure_2_boxplot() -> plt.Figure:
    """Figure 2: 数值特征标准化箱线图 (300 DPI)

    19个数值特征，Z-score标准化后横向箱线图，按领域分组排列。
    """

    from src.data_loader import load_and_clean
    from src.preprocessing import MissingValueImputer
    from src.feature_engineering import FeatureEngineer

    print("加载预处理数据...")
    df = load_and_clean()
    df = MissingValueImputer().fit_transform(df)
    df = FeatureEngineer().fit_transform(df)

    X = df.drop(columns=[TARGET])
    _, num_cols = _get_column_lists(X)
    X_num = X[num_cols]

    # Z-score 标准化 (逐列忽略NaN, 排除零方差列)
    X_scaled = X_num.copy()
    zero_var_cols = []
    for col in X_scaled.columns:
        col_std = X_scaled[col].std()
        if col_std < 1e-10:
            zero_var_cols.append(col)
            X_scaled = X_scaled.drop(columns=[col])
        else:
            mu = X_scaled[col].mean()
            X_scaled[col] = (X_scaled[col] - mu) / col_std

    if zero_var_cols:
        print(f"  排除零方差特征: {zero_var_cols}")
        num_cols = [c for c in num_cols if c not in zero_var_cols]

    # 按领域分组排序
    feature_groups = [
        ("孔结构特征", [
            "SBET_m2_g", "Vtotal_cm3_g", "Vmicro_cm3_g", "Vmeso_cm3_g", "microporosity",
        ]),
        ("MgO 负载特征", [
            "MgO_mass_ratio", "MgO_surface_density",
        ]),
        ("工艺条件", [
            "temperature_C", "pressure_bar", "T_lnP", "inv_T_K",
        ]),
        ("活化参数", [
            "act1_temp_C", "act1_duration_h", "act2_temp_C", "act2_duration_h",
        ]),
        ("碳化参数", [
            "carb1_temp_C", "carb1_duration_h", "carb2_temp_C", "carb2_duration_h",
        ]),
    ]

    ordered_feats = []
    for _, feats in feature_groups:
        for f in feats:
            if f in X_scaled.columns:
                ordered_feats.append(f)

    X_plot = X_scaled[ordered_feats]

    # 转换为长格式用于 seaborn
    df_long = X_plot.melt(var_name="Feature", value_name="Z-score")

    # 绘图
    n_feat = len(ordered_feats)
    fig_h = max(5.5, n_feat * 0.38)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    # 按组分配明亮柔和色
    group_colors = ["#7EC8E3", "#F4A87C", "#A8D8A8", "#D4B5E1", "#F7C873"]
    feat_to_color = {}
    for gi, (_, feats) in enumerate(feature_groups):
        for f in feats:
            if f in ordered_feats:
                feat_to_color[f] = group_colors[gi]

    bp = sns.boxplot(
        data=df_long,
        y="Feature",
        x="Z-score",
        order=ordered_feats,
        palette=feat_to_color,
        linewidth=0.8,
        fliersize=2,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
        ax=ax,
    )

    x_min = np.nanpercentile(X_plot.values, 1)
    x_max = np.nanpercentile(X_plot.values, 99)
    if np.isnan(x_min) or np.isinf(x_min):
        x_min = -4
    if np.isnan(x_max) or np.isinf(x_max):
        x_max = 4
    ax.set_xlim(x_min - 0.3, x_max + 0.3)

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

    ax.set_xlabel("Z-score", fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(labelsize=10)

    # 保存
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / "Figure_2_Boxplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {output_path}")
    print(f"     分辨率: 300 DPI, 特征数: {n_feat}")

    plt.close("all")
    return fig


# ============================================================================
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=int, default=0, help="指定图号 (0=全部)")
    args = parser.parse_args()

    if args.figure == 0 or args.figure == 1:
        print("=" * 60)
        print("Figure 1: Spearman 相关矩阵热力图")
        print("=" * 60)
        figure_1_clustermap()

    if args.figure == 0 or args.figure == 2:
        print("=" * 60)
        print("Figure 2: 数值特征标准化箱线图")
        print("=" * 60)
        figure_2_boxplot()

    print("完成。")
