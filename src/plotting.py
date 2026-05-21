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


# ============================================================================
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 60)
    print("Figure 1: Spearman 相关矩阵热力图")
    print("=" * 60)
    figure_1_clustermap()
    print("完成。")
