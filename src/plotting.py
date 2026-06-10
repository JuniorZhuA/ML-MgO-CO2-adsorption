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

    17个数值特征，Z-score标准化后横向箱线图。
    特征名: mathtext 正确渲染下标/上标，单位放入括号。
    Times New Roman 全局字体，不加分组。
    """

    # ── 微软雅黑 (局部, 不影响其他图); SVG文字存为文本 ──
    _orig_family = matplotlib.rcParams['font.family']
    _orig_sans = matplotlib.rcParams['font.sans-serif'].copy()
    _orig_uminus = matplotlib.rcParams['axes.unicode_minus']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] + _orig_sans
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

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

    # ── 特征名映射: 无单位，空格分隔，act/carb 简写 ──
    FEATURE_NAME_MAP = {
        # 孔结构特征
        "SBET_m2_g":        "SBET",
        "Vmicro_cm3_g":     "Vmicro",
        "Vmeso_cm3_g":      "Vmeso",
        "microporosity":    "Microporosity",
        # MgO 负载特征
        "MgO_mass_ratio":       "MgO Mass Ratio",
        "MgO_surface_density":  "MgO Surface Density",
        # 工艺条件
        "temperature_C":    "Temperature",
        "pressure_bar":     "Pressure",
        "T_lnP":            "T·ln(P)",
        "inv_T_K":          "1/T",
        # 活化参数
        "act1_temp_C":      "act1 temp",
        "act1_duration_h":  "act1 duration",
        "act2_temp_C":      "act2 temp",
        "act2_duration_h":  "act2 duration",
        # 碳化参数
        "carb1_temp_C":     "carb1 temp",
        "carb1_duration_h": "carb1 duration",
        "carb2_temp_C":     "carb2 temp",
        "carb2_duration_h": "carb2 duration",
    }

    # 特征显示顺序 (逻辑排列, 不加分组)
    FEATURE_ORDER = [
        "SBET_m2_g", "Vmicro_cm3_g", "Vmeso_cm3_g", "microporosity",
        "MgO_mass_ratio", "MgO_surface_density",
        "temperature_C", "pressure_bar", "T_lnP", "inv_T_K",
        "act1_temp_C", "act1_duration_h", "act2_temp_C", "act2_duration_h",
        "carb1_temp_C", "carb1_duration_h", "carb2_temp_C", "carb2_duration_h",
    ]

    ordered_feats_raw = [f for f in FEATURE_ORDER if f in X_scaled.columns]
    ordered_feats_display = [FEATURE_NAME_MAP[f] for f in ordered_feats_raw]

    X_plot = X_scaled[ordered_feats_raw].copy()
    X_plot.columns = ordered_feats_display
    df_long = X_plot.melt(var_name="Feature", value_name="Z-score")

    # 绘图
    n_feat = len(ordered_feats_display)
    fig_h = max(6, n_feat * 0.48)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # ── 统一明亮柔和色 ──
    BOX_COLOR = "#EFC8B0"  # 柔和活泼珊瑚橙

    bp = sns.boxplot(
        data=df_long,
        y="Feature",
        x="Z-score",
        order=ordered_feats_display,
        color=BOX_COLOR,
        linewidth=1.1,
        fliersize=3.5,
        flierprops={"marker": "o", "markersize": 3.5, "alpha": 0.4},
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

    # ── 标签: 微软雅黑 Bold, 加大字号 ──
    ax.set_xlabel("Z-score", fontsize=18, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(labelsize=13)

    ax.set_yticklabels(ordered_feats_display, fontsize=14, fontweight="bold")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    # 保存 (PNG + SVG + PDF)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES / "Figure_2_Boxplot.png"
    svg_path = FIGURES / "Figure_2_Boxplot.svg"
    pdf_path = FIGURES / "Figure_2_Boxplot.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {png_path}")
    print(f"     已保存: {svg_path}")
    print(f"     已保存: {pdf_path}")
    print(f"     分辨率: 300 DPI, 特征数: {n_feat}, Z-score 标准化")
    print(f"     字体: 微软雅黑 Bold, 配色: {BOX_COLOR}, 无分组")

    # 恢复全局字体设置
    matplotlib.rcParams['font.family'] = _orig_family
    matplotlib.rcParams['font.sans-serif'] = _orig_sans
    matplotlib.rcParams['axes.unicode_minus'] = _orig_uminus

    plt.close("all")
    return fig


def figure_S1_precursor_distribution() -> plt.Figure:
    """Figure S1: 不同碳前驱体类型的 CO₂ 吸附量分布 (300 DPI)

    按碳前驱体类型分组展示 CO₂ uptake 分布，n<10 的类别合并为 Other，
    "saw dust" 合并入 "sawdust"，按中位数降序排列。
    """

    from src.data_loader import load_and_clean
    from src.preprocessing import MissingValueImputer
    from src.feature_engineering import FeatureEngineer

    print("加载预处理数据...")
    df = load_and_clean()
    df = MissingValueImputer().fit_transform(df)
    df = FeatureEngineer().fit_transform(df)

    # 合并拼写变体: "saw dust" → "sawdust"
    df["carbon_precursors"] = df["carbon_precursors"].replace({"saw dust": "sawdust"})

    # 合并 n<10 为 "Other"
    counts = df["carbon_precursors"].value_counts()
    small_cats = counts[counts < 10].index.tolist()
    df["carbon_precursors"] = df["carbon_precursors"].replace(
        {c: "Other" for c in small_cats}
    )

    # 统一 title case（Other 除外）
    df["carbon_precursors"] = df["carbon_precursors"].apply(
        lambda x: x if x == "Other" else x.title()
    )

    # 按中位数降序排列，Other 固定在最后
    order = (
        df.groupby("carbon_precursors")["CO2_uptake_mg_g"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    other_cats = [c for c in order if "Other" in c]
    other_cats.sort()  # 多个Other按字母序
    order = [c for c in order if c not in other_cats] + other_cats

    labels = [str(cat) for cat in order]

    n_cats = len(order)
    fig_h = max(4.5, n_cats * 0.55)

    # 全局字体: 微软雅黑加粗
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["font.weight"] = "bold"

    # ── 横向箱线图 ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, fig_h))

    palette = ["#5DADE2"] * len(order)

    bp = sns.boxplot(
        data=df,
        y="carbon_precursors",
        x="CO2_uptake_mg_g",
        order=order,
        palette=palette,
        linewidth=0.8,
        fliersize=2.5,
        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.4},
        ax=ax,
    )

    # 叠加散点 (strip plot)
    sns.stripplot(
        data=df,
        y="carbon_precursors",
        x="CO2_uptake_mg_g",
        order=order,
        color="black",
        size=2.5,
        alpha=0.3,
        jitter=True,
        ax=ax,
    )

    ax.set_yticklabels(labels, fontsize=13, fontweight="bold")
    ax.set_xlabel("CO2 Uptake (mg/g)", fontsize=16, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(labelsize=13)

    # x 轴从 0 开始
    ax.set_xlim(-5, df["CO2_uptake_mg_g"].max() * 1.06)

    # ── 保存 ─────────────────────────────────────────────────────────
    fig.tight_layout()
    # x轴刻度数值加粗
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / "Figure_S1_Precursor_Distribution.pdf"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {output_path}")
    print(f"     分辨率: 300 DPI, 类别数: {n_cats}, 合并小类: {small_cats}")

    plt.close("all")
    return fig


def figure_3_model_comparison() -> plt.Figure:
    """Figure 3: 模型性能对比横向柱状图 — R² + RMSE 双面板 (300 DPI)

    9个模型按 R² 降序排列，按模型类别着色，含误差棒和性能差距标注。
    """
    import json

    # ── 1. 加载 CV 结果 ──────────────────────────────────────────────────
    from src.config import TABLES
    cv_path = TABLES / "cv_results.json"
    with open(cv_path, "r", encoding="utf-8") as f:
        cv_data = json.load(f)

    models_order = ["TabPFN", "GBDT", "RF", "XGBoost", "LightGBM", "GPR", "SVR", "Ridge", "Lasso"]

    records = []
    for name in models_order:
        d = cv_data[name]["aggregate"]
        records.append({
            "model": name,
            "R²": d["R²_mean"],
            "R²_std": d["R²_std"],
            "RMSE": d["RMSE_mean"],
            "RMSE_std": d["RMSE_std"],
        })

    df = pd.DataFrame(records)

    category_map = {
        "TabPFN":   ("Foundation", "#C62828"),
        "GBDT":     ("Tree Ensemble", "#2E7D32"),
        "RF":       ("Tree Ensemble", "#2E7D32"),
        "XGBoost":  ("Tree Ensemble", "#2E7D32"),
        "LightGBM": ("Tree Ensemble", "#2E7D32"),
        "GPR":      ("Kernel Method", "#E65100"),
        "SVR":      ("Kernel Method", "#E65100"),
        "Ridge":    ("Linear Baseline", "#1565C0"),
        "Lasso":    ("Linear Baseline", "#1565C0"),
    }

    legend_spec = [
        ("Foundation (TabPFN)", "#C62828"),
        ("Tree Ensemble",       "#2E7D32"),
        ("Kernel Method",       "#E65100"),
        ("Linear Baseline",     "#1565C0"),
    ]

    df["category"] = df["model"].map(lambda m: category_map[m][0])
    df["color"]    = df["model"].map(lambda m: category_map[m][1])

    # ── 2. 双面板绘图 ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    n = len(df)
    y_pos = list(range(n - 1, -1, -1))
    divider_y = 1.5  # Ridge在index 7→y_pos=1, SVR在index 6→y_pos=2

    # 计算各面板 xlim — 左边留空，右边收紧
    r2_text_right = max(df["R²"] + df["R²_std"] + 0.01)
    rmse_text_right = max(df["RMSE"] + df["RMSE_std"] + 0.6)

    for ax, metric, std_col, unit, title, fmt, x_left, x_right in [
        (ax1, "R²", "R²_std", "", "Determination Coefficient R²", ".4f",
         0, r2_text_right * 1.14),
        (ax2, "RMSE", "RMSE_std", " (mg/g)", "Root Mean Square Error", ".2f",
         0, rmse_text_right * 1.14),
    ]:
        # 背景分区: 蓝=线性基线, 绿=非线性模型
        ax.axhspan(-0.5, divider_y, facecolor="#E3F2FD", alpha=0.30, zorder=0, edgecolor="none")
        ax.axhspan(divider_y, n - 0.5, facecolor="#E8F5E9", alpha=0.25, zorder=0, edgecolor="none")

        ax.barh(y_pos, df[metric],
                xerr=df[std_col], capsize=2.5,
                color=df["color"], edgecolor="white", linewidth=0.5,
                height=0.62, error_kw={"linewidth": 1.0}, zorder=3)

        for i, (val, std) in enumerate(zip(df[metric], df[std_col])):
            text_x = val + std + (0.01 if metric == "R²" else 0.6)
            ax.text(text_x, y_pos[i],
                    f"{val:{fmt}}", va="center", fontsize=11, fontweight="bold", zorder=4)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["model"], fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{metric} (mean ± std){unit}", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlim(x_left, x_right)
        ax.invert_yaxis()
        ax.tick_params(labelsize=10)
        ax.axhline(y=divider_y, color="#333333", linewidth=1.0, linestyle="--", zorder=2)

    # ── 3. 统一图例 ──────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, edgecolor="white", label=lab)
                      for lab, c in legend_spec]
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=4, fontsize=10.5, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.01))

    # ── 4. 保存 ──────────────────────────────────────────────────────────
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / "Figure_3_Model_Comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {output_path}")
    print(f"     分辨率: 300 DPI, 模型数: {len(df)}")

    plt.close("all")
    return fig


def figure_4_tabpfn_marginal() -> plt.Figure:
    """Figure 4: TabPFN 带边缘分布 + 残差图 (300 DPI)

    GridSpec 3行×6列, hspace/wspace=0, sharex/sharey 共用脊柱:
    - 顶部: gaussian_kde 密度 (sharex, 底边=主体顶边X轴)
    - 主体: 散点 + y=x + 线性回归 + sns.regplot 95%CI
    - 右侧: gaussian_kde 密度 (sharey, 左边=主体右边Y轴)
    - 底部: 残差散点 (sharex, 顶边=主体底边X轴)
    """

    import json
    import joblib
    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.stats import linregress
    from matplotlib.gridspec import GridSpec
    from src.config import ROOT
    from src.data_loader import load_and_clean

    model_name = "TabPFN"

    print("加载数据...")
    df = load_and_clean()
    y_true = df[TARGET].values

    cv_path = ROOT / "outputs" / "tables" / "cv_predictions.json"
    with open(cv_path, "r", encoding="utf-8") as f:
        cv_data = json.load(f)

    models_dir = ROOT / "outputs" / "models"
    pipe = joblib.load(models_dir / f"{model_name}_final.pkl")
    tr_preds = pipe.predict(df.drop(columns=[TARGET]))
    if tr_preds.ndim == 2 and tr_preds.shape[1] == 1:
        tr_preds = tr_preds.ravel()

    cv_preds = np.array(cv_data[model_name]["y_pred"])

    r2_train = r2_score(y_true, tr_preds)
    rmse_train = np.sqrt(mean_squared_error(y_true, tr_preds))
    r2_test = r2_score(y_true, cv_preds)
    rmse_test = np.sqrt(mean_squared_error(y_true, cv_preds))
    print(f"  {model_name}: Train R²={r2_train:.4f} RMSE={rmse_train:.1f} | "
          f"Test R²={r2_test:.4f} RMSE={rmse_test:.1f}")

    # ── 配色 ─────────────────────────────────────────────────────────────
    TRAIN_FILL  = "#F4A8A0"
    TRAIN_LINE  = "#C62828"
    TEST_FILL   = "#A0C8E8"
    TEST_LINE   = "#0D47A1"

    lo = min(y_true.min(), cv_preds.min(), tr_preds.min()) - 8
    hi = max(y_true.max(), cv_preds.max(), tr_preds.max()) + 8

    # ── GridSpec: 主体先建, 边缘共享轴 ──────────────────────────────────
    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(3, 6, figure=fig,
                  height_ratios=[1, 5, 1.3],
                  width_ratios=[1, 1, 1, 1, 1, 1],
                  hspace=0.00, wspace=0.00,
                  left=0.085, right=0.935, top=0.95, bottom=0.085)

    ax_main  = fig.add_subplot(gs[1, :5])        # 主体
    ax_top   = fig.add_subplot(gs[0, :5], sharex=ax_main)   # 顶部KDE, 底边=主体顶边
    ax_right = fig.add_subplot(gs[1, 5], sharey=ax_main)    # 右侧KDE, 左边=主体右边
    ax_resid = fig.add_subplot(gs[2, :5], sharex=ax_main)   # 残差, 顶边=主体底边

    # ══════════════════════════════════════════════════════════════════════
    # 主体散点图
    # ══════════════════════════════════════════════════════════════════════

    # 训练集: 浅粉圆点
    ax_main.scatter(y_true, tr_preds, c=TRAIN_FILL, marker="o", s=26,
                    edgecolors="white", linewidths=0.3, alpha=0.65, zorder=3,
                    label="Train")
    # 测试集: 浅蓝三角
    ax_main.scatter(y_true, cv_preds, c=TEST_FILL, marker="^", s=30,
                    edgecolors="white", linewidths=0.3, alpha=0.7, zorder=4,
                    label="Test")

    # y=x 理想线
    ax_main.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.45, zorder=1)

    # 训练集线性回归 (红线, 无CI)
    sl_tr, ic_tr, _, _, _ = linregress(y_true, tr_preds)
    x_fit = np.linspace(lo, hi, 200)
    ax_main.plot(x_fit, sl_tr * x_fit + ic_tr,
                 color=TRAIN_LINE, linewidth=1.5, zorder=5)

    # 测试集: 线性回归 + 95% 参数化 CI (手算, 避免sns.regplot bootstrap线条)
    sl_te, ic_te, r_te, _, std_err = linregress(y_true, cv_preds)
    ax_main.plot(x_fit, sl_te * x_fit + ic_te,
                 color=TEST_LINE, linewidth=1.8, zorder=6)
    # 参数化 95% CI: y_pred ± t_0.025 * SE
    from scipy.stats import t as t_dist
    n = len(y_true)
    t_crit = t_dist.ppf(0.975, n - 2)
    x_mean = np.mean(y_true)
    ssx = np.sum((y_true - x_mean) ** 2)
    se_fit = std_err * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / ssx)
    ci_upper = sl_te * x_fit + ic_te + t_crit * se_fit
    ci_lower = sl_te * x_fit + ic_te - t_crit * se_fit
    ax_main.fill_between(x_fit, ci_lower, ci_upper,
                         color=TEST_LINE, alpha=0.12, edgecolor="none", zorder=5)

    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_aspect("equal")

    # 图例 (左上)
    ax_main.legend(loc="upper left", fontsize=9, frameon=True,
                   fancybox=True, markerscale=1.1)

    # R² 标注 (右下)
    r2_text = (f"Train R² = {r2_train:.4f}\nTest R² = {r2_test:.4f}")
    ax_main.text(0.96, 0.055, r2_text, transform=ax_main.transAxes,
                 fontsize=9.5, fontweight="bold", va="bottom", ha="right",
                 color="#37474F", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                           edgecolor="#BDBDBD", linewidth=0.5, alpha=0.9))

    ax_main.grid(False)
    ax_main.set_xlabel("Experimental CO₂ Uptake (mg/g)", fontsize=13,
                       fontweight="bold")
    ax_main.set_ylabel("Predicted CO₂ Uptake (mg/g)", fontsize=13,
                       fontweight="bold")
    ax_main.tick_params(labelsize=9.5)
    ax_main.set_title(model_name, fontsize=15, fontweight="bold", pad=5)

    # ══════════════════════════════════════════════════════════════════════
    # 顶部边缘: X 轴密度 (gaussian_kde, sharex → 底边=主体顶边)
    # ══════════════════════════════════════════════════════════════════════
    from scipy.stats import gaussian_kde
    x_grid = np.linspace(lo, hi, 300)
    kde_x = gaussian_kde(y_true)
    ax_top.fill_between(x_grid, kde_x(x_grid), alpha=0.25,
                        color=TRAIN_LINE, edgecolor="none")
    ax_top.plot(x_grid, kde_x(x_grid), color=TRAIN_LINE, linewidth=1.5)

    ax_top.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    for s in ax_top.spines.values():
        s.set_visible(False)

    # ══════════════════════════════════════════════════════════════════════
    # 右侧边缘: Y 轴密度 (gaussian_kde, sharey → 左边=主体右边)
    # ══════════════════════════════════════════════════════════════════════
    y_grid = np.linspace(lo, hi, 300)
    kde_tr_y = gaussian_kde(tr_preds)
    kde_te_y = gaussian_kde(cv_preds)
    ax_right.fill_betweenx(y_grid, kde_tr_y(y_grid), alpha=0.18,
                           color=TRAIN_LINE, edgecolor="none")
    ax_right.plot(kde_tr_y(y_grid), y_grid, color=TRAIN_LINE, linewidth=1.3,
                  label="Train")
    ax_right.fill_betweenx(y_grid, kde_te_y(y_grid), alpha=0.22,
                           color=TEST_LINE, edgecolor="none")
    ax_right.plot(kde_te_y(y_grid), y_grid, color=TEST_LINE, linewidth=1.5,
                  label="Test")

    ax_right.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax_right.legend(fontsize=7, loc="upper right", frameon=True, fancybox=True)
    for s in ax_right.spines.values():
        s.set_visible(False)

    # ══════════════════════════════════════════════════════════════════════
    # 底部残差图: Y = Predicted − True (sharex=主体 → 顶部脊柱共用)
    # ══════════════════════════════════════════════════════════════════════
    resid_train = tr_preds - y_true
    resid_test  = cv_preds - y_true

    ax_resid.axhline(y=0, color="gray", linewidth=0.7, linestyle="--",
                     alpha=0.55, zorder=1)

    ax_resid.scatter(y_true, resid_train, c=TRAIN_FILL, marker="o", s=16,
                     edgecolors="white", linewidths=0.2, alpha=0.55, zorder=2)
    ax_resid.scatter(y_true, resid_test, c=TEST_FILL, marker="^", s=20,
                     edgecolors="white", linewidths=0.2, alpha=0.6, zorder=3)

    r_abs = max(abs(resid_train).max(), abs(resid_test).max()) * 1.2
    ax_resid.set_ylim(-r_abs, r_abs)

    ax_resid.grid(False)
    ax_resid.set_xlabel("Experimental CO₂ Uptake (mg/g)", fontsize=13,
                        fontweight="bold")
    ax_resid.set_ylabel("Residuals (mg/g)", fontsize=12, fontweight="bold")
    ax_resid.tick_params(labelsize=9)

    # ── 保存 ─────────────────────────────────────────────────────────────
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / f"Figure_4_{model_name}_Marginal.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  [OK] 已保存: {output_path}")
    plt.close(fig)
    return fig


# ============================================================================
# 共享缓存加载器 —— 所有SHAP绘图函数从此读取，避免重复计算
# ============================================================================

def _load_shap_cache():
    """加载缓存的 SHAP 中间数据。

    优先从 outputs/tables/ 读取，缓存不存在时自动计算并保存。
    这样绘图函数修改后无需重跑昂贵的SHAP计算。

    Returns
    -------
    dict with keys:
        X_transformed : DataFrame (341 × 27)
        shap_values   : ndarray (341 × 27)
        feature_names : list[str]
        importance    : DataFrame (27 × 2)
        dropped_features : list[str]
    """
    import json
    import joblib
    import shap
    from src.data_loader import load_and_clean
    from src.config import ROOT

    tables_dir = ROOT / "outputs" / "tables"
    figures_dir = ROOT / "outputs" / "figures"

    X_csv = tables_dir / "X_transformed_gbdt.csv"
    shap_csv = tables_dir / "shap_values_gbdt.csv"
    imp_csv = tables_dir / "shap_importance_gbdt.csv"
    feat_json = tables_dir / "feature_names.json"
    dropped_json = tables_dir / "dropped_features.json"

    # ---- 缓存命中：从磁盘直接读取 ----
    if all(f.exists() for f in [X_csv, shap_csv, imp_csv, feat_json]):
        print("  [缓存] 从 outputs/tables/ 加载 SHAP 数据...")
        with open(feat_json, "r", encoding="utf-8") as f:
            feat_meta = json.load(f)
        feature_names = feat_meta["feature_names"]

        dropped_features = []
        if dropped_json.exists():
            with open(dropped_json, "r", encoding="utf-8") as f:
                dropped_features = json.load(f).get("dropped_features", [])

        return {
            "X_transformed": pd.read_csv(X_csv),
            "shap_values": pd.read_csv(shap_csv).values,
            "feature_names": feature_names,
            "importance": pd.read_csv(imp_csv),
            "dropped_features": dropped_features,
        }

    # ---- 缓存缺失：计算并保存 ----
    print("  [计算] 缓存不存在，重新计算 GBDT SHAP...")
    df = load_and_clean()
    X_raw = df.drop(columns=[TARGET])

    gbdt_path = ROOT / "outputs" / "models" / "GBDT_final.pkl"
    gbdt_pipe = joblib.load(gbdt_path)

    imputer = gbdt_pipe.named_steps["impute"]
    engineer = gbdt_pipe.named_steps["features"]
    preprocessor = gbdt_pipe.named_steps["preprocessor"]
    model = gbdt_pipe.named_steps["model"]

    X_imputed = imputer.transform(X_raw)
    X_fe = engineer.transform(X_imputed)
    X_transformed = preprocessor.transform(X_fe)

    cat_cols, num_cols = _get_column_lists(X_fe)
    feature_names = cat_cols + num_cols

    if not isinstance(X_transformed, pd.DataFrame):
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

    print("  计算 TreeExplainer SHAP 值并缓存...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_transformed)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    importance = np.abs(shap_vals).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "shap_importance_mean": importance,
    }).sort_values("shap_importance_mean", ascending=False).reset_index(drop=True)

    # 保存缓存
    tables_dir.mkdir(parents=True, exist_ok=True)
    X_transformed.to_csv(X_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(shap_vals, columns=feature_names).to_csv(shap_csv, index=False, encoding="utf-8-sig")
    imp_df.to_csv(imp_csv, index=False, encoding="utf-8-sig")
    with open(feat_json, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names, "cat_cols": cat_cols, "num_cols": num_cols}, f, ensure_ascii=False, indent=2)
    # dropped_features 由 shap_analysis.py 写入，此处不覆盖

    # 加载 dropped_features（如果存在）
    dropped_features = []
    if dropped_json.exists():
        with open(dropped_json, "r", encoding="utf-8") as f:
            dropped_features = json.load(f).get("dropped_features", [])

    print(f"  缓存已保存: X_transformed ({X_transformed.shape}), SHAP ({shap_vals.shape}), importance ({len(imp_df)} features)")

    return {
        "X_transformed": X_transformed,
        "shap_values": shap_vals,
        "feature_names": feature_names,
        "importance": imp_df,
        "dropped_features": dropped_features,
    }


def figure_5_shap_beeswarm() -> plt.Figure:
    """Figure 5: GBDT SHAP beeswarm 汇总图 — Top 15 特征 (300 DPI)"""

    # ★ 从缓存加载（首次计算自动缓存）
    cache = _load_shap_cache()
    X_transformed = cache["X_transformed"]
    shap_vals = cache["shap_values"]
    feature_names = cache["feature_names"]

    # 计算 mean(|SHAP|) 并选 Top 15
    importance = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(importance)[-15:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    shap_top = shap_vals[:, top_idx]
    X_top = X_transformed.iloc[:, top_idx]

    print(f"  Top 15 特征: {top_features}")

    # ── 自定义 beeswarm ───────────────────────────────────────────────────
    n_features = len(top_features)
    fig, ax = plt.subplots(figsize=(10, 0.45 * n_features + 1.5))

    # 颜色映射基于特征值
    for i, (feat_name, feat_idx) in enumerate(zip(top_features, range(n_features))):
        shaps = shap_top[:, feat_idx]
        vals = X_top.iloc[:, feat_idx].values

        # 归一化特征值用于着色
        vmin, vmax = np.nanpercentile(vals, [2, 98])
        if vmax - vmin < 1e-10:
            vmin, vmax = vals.min(), vals.max() + 1e-6
        norm_vals = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
        colors = plt.cm.RdYlBu_r(norm_vals)

        # 垂直抖动 (beeswarm 式)
        y_jitter = np.zeros(len(shaps))
        np.random.seed(42 + i)
        # 沿 SHAP 值排序后做简单抖动
        order = np.argsort(np.abs(shaps))
        for rank, j in enumerate(order):
            # 用 rank 加小噪声替代真正的 beeswarm packing
            y_jitter[j] = rank / len(order) * 0.7 - 0.35

        ax.scatter(shaps, i + y_jitter, c=colors, s=8, edgecolors="none", alpha=0.6, rasterized=True)

    ax.axvline(x=0, color="black", linewidth=0.6, linestyle="-", alpha=0.5)

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(top_features, fontsize=10, fontweight="bold")
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(labelsize=9)
    ax.set_ylim(-0.8, n_features - 0.2)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r,
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, aspect=25, pad=0.02)
    cbar.set_label("Feature value", fontsize=10, fontweight="bold")
    cbar.ax.set_yticks([0, 0.5, 1])
    cbar.ax.set_yticklabels(["Low", "Mid", "High"], fontsize=8, fontweight="bold")

    fig.suptitle("GBDT SHAP Feature Importance (Top 15)", fontsize=13,
                 fontweight="bold", y=0.98)

    # ── 保存 ─────────────────────────────────────────────────────────────
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / "Figure_5_SHAP_Beeswarm.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {output_path}")

    plt.close("all")
    return fig


def figure_6_shap_dependence() -> plt.Figure:
    """Figure 6: Top 3 特征 SHAP 依赖图 — 横向排列 (300 DPI)"""

    from scipy.stats import spearmanr

    # ★ 从缓存加载
    cache = _load_shap_cache()
    X_transformed = cache["X_transformed"]
    shap_vals = cache["shap_values"]
    all_features = cache["feature_names"]

    # Top 3 特征
    importance = np.abs(shap_vals).mean(axis=0)
    top3_idx = np.argsort(importance)[-3:][::-1]
    top3_features = [all_features[i] for i in top3_idx]

    # 为每个 top 特征找最佳着色特征 (SHAP 值与其他特征值的最高 Spearman 相关)
    from scipy.stats import spearmanr

    interaction_features = []
    for i, feat_idx in enumerate(top3_idx):
        target_shap = shap_vals[:, feat_idx]
        best_corr = -1
        best_feat = None
        for j in range(len(all_features)):
            if j == feat_idx:
                continue
            corr, _ = spearmanr(target_shap, X_transformed.iloc[:, j].values)
            abs_corr = abs(corr)
            if abs_corr > best_corr:
                best_corr = abs_corr
                best_feat = all_features[j]
        interaction_features.append(best_feat)
        print(f"  {top3_features[i]}: 着色特征 = {best_feat} (|ρ|={best_corr:.3f})")

    # ── 绘图: 1×3 横向面板 ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, feat_name, feat_idx, color_feat in zip(
        axes, top3_features, top3_idx, interaction_features
    ):
        shaps = shap_vals[:, feat_idx]
        feat_vals = X_transformed[feat_name].values
        color_vals = X_transformed[color_feat].values

        # 颜色归一化
        vmin, vmax = np.nanpercentile(color_vals, [3, 97])
        if vmax - vmin < 1e-10:
            vmin, vmax = color_vals.min(), color_vals.max() + 1e-6
        norm = plt.Normalize(vmin, vmax)

        scatter = ax.scatter(
            feat_vals, shaps, c=color_vals, cmap=plt.cm.RdYlBu_r,
            norm=norm, s=18, edgecolors="none", alpha=0.65,
        )

        ax.axhline(y=0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

        ax.set_xlabel(feat_name, fontsize=12, fontweight="bold")
        ax.set_ylabel(f"SHAP value for {feat_name}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

        # 颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, aspect=25, pad=0.02)
        cbar.set_label(color_feat, fontsize=9, fontweight="bold")
        cbar.ax.tick_params(labelsize=7)

        # R² 趋势线 (LOWESS 近似: 分段均值)
        sort_idx = np.argsort(feat_vals)
        window = max(len(sort_idx) // 15, 10)
        x_smooth = []
        y_smooth = []
        for k in range(0, len(sort_idx), window):
            chunk = sort_idx[k:k + window]
            x_smooth.append(np.median(feat_vals[chunk]))
            y_smooth.append(np.median(shaps[chunk]))
        ax.plot(x_smooth, y_smooth, "k-", linewidth=1.8, alpha=0.7)

    fig.suptitle("SHAP Dependence Plots — Top 3 Features (GBDT)", fontsize=14,
                 fontweight="bold", y=1.01)

    # ── 保存 ─────────────────────────────────────────────────────────────
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / "Figure_6_SHAP_Dependence.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {output_path}")

    plt.close("all")
    return fig


def figure_shap_bar() -> plt.Figure:
    """Figure: GBDT SHAP 全局特征重要性条形图 (Mean |SHAP|, Morandi配色, 300 DPI)

    展示所有有效特征的 mean(|SHAP|) 降序排列。
    使用莫兰迪色系（低饱和度），适合论文发表。
    """

    # ── 莫兰迪色系 ────────────────────────────────────────────────────────
    MORANDI_SLATE = "#8B9DAF"      # 板岩蓝

    # ★ 从缓存加载
    cache = _load_shap_cache()
    importance = cache["importance"]["shap_importance_mean"].values
    all_features = cache["importance"]["feature"].values

    # 降序 → 水平条形图自下而上需要升序
    imp_df = pd.DataFrame({
        "feature": all_features,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=True)

    # 过滤零重要性特征 (如 act2_*)
    imp_df = imp_df[imp_df["mean_abs_shap"] > 1e-10]

    n_features = len(imp_df)
    print(f"  有效特征数: {n_features} (排除零重要性)")

    # ── 绘图 ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 0.35 * n_features + 1.5))

    bars = ax.barh(
        range(n_features),
        imp_df["mean_abs_shap"].values,
        height=0.7,
        color=MORANDI_SLATE,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.9,
    )

    # 在条形末端标注数值
    for i, (val,) in enumerate(zip(imp_df["mean_abs_shap"].values)):
        ax.text(
            val + max(importance) * 0.01,
            i,
            f"{val:.1f}",
            va="center",
            fontsize=8,
            color="#555555",
        )

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(imp_df["feature"].values, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, imp_df["mean_abs_shap"].max() * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="-")

    ax.set_title("GBDT SHAP Feature Importance (Global)", fontsize=13,
                 fontweight="bold", pad=12)

    # ── 保存 ──────────────────────────────────────────────────────────────
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / "shap_bar_plot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {output_path}")

    plt.close("all")
    return fig


def figure_7_residuals() -> plt.Figure:
    """Figure 7: GBDT 残差分析 — 2×2 面板 (300 DPI)

    1. 残差 vs 拟合值
    2. Q-Q 图
    3. 残差直方图 + KDE
    4. Cook's 距离
    """

    import joblib
    from scipy import stats as scipy_stats
    from sklearn.metrics import r2_score, mean_squared_error
    from src.data_loader import load_and_clean
    from src.config import ROOT

    print("加载 GBDT 最终模型...")
    df = load_and_clean()
    X_raw = df.drop(columns=[TARGET])
    y_true = df[TARGET].values

    gbdt_path = ROOT / "outputs" / "models" / "GBDT_final.pkl"
    gbdt_pipe = joblib.load(gbdt_path)

    preds = gbdt_pipe.predict(X_raw)
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.ravel()

    residuals = y_true - preds
    r2 = r2_score(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))

    # ── Cook's 距离计算 (基于 LinearRegression 近似) ──────────────────────
    # 先获取变换后的特征矩阵
    imputer = gbdt_pipe.named_steps["impute"]
    engineer = gbdt_pipe.named_steps["features"]
    preprocessor = gbdt_pipe.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(engineer.transform(imputer.transform(X_raw)))
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_transformed, y_true)
    lr_preds = lr.predict(X_transformed)

    n = len(y_true)
    p = X_transformed.shape[1]
    leverage = np.diag(X_transformed @ np.linalg.pinv(X_transformed.T @ X_transformed) @ X_transformed.T)
    mse = np.sum((y_true - lr_preds) ** 2) / (n - p)
    cooks_d = (residuals ** 2 / (p * mse)) * (leverage / (1 - leverage) ** 2)

    # ── 2×2 面板 ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # (a) 残差 vs 拟合值
    ax = axes[0, 0]
    ax.scatter(preds, residuals, c="#2E7D32", s=18, edgecolors="none", alpha=0.6, zorder=3)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    # ±2σ 参考线
    std_resid = np.std(residuals)
    ax.axhline(y=2 * std_resid, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.axhline(y=-2 * std_resid, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)

    # LOWESS 平滑
    sort_idx = np.argsort(preds)
    window = max(len(sort_idx) // 12, 8)
    x_smooth, y_smooth = [], []
    for k in range(0, len(sort_idx), window):
        chunk = sort_idx[k:k + window]
        x_smooth.append(np.median(preds[chunk]))
        y_smooth.append(np.median(residuals[chunk]))
    ax.plot(x_smooth, y_smooth, "r-", linewidth=1.8, alpha=0.7)

    ax.set_xlabel("Fitted values (mg/g)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Residuals (mg/g)", fontsize=11, fontweight="bold")
    ax.set_title("Residuals vs. Fitted", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)

    # (b) Q-Q 图 (手动计算，避免 scipy probplot 版本兼容问题)
    ax = axes[0, 1]
    n = len(residuals)
    sorted_resid = np.sort(residuals)
    # 理论正态分位数 (Filliben估计)
    i = np.arange(1, n + 1)
    p = (i - 0.3175) / (n + 0.365)
    theoretical = scipy_stats.norm.ppf(p, loc=0, scale=np.std(residuals))
    ax.scatter(theoretical, sorted_resid, c="#2E7D32", s=18, edgecolors="none", alpha=0.6, zorder=3)
    lims = [min(theoretical.min(), sorted_resid.min()) - 3,
            max(theoretical.max(), sorted_resid.max()) + 3]
    ax.plot(lims, lims, "r--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Theoretical Quantiles (Normal)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sample Quantiles", fontsize=11, fontweight="bold")
    ax.set_title("Normal Q-Q Plot", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)

    # (c) 残差直方图 + KDE
    ax = axes[1, 0]
    ax.hist(residuals, bins=25, density=True, color="#2E7D32", edgecolor="white",
            alpha=0.7, linewidth=0.5)
    # KDE 曲线
    kde_x = np.linspace(residuals.min(), residuals.max(), 200)
    kde = scipy_stats.gaussian_kde(residuals)
    ax.plot(kde_x, kde(kde_x), "r-", linewidth=1.8)
    # 正态参考
    norm_x = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(norm_x, scipy_stats.norm.pdf(norm_x, loc=0, scale=np.std(residuals)),
            "k--", linewidth=1.0, alpha=0.5, label="Normal PDF")
    ax.set_xlabel("Residuals (mg/g)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=9)

    # (d) Cook's 距离
    ax = axes[1, 1]
    colors_cook = np.where(cooks_d > 4 / n, "#C62828", "#2E7D32")
    ax.stem(range(n), cooks_d, linefmt="gray", markerfmt=" ", basefmt=" ")
    ax.scatter(range(n), cooks_d, c=colors_cook, s=12, edgecolors="none", alpha=0.7, zorder=3)
    ax.axhline(y=4 / n, color="red", linewidth=0.8, linestyle="--", alpha=0.7,
               label=f"4/n = {4/n:.4f}")
    n_high = np.sum(cooks_d > 4 / n)
    ax.set_xlabel("Sample index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cook's Distance", fontsize=11, fontweight="bold")
    ax.set_title(f"Cook's Distance ({n_high} influential points)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=9)

    fig.suptitle(f"Residual Diagnostics — GBDT (R²={r2:.4f}, RMSE={rmse:.1f})",
                 fontsize=14, fontweight="bold", y=0.99)

    # ── 保存 ─────────────────────────────────────────────────────────────
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / "Figure_7_Residuals.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {output_path}")

    plt.close("all")
    return fig


def figure_8_pdp_ice() -> plt.Figure:
    """Figure 8: Top 4 特征部分依赖图 (PDP) + ICE 曲线 — 2×2 面板 (300 DPI)"""

    import joblib
    from sklearn.inspection import partial_dependence
    from src.config import ROOT

    # ★ 从缓存加载 SHAP 重要性（获取 Top 4 特征）
    cache = _load_shap_cache()
    importance_df = cache["importance"]
    all_features = cache["feature_names"]
    dropped_features = cache["dropped_features"]

    # 过滤共线特征后的 Top 4
    importance_clean = importance_df[~importance_df["feature"].isin(dropped_features)]
    top4 = importance_clean["feature"].head(4).tolist()
    top4_indices = [all_features.index(f) for f in top4]
    print(f"  Top 4 特征 (SHAP, 有效特征): {top4}")

    # PDP 需要模型，仅加载模型和前处理（不需要重算 SHAP）
    print("  加载 GBDT 模型用于 PDP 计算...")
    gbdt_path = ROOT / "outputs" / "models" / "GBDT_final.pkl"
    gbdt_pipe = joblib.load(gbdt_path)
    model = gbdt_pipe.named_steps["model"]
    X_transformed = cache["X_transformed"]

    # ── 2×2 面板 ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    n_ice = min(60, X_transformed.shape[0])
    rng = np.random.RandomState(42)
    ice_indices = rng.choice(X_transformed.shape[0], size=n_ice, replace=False)

    for ax, feat_name, feat_idx in zip(axes, top4, top4_indices):
        # 计算 PDP
        pdp_result = partial_dependence(
            model, X_transformed, features=[feat_idx],
            kind="both", grid_resolution=40,
        )

        grid = pdp_result["grid_values"][0]
        pd_values = pdp_result["average"].ravel()
        ice_values = pdp_result["individual"][0]  # shape: (n_samples, n_grid)

        # ICE 曲线 (随机子集，浅色)
        for k in range(n_ice):
            ax.plot(grid, ice_values[ice_indices[k], :],
                    color="gray", linewidth=0.25, alpha=0.3)

        # PDP 均值线 (粗线)
        ax.plot(grid, pd_values, color="#C62828", linewidth=2.5, label="PDP (mean)")

        # 特征值分布 rug (底部)
        feat_vals = X_transformed.iloc[:, feat_idx].values
        ax.axhline(y=ax.get_ylim()[0] if ax.get_ylim()[0] < ax.get_ylim()[1] else 0,
                   color="black", linewidth=0.3, alpha=0.3)
        y_min, y_max = ax.get_ylim()
        rug_y = y_min - 0.02 * (y_max - y_min)
        ax.plot(feat_vals, [rug_y] * len(feat_vals), "|", color="#424242",
                markersize=2, alpha=0.3)

        ax.set_xlabel(feat_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted CO₂ uptake (mg/g)", fontsize=11, fontweight="bold")
        ax.set_title(f"PDP + ICE: {feat_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=9)

    fig.suptitle("Partial Dependence Plots with ICE — Top 4 Features (GBDT)",
                 fontsize=14, fontweight="bold", y=0.99)

    # ── 保存 ─────────────────────────────────────────────────────────────
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / "Figure_8_PDP_ICE.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {output_path}")

    plt.close("all")
    return fig


# ============================================================================
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=str, default="0", help="指定图号 (0=全部, 或 1-8, S1)")
    args = parser.parse_args()
    fig_arg = args.figure.upper()

    if fig_arg in ("0", "S1"):
        print("=" * 60)
        print("Figure S1: 不同碳前驱体类型的 CO₂ 吸附量分布")
        print("=" * 60)
        figure_S1_precursor_distribution()

    if fig_arg == "0" or fig_arg == "1":
        print("=" * 60)
        print("Figure 1: Spearman 相关矩阵热力图")
        print("=" * 60)
        figure_1_clustermap()

    if fig_arg == "0" or fig_arg == "2":
        print("=" * 60)
        print("Figure 2: 数值特征标准化箱线图")
        print("=" * 60)
        figure_2_boxplot()

    if fig_arg == "0" or fig_arg == "3":
        print("=" * 60)
        print("Figure 3: 模型性能对比柱状图 (R² + RMSE)")
        print("=" * 60)
        figure_3_model_comparison()

    if fig_arg == "0" or fig_arg == "4":
        print("=" * 60)
        print("Figure 4: TabPFN 边缘分布 + 残差图 (GridSpec 3×6)")
        print("=" * 60)
        figure_4_tabpfn_marginal()

    if fig_arg == "0" or fig_arg == "5":
        print("=" * 60)
        print("Figure 5: GBDT SHAP beeswarm 汇总图")
        print("=" * 60)
        figure_5_shap_beeswarm()

    if fig_arg == "0" or fig_arg == "6":
        print("=" * 60)
        print("Figure 6: Top 3 特征 SHAP 依赖图")
        print("=" * 60)
        figure_6_shap_dependence()

    if fig_arg == "0" or fig_arg == "7":
        print("=" * 60)
        print("Figure 7: GBDT 残差分析 2×2 面板")
        print("=" * 60)
        figure_7_residuals()

    if fig_arg == "0" or fig_arg == "8":
        print("=" * 60)
        print("Figure 8: Top 4 特征 PDP + ICE")
        print("=" * 60)
        figure_8_pdp_ice()

    if fig_arg == "0" or fig_arg.upper() == "BAR":
        print("=" * 60)
        print("Figure BAR: GBDT SHAP 全局特征重要性条形图")
        print("=" * 60)
        figure_shap_bar()

    print("完成。")
