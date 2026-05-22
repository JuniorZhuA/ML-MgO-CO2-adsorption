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


def figure_4_pred_vs_exp() -> plt.Figure:
    """Figure 4: 预测值 vs 实验值散点图 — 3×3 网格，按 SBET 着色 (300 DPI)"""

    import joblib
    from sklearn.metrics import r2_score, mean_squared_error
    from src.data_loader import load_and_clean
    from src.preprocessing import MissingValueImputer
    from src.feature_engineering import FeatureEngineer
    from src.config import ROOT

    print("加载数据与预处理...")
    df = load_and_clean()
    y_true = df[TARGET].values

    # 独立预处理以获取 SBET 用于着色
    df_processed = MissingValueImputer().fit_transform(df)
    df_processed = FeatureEngineer().fit_transform(df_processed)
    sbet = df_processed["SBET_m2_g"].values

    # 加载模型并预测
    models_dir = ROOT / "outputs" / "models"
    model_names = ["TabPFN", "GBDT", "RF", "XGBoost", "LightGBM", "GPR", "SVR", "Ridge", "Lasso"]
    pipeline_labels = ["D", "B", "B", "B", "B", "C", "C", "A", "A"]

    predictions: dict[str, np.ndarray] = {}
    for name in model_names:
        path = models_dir / f"{name}_final.pkl"
        pipe = joblib.load(path)
        X_raw = df.drop(columns=[TARGET])
        preds = pipe.predict(X_raw)
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.ravel()
        predictions[name] = preds
        print(f"  {name}: R²={r2_score(y_true, preds):.4f}, RMSE={np.sqrt(mean_squared_error(y_true, preds)):.1f}")

    # ── 绘图 ─────────────────────────────────────────────────────────────
    n_models = len(model_names)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 17))
    axes = axes.flatten()

    # SBET 颜色归一化
    sbet_clean = sbet[~np.isnan(sbet)]
    norm = plt.Normalize(sbet_clean.min(), sbet_clean.max())
    cmap = plt.cm.viridis

    for idx, (name, ax) in enumerate(zip(model_names, axes)):
        preds = predictions[name]
        r2 = r2_score(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))

        # 有效 SBET 的点用颜色映射，NaN 用灰色
        mask_valid = ~np.isnan(sbet)
        scatter = ax.scatter(
            y_true[mask_valid], preds[mask_valid],
            c=sbet[mask_valid], cmap=cmap, norm=norm,
            s=18, edgecolors="none", alpha=0.75, zorder=3,
        )
        if not mask_valid.all():
            ax.scatter(
                y_true[~mask_valid], preds[~mask_valid],
                c="#BDBDBD", s=18, edgecolors="none", alpha=0.5, zorder=2,
            )

        # y=x 对角线
        lims = [min(y_true.min(), preds.min()) - 5,
                max(y_true.max(), preds.max()) + 5]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.6, zorder=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # 标注
        ax.text(0.05, 0.92, f"{name}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85))
        ax.text(0.05, 0.80, f"R²={r2:.4f}\nRMSE={rmse:.1f}", transform=ax.transAxes,
                fontsize=8.5, va="top", color="#424242",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7))

        ax.set_xlabel("Experimental CO₂ uptake (mg/g)", fontsize=9, fontweight="bold")
        ax.set_ylabel("Predicted CO₂ uptake (mg/g)", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")

    # 隐藏多余子图
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    # 统一颜色条
    cbar_ax = fig.add_axes([0.92, 0.08, 0.012, 0.84])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("SBET (m²/g)", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.suptitle("Predicted vs. Experimental CO₂ Uptake", fontsize=15,
                 fontweight="bold", y=0.98)

    # ── 保存 ─────────────────────────────────────────────────────────────
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / "Figure_4_Pred_vs_Exp.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] 已保存: {output_path}")
    print(f"     分辨率: 300 DPI, 模型数: {n_models}")

    plt.close("all")
    return fig


def figure_5_shap_beeswarm() -> plt.Figure:
    """Figure 5: GBDT SHAP beeswarm 汇总图 — Top 15 特征 (300 DPI)"""

    import joblib
    import shap
    from src.data_loader import load_and_clean
    from src.config import ROOT

    print("加载 GBDT 最终模型...")
    df = load_and_clean()
    X_raw = df.drop(columns=[TARGET])
    y = df[TARGET]

    gbdt_path = ROOT / "outputs" / "models" / "GBDT_final.pkl"
    gbdt_pipe = joblib.load(gbdt_path)

    # 逐级变换
    imputer = gbdt_pipe.named_steps["impute"]
    engineer = gbdt_pipe.named_steps["features"]
    preprocessor = gbdt_pipe.named_steps["preprocessor"]
    model = gbdt_pipe.named_steps["model"]

    X_imputed = imputer.transform(X_raw)
    X_fe = engineer.transform(X_imputed)
    X_transformed = preprocessor.transform(X_fe)

    # 获取特征名
    cat_cols, num_cols = _get_column_lists(X_fe)
    feature_names = cat_cols + num_cols

    # 确保 X_transformed 是 DataFrame
    if not isinstance(X_transformed, pd.DataFrame):
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

    # 计算 SHAP
    print("  计算 TreeExplainer SHAP 值...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_transformed)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

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

    import joblib
    import shap
    from src.data_loader import load_and_clean
    from src.config import ROOT

    print("加载 GBDT 最终模型...")
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
    all_features = cat_cols + num_cols
    if not isinstance(X_transformed, pd.DataFrame):
        X_transformed = pd.DataFrame(X_transformed, columns=all_features)

    # 计算 SHAP
    print("  计算 SHAP 值...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_transformed)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

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
    from src.data_loader import load_and_clean
    from src.config import ROOT

    print("加载 GBDT 最终模型...")
    df = load_and_clean()
    X_raw = df.drop(columns=[TARGET])

    gbdt_path = ROOT / "outputs" / "models" / "GBDT_final.pkl"
    gbdt_pipe = joblib.load(gbdt_path)

    # 获取变换后的特征矩阵
    imputer = gbdt_pipe.named_steps["impute"]
    engineer = gbdt_pipe.named_steps["features"]
    preprocessor = gbdt_pipe.named_steps["preprocessor"]
    model = gbdt_pipe.named_steps["model"]

    X_transformed = preprocessor.transform(engineer.transform(imputer.transform(X_raw)))
    cat_cols, num_cols = _get_column_lists(engineer.transform(imputer.transform(X_raw)))
    all_features = cat_cols + num_cols
    if not isinstance(X_transformed, pd.DataFrame):
        X_transformed = pd.DataFrame(X_transformed, columns=all_features)

    # Top 4 特征 (from SHAP Figure 5)
    top4 = ["pressure_bar", "microporosity", "temperature_C", "T_lnP"]
    top4_indices = [all_features.index(f) for f in top4]

    print(f"  Top 4 特征: {top4}")

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

    if args.figure == 0 or args.figure == 3:
        print("=" * 60)
        print("Figure 3: 模型性能对比柱状图 (R² + RMSE)")
        print("=" * 60)
        figure_3_model_comparison()

    if args.figure == 0 or args.figure == 4:
        print("=" * 60)
        print("Figure 4: 预测值 vs 实验值散点图 (按 SBET 着色)")
        print("=" * 60)
        figure_4_pred_vs_exp()

    if args.figure == 0 or args.figure == 5:
        print("=" * 60)
        print("Figure 5: GBDT SHAP beeswarm 汇总图")
        print("=" * 60)
        figure_5_shap_beeswarm()

    if args.figure == 0 or args.figure == 6:
        print("=" * 60)
        print("Figure 6: Top 3 特征 SHAP 依赖图")
        print("=" * 60)
        figure_6_shap_dependence()

    if args.figure == 0 or args.figure == 7:
        print("=" * 60)
        print("Figure 7: GBDT 残差分析 2×2 面板")
        print("=" * 60)
        figure_7_residuals()

    if args.figure == 0 or args.figure == 8:
        print("=" * 60)
        print("Figure 8: Top 4 特征 PDP + ICE")
        print("=" * 60)
        figure_8_pdp_ice()

    print("完成。")
