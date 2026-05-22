"""步骤6: SHAP可解释性分析

1. VIF多重共线性诊断
2. Spearman相关系数 + Ward层次聚类 → 共线特征簇
3. TabPFN排列特征重要性
4. GBDT TreeExplainer SHAP值
5. TabPFN vs GBDT 归因一致性检验（Spearman秩相关）
6. 所有中间结果保存至 outputs/tables/
"""

import sys
import io
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
import joblib
import shap

from src.config import SEED, TARGET, ROOT
from src.models import _get_column_lists

warnings.filterwarnings("ignore")

# ============================================================================
# 1. 多重共线性诊断
# ============================================================================


def compute_vif(X_num: pd.DataFrame) -> pd.DataFrame:
    """对数值特征计算方差膨胀因子（VIF = 1 / (1 - R²)）。

    对每个特征，用其余数值特征做线性回归，计算决定系数 R²。
    VIF > 10 标记为高度共线。
    """
    n = X_num.shape[1]
    results = []
    for i, col in enumerate(X_num.columns):
        y = X_num[col].values
        X_rest = X_num.drop(columns=[col]).values

        model = LinearRegression()
        model.fit(X_rest, y)
        y_pred = model.predict(X_rest)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        vif = 1.0 / (1.0 - r2) if r2 < 0.999 else np.inf

        results.append({
            "feature": col,
            "VIF": round(vif, 2),
            "R²": round(r2, 4),
            "high_collinear": vif > 10,
        })

    return pd.DataFrame(results).sort_values("VIF", ascending=False).reset_index(drop=True)


def compute_spearman_clusters(X_num: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """计算 Spearman 相关系数矩阵，用 Ward 层次聚类识别共线特征簇。

    Parameters
    ----------
    threshold : float
        距离阈值，用于切分簇（默认 0.7 对应相关系数 0.3）。

    Returns
    -------
    DataFrame with columns: feature, cluster_id, cluster_members
    """
    corr = X_num.corr(method="spearman")
    # NaN 可能来自方差为零的特征 → 填充 0（无相关性）
    corr = corr.fillna(0.0)
    # 转换为距离矩阵 (1 - |r|)
    dist = 1 - np.abs(corr.values)
    # 确保对角线为零且对称
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2

    # Ward 层次聚类
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    # 切分簇
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")
    # 簇号重排，使大簇优先显示
    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    cluster_order = unique_clusters[np.argsort(-counts)]
    remap = {old: new for new, old in enumerate(cluster_order, 1)}
    cluster_ids = np.array([remap[c] for c in cluster_ids])

    # 构建结果
    features = X_num.columns.tolist()
    cluster_map: dict[int, list[str]] = {}
    for feat, cid in zip(features, cluster_ids):
        cluster_map.setdefault(int(cid), []).append(feat)

    rows = []
    for feat, cid in zip(features, cluster_ids):
        members = cluster_map[int(cid)]
        rows.append({
            "feature": feat,
            "cluster_id": int(cid),
            "cluster_size": len(members),
            "cluster_members": ", ".join(members),
        })

    return pd.DataFrame(rows).sort_values(["cluster_id", "feature"]).reset_index(drop=True), Z, corr


# ============================================================================
# 2. TabPFN 排列重要性
# ============================================================================


def compute_tabpfn_permutation_importance(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 30, random_state: int = SEED
) -> pd.DataFrame:
    """对 TabPFN 模型计算排列特征重要性（RMSE增量）。

    使用 sklearn.inspection.permutation_importance。
    """
    print("  计算 TabPFN 排列重要性 (n_repeats=5, n_jobs=1)...")
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )

    df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # 负RMSE增量 → 正值 = RMSE增加越多 = 特征越重要
    return df


# ============================================================================
# 3. GBDT SHAP 分析
# ============================================================================


def compute_gbdt_shap(
    pipeline, X: pd.DataFrame, feature_names: list[str]
) -> dict:
    """用 TreeExplainer 计算 GBDT 模型的 SHAP 值。

    pipeline 结构: impute → features → preprocessor → model
    X: 原始DataFrame（未经任何预处理）
    feature_names: FeatureEngineer 输出后的特征名列表

    Returns
    -------
    dict with keys: shap_values, explainer, X_transformed, shap_importance
    """
    print("  计算 GBDT SHAP 值 (TreeExplainer)...")

    imputer = pipeline.named_steps["impute"]
    engineer = pipeline.named_steps["features"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # 逐级变换: raw → imputed → feature-engineered → model-ready
    X_imputed = imputer.transform(X)
    X_fe = engineer.transform(X_imputed)
    X_transformed = preprocessor.transform(X_fe)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_transformed)

    # 特征重要性 = mean(|SHAP|)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0] if shap_vals.shape[2] == 1 else shap_vals[:, :, 0]

    importance = np.abs(shap_vals).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "shap_importance_mean": importance,
    }).sort_values("shap_importance_mean", ascending=False).reset_index(drop=True)

    return {
        "shap_values": shap_vals,
        "explainer": explainer,
        "X_transformed": X_transformed,
        "shap_importance": shap_importance,
    }


# ============================================================================
# 4. 一致性检验
# ============================================================================


def compute_consistency(
    tabpfn_imp: pd.DataFrame, gbdt_shap_imp: pd.DataFrame
) -> dict:
    """计算 TabPFN 排列重要性 与 GBDT SHAP 重要性 之间的 Spearman 秩相关系数。"""
    # 合并两个排名表
    merged = tabpfn_imp.merge(gbdt_shap_imp, on="feature", how="inner")

    rho, pval = spearmanr(
        merged["importance_mean"], merged["shap_importance_mean"]
    )

    return {
        "spearman_rho": rho,
        "spearman_pval": pval,
        "n_features": len(merged),
        "merged_table": merged,
    }


# ============================================================================
# 主入口
# ============================================================================


def main():
    from src.data_loader import load_and_clean

    print("=" * 60)
    print("步骤6: SHAP 可解释性分析")
    print("=" * 60)

    # ---- 数据准备（仅加载，预处理由最终模型Pipeline内部完成）----
    print("\n[1/5] 加载数据...")
    df = load_and_clean()
    X_raw = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 用已保存的GBDT最终模型Pipeline做预处理以获取数值特征（供VIF/聚类使用）
    gbdt_path = ROOT / "outputs" / "models" / "GBDT_final.pkl"
    gbdt_pipe = joblib.load(gbdt_path)
    # 仅跑Pipeline的前两步: impute → features
    imputer = gbdt_pipe.named_steps["impute"]
    engineer = gbdt_pipe.named_steps["features"]
    X_imputed = imputer.transform(X_raw)
    X_fe = engineer.transform(X_imputed)

    cat_cols, num_cols = _get_column_lists(X_fe)
    print(f"  数值特征 ({len(num_cols)}): {num_cols}")
    print(f"  分类特征 ({len(cat_cols)}): {cat_cols}")
    print(f"  样本数: {len(X_raw)}, 输入特征数: {X_raw.shape[1]}")

    # ---- 输出目录 ----
    tables_dir = ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 2a. VIF 多重共线性诊断
    # ========================================================================
    print("\n[2/5] VIF 多重共线性诊断...")
    X_num = X_fe[num_cols].copy()
    # 填充正则提取留下的 NaN（工艺未执行 → 中位数填充，仅用于 VIF 计算）
    X_num_imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_num),
        columns=X_num.columns,
        index=X_num.index,
    )
    vif_table = compute_vif(X_num_imputed)

    n_high = vif_table["high_collinear"].sum()
    print(f"  共 {n_high} 个特征 VIF > 10:")
    high_vif = vif_table[vif_table["high_collinear"]]
    for _, row in high_vif.iterrows():
        print(f"    {row['feature']}: VIF={row['VIF']:.1f}")

    vif_table.to_csv(tables_dir / "vif_table.csv", index=False, encoding="utf-8-sig")
    print(f"  已保存: {tables_dir / 'vif_table.csv'}")

    # ========================================================================
    # 2b. Spearman 聚类
    # ========================================================================
    print("\n[3/5] Spearman 相关系数 + Ward 层次聚类...")
    cluster_table, Z, spearman_corr = compute_spearman_clusters(X_num_imputed, threshold=0.7)

    n_clusters = cluster_table["cluster_id"].nunique()
    print(f"  识别出 {n_clusters} 个共线特征簇:")
    for cid in sorted(cluster_table["cluster_id"].unique()):
        members = cluster_table[cluster_table["cluster_id"] == cid]["feature"].tolist()
        print(f"    簇{cid}: {members}")

    cluster_table.to_csv(tables_dir / "feature_clusters.csv", index=False, encoding="utf-8-sig")
    spearman_corr.to_csv(tables_dir / "spearman_correlation.csv", encoding="utf-8-sig")
    print(f"  已保存: {tables_dir / 'feature_clusters.csv'}")
    print(f"  已保存: {tables_dir / 'spearman_correlation.csv'}")

    # ========================================================================
    # 3a. TabPFN 排列重要性（输入原始DataFrame，Pipeline内部处理预处理）
    # ========================================================================
    print("\n[4/5] TabPFN 排列重要性...")
    tabpfn_csv = tables_dir / "permutation_importance_tabpfn.csv"
    if tabpfn_csv.exists():
        print("  从缓存加载 TabPFN 排列重要性...")
        tabpfn_imp = pd.read_csv(tabpfn_csv)
    else:
        tabpfn_path = ROOT / "outputs" / "models" / "TabPFN_final.pkl"
        tabpfn_pipe = joblib.load(tabpfn_path)
        tabpfn_imp = compute_tabpfn_permutation_importance(
            tabpfn_pipe, X_raw, y, n_repeats=5, random_state=SEED
        )
        tabpfn_imp.to_csv(tabpfn_csv, index=False, encoding="utf-8-sig")
        print(f"  已保存: {tabpfn_csv}")

    print("  Top 5 重要特征 (TabPFN):")
    for _, row in tabpfn_imp.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance_mean']:.2f} +- {row['importance_std']:.2f}")

    # ========================================================================
    # 3b. GBDT SHAP 分析 —— 使用 FeatureEngineer 后的特征名
    # ========================================================================
    # GBDT 管道B: 变换后特征顺序 = 分类列 (OrdinalEncoded) + 数值列 (passthrough)
    feature_names = cat_cols + num_cols

    shap_csv = tables_dir / "shap_importance_gbdt.csv"
    if shap_csv.exists():
        print("  从缓存加载 GBDT SHAP 重要性...")
        shap_result = {
            "shap_importance": pd.read_csv(shap_csv),
            "shap_values": pd.read_csv(tables_dir / "shap_values_gbdt.csv").values,
        }
    else:
        print("\n[5/5] GBDT SHAP 分析 (TreeExplainer)...")
        # gbdt_pipe 已在上面加载，传入原始DataFrame
        shap_result = compute_gbdt_shap(gbdt_pipe, X_raw, feature_names)

        shap_df = pd.DataFrame(shap_result["shap_values"], columns=feature_names)
        shap_df.to_csv(tables_dir / "shap_values_gbdt.csv", index=False, encoding="utf-8-sig")
        shap_result["shap_importance"].to_csv(shap_csv, index=False, encoding="utf-8-sig")
        print(f"  已保存: {tables_dir / 'shap_values_gbdt.csv'}")
        print(f"  已保存: {shap_csv}")

    print("  Top 5 重要特征 (GBDT SHAP):")
    for _, row in shap_result["shap_importance"].head(5).iterrows():
        print(f"    {row['feature']}: {row['shap_importance_mean']:.4f}")

    # ========================================================================
    # 4. 一致性检验
    # ========================================================================
    print("\n" + "=" * 60)
    print("一致性检验: TabPFN 排列重要性 vs GBDT SHAP")
    print("=" * 60)

    consistency = compute_consistency(tabpfn_imp, shap_result["shap_importance"])
    rho = consistency["spearman_rho"]
    pval = consistency["spearman_pval"]

    print(f"  Spearman ρ = {rho:.4f}  (p = {pval:.4f})")
    if rho > 0.6:
        print(f"  [OK] ρ > 0.6 — 两种范式归因高度一致")
    elif rho > 0.4:
        print(f"  [WARN] 0.4 < ρ ≤ 0.6 — 中等一致，需在论文中讨论差异")
    else:
        print(f"  [FAIL] ρ ≤ 0.4 — 归因不一致，基础模型与传统模型可能关注不同特征")

    # 并排对比表
    merged = consistency["merged_table"].copy()
    merged["tabpfn_rank"] = merged["importance_mean"].rank(ascending=False)
    merged["gbdt_rank"] = merged["shap_importance_mean"].rank(ascending=False)
    merged["rank_diff"] = (merged["tabpfn_rank"] - merged["gbdt_rank"]).abs()

    print(f"\n  Top 10 特征排名对比:")
    print(f"  {'特征':<25} {'TabPFN排名':>10} {'GBDT排名':>10} {'排名差':>8}")
    print(f"  {'-' * 55}")
    for _, row in merged.sort_values("importance_mean", ascending=False).head(10).iterrows():
        print(f"  {row['feature']:<25} {row['tabpfn_rank']:10.0f} {row['gbdt_rank']:10.0f} {row['rank_diff']:8.0f}")

    merged.to_csv(tables_dir / "consistency_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {tables_dir / 'consistency_comparison.csv'}")

    # ========================================================================
    # 最终汇总
    # ========================================================================
    print(f"\n{'=' * 60}")
    print("步骤6完成。所有结果已保存至 outputs/tables/")
    print(f"{'=' * 60}")
    print(f"\n  已保存文件:")
    for f in sorted(tables_dir.glob("*.csv")):
        print(f"    {f.name}")

    # 关键指标
    print(f"\n  关键指标摘要:")
    print(f"    VIF > 10 的特征数: {n_high}")
    print(f"    共线特征簇数: {n_clusters}")
    print(f"    TabPFN vs GBDT Spearman ρ: {rho:.4f}")
    print(f"    验证标准 ρ > 0.6: {'[OK] 通过' if rho > 0.6 else '[FAIL] 未通过'}")

    # 簇级SHAP汇总提示
    if n_clusters < len(num_cols):
        print(f"\n  [WARN] 存在共线簇（{n_clusters}个簇 vs {len(num_cols)}个数值特征），"
              f"步骤7出图时请生成簇级SHAP汇总图。")


if __name__ == "__main__":
    main()
