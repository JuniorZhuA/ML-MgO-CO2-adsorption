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


def diagnose_and_remove_collinear(X_num: pd.DataFrame) -> "tuple[pd.DataFrame, list[str]]":
    """自动诊断并移除恶意的共线特征。

    策略：
    1. 零方差特征（所有样本值相同）→ 直接剔除
    2. 完美线性依赖（VIF=INF）→ 领域规则优先剔除，迭代清理残余
       - Vtotal = Vmicro + Vmeso → 保留细分特征，剔除 Vtotal
    3. 迭代重算VIF，直到无INF特征为止

    Returns
    -------
    X_clean : DataFrame, 清洗后的特征矩阵
    dropped : list[str], 被剔除的特征名列表
    """
    dropped = []
    X_clean = X_num.copy()

    # ---- 1. 零方差检测 ----
    zero_var = []
    for col in X_clean.columns:
        if X_clean[col].std() < 1e-10:
            zero_var.append(col)
    if zero_var:
        X_clean = X_clean.drop(columns=zero_var)
        dropped.extend(zero_var)
        print(f"\n  [剔除] 零方差特征 ({len(zero_var)}个):")
        for col in zero_var:
            print(f"         {col}: std={X_num[col].std():.2e}, 所有样本值={X_num[col].iloc[0]:.4f}")

    # ---- 2. 完美线性关系（领域知识）----
    # Vmeso = Vtotal - Vmicro → 三者构成完美线性依赖，保留细分特征
    pore_cols = ["Vtotal_cm3_g", "Vmicro_cm3_g", "Vmeso_cm3_g"]
    if all(c in X_clean.columns for c in pore_cols):
        X_clean = X_clean.drop(columns=["Vtotal_cm3_g"])
        dropped.append("Vtotal_cm3_g")
        print(f"\n  [剔除] 完美线性依赖 Vtotal = Vmicro + Vmeso → 保留 Vmicro、Vmeso，剔除 Vtotal")

    # ---- 3. 迭代清理残余 INF ----
    max_iter = 5
    for iteration in range(max_iter):
        vif_check = compute_vif(X_clean)
        still_inf = vif_check[vif_check["VIF"] == np.inf]
        still_inf = still_inf[~still_inf["feature"].isin(dropped)]
        if len(still_inf) == 0:
            break
        print(f"\n  [迭代{iteration + 1}] 仍有 {len(still_inf)} 个 VIF=INF 特征，继续剔除:")
        for _, row in still_inf.iterrows():
            if row["feature"] in X_clean.columns:
                X_clean = X_clean.drop(columns=[row["feature"]])
                dropped.append(row["feature"])
                r2_str = f"{row['R²']:.4f}" if not np.isnan(row["R²"]) else "NaN"
                print(f"         {row['feature']}: VIF=INF, R²={r2_str}")

    # ---- 4. VIF > 100 警告 ----
    vif_final = compute_vif(X_clean)
    high_vif = vif_final[vif_final["VIF"] > 100]
    if len(high_vif) > 0:
        print(f"\n  [警告] {len(high_vif)} 个特征 VIF > 100 (已保留，仅警示):")
        for _, row in high_vif.iterrows():
            print(f"         {row['feature']}: VIF={row['VIF']:.1f}")

    # ---- 5. 输出总结 ----
    if dropped:
        print(f"\n  === 共剔除 {len(dropped)} 个特征: {dropped} ===")
        print(f"  剩余数值特征: {X_clean.shape[1]} (原 {X_num.shape[1]})")
    else:
        print(f"\n  未发现需剔除的共线特征。")

    return X_clean, dropped


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

    手动实现（非 sklearn），支持实时进度输出，方便监控长时间运行。
    """
    from sklearn.metrics import root_mean_squared_error

    rng = np.random.RandomState(random_state)
    feature_names = list(X.columns)
    n_features = len(feature_names)
    n_samples = len(X)

    # 基准 RMSE
    y_pred_base = model.predict(X)
    base_rmse = root_mean_squared_error(y, y_pred_base)
    print(f"  基准 RMSE: {base_rmse:.2f}")
    print(f"  开始 {n_repeats} 次重复 × {n_features} 特征 = {n_repeats * n_features} 次排列...")

    # 存储: (n_features, n_repeats)
    scores = np.full((n_features, n_repeats), np.nan)

    t_start = time.time()
    for rep in range(n_repeats):
        rep_start = time.time()
        for feat_idx in range(n_features):
            X_perm = X.copy()
            X_perm.iloc[:, feat_idx] = rng.permutation(X_perm.iloc[:, feat_idx].values)
            y_pred_perm = model.predict(X_perm)
            scores[feat_idx, rep] = root_mean_squared_error(y, y_pred_perm) - base_rmse

        rep_elapsed = time.time() - rep_start
        total_elapsed = time.time() - t_start
        avg_per_rep = total_elapsed / (rep + 1)
        eta = avg_per_rep * (n_repeats - rep - 1)

        print(f"    重复 {rep + 1:2d}/{n_repeats} | "
              f"耗时 {rep_elapsed:.0f}s | "
              f"累计 {total_elapsed/60:.1f}min | "
              f"预计剩余 {eta/60:.1f}min | "
              f"当前Top3: {', '.join(feature_names[i] for i in np.argsort(-scores[:, rep])[:3])}",
              flush=True)

    # 汇总: mean 和 std across repeats
    importance_mean = scores.mean(axis=1)
    importance_std = scores.std(axis=1, ddof=1)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importance_mean,
        "importance_std": importance_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    total_time = time.time() - t_start
    print(f"  完成! 总耗时 {total_time/60:.1f} 分钟", flush=True)

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

    # 确保 X_transformed 是 DataFrame（缓存时需要列名）
    if not isinstance(X_transformed, pd.DataFrame):
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

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
    # 安全设置UTF-8输出
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass  # 部分环境不支持 reconfigure（如某些 IDE 终端）

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
    # 2a. VIF 多重共线性诊断（含自动共线特征剔除）
    # ========================================================================
    print("\n[2/5] VIF 多重共线性诊断 + 自动剔除...")
    X_num = X_fe[num_cols].copy()
    # 填充正则提取留下的 NaN（工艺未执行 → 中位数填充，仅用于 VIF 计算）
    X_num_imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_num),
        columns=X_num.columns,
        index=X_num.index,
    )

    # ---- 自动诊断并剔除完美共线特征 ----
    X_num_clean, dropped_features = diagnose_and_remove_collinear(X_num_imputed)

    # 在清洗后的特征集上计算 VIF
    vif_table = compute_vif(X_num_clean)

    n_high = vif_table["high_collinear"].sum()
    print(f"\n  清洗后共 {n_high} 个特征 VIF > 10:")
    if n_high > 0:
        high_vif = vif_table[vif_table["high_collinear"]]
        for _, row in high_vif.iterrows():
            print(f"    {row['feature']}: VIF={row['VIF']:.1f}")

    vif_table.to_csv(tables_dir / "vif_table.csv", index=False, encoding="utf-8-sig")
    print(f"  已保存: {tables_dir / 'vif_table.csv'}")

    # 构建清洗后的数值特征列表（用于聚类分析）
    num_cols_clean = [c for c in num_cols if c not in dropped_features]
    # 分类特征中可能也有共线特征？检查 dropped 中是否有分类列
    cat_cols_clean = [c for c in cat_cols if c not in dropped_features]

    # ========================================================================
    # 2b. Spearman 聚类 (阈值收紧至 0.3，对应 |ρ| ≥ 0.7)
    # ========================================================================
    print("\n[3/5] Spearman 相关系数 + Ward 层次聚类 (threshold=0.3)...")
    cluster_table, Z, spearman_corr = compute_spearman_clusters(X_num_clean, threshold=0.3)

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
    # 3a. TabPFN 排列重要性（n_repeats=30, 在预变换数据上运行以加速）
    # ========================================================================
    print("\n[4/5] TabPFN 排列重要性 (n_repeats=30, 预变换加速)...")
    tabpfn_csv = tables_dir / "permutation_importance_tabpfn.csv"
    # 删除旧缓存（n_repeats=5 的过期结果）
    if tabpfn_csv.exists():
        old = pd.read_csv(tabpfn_csv)
        # 旧缓存只有5次重复，std可靠性不足 → 删除重新计算
        tabpfn_csv.unlink()
        print("  已删除旧缓存 (n_repeats=5)，将重新计算...")

    if tabpfn_csv.exists():
        print("  从缓存加载 TabPFN 排列重要性...")
        tabpfn_imp = pd.read_csv(tabpfn_csv)
    else:
        tabpfn_path = ROOT / "outputs" / "models" / "TabPFN_final.pkl"
        tabpfn_pipe = joblib.load(tabpfn_path)

        # 关键优化：预计算 X_fe，避免每次repeat都重跑imputation+特征工程
        # TabPFN Pipeline D: impute → features → preprocessor → model
        tabpfn_imputer = tabpfn_pipe.named_steps["impute"]
        tabpfn_engineer = tabpfn_pipe.named_steps["features"]
        tabpfn_preprocessor = tabpfn_pipe.named_steps["preprocessor"]
        tabpfn_model = tabpfn_pipe.named_steps["model"]

        from sklearn.pipeline import Pipeline as SkPipeline
        # 构建精简管道：仅 preprocessor → model，输入已变换的 X_fe
        slim_pipe = SkPipeline([
            ("preprocessor", tabpfn_preprocessor),
            ("model", tabpfn_model),
        ])

        print("  预变换数据 (impute + feature engineering)...")
        X_tabpfn_fe = tabpfn_engineer.transform(tabpfn_imputer.transform(X_raw))
        # 确保列为字符串（OrdinalEncoder 要求）
        X_tabpfn_fe.columns = [str(c) for c in X_tabpfn_fe.columns]

        N_REPEATS = 30
        print(f"  在 {X_tabpfn_fe.shape[1]} 个特征上运行排列重要性 ({N_REPEATS} 次重复)...")
        tabpfn_imp = compute_tabpfn_permutation_importance(
            slim_pipe, X_tabpfn_fe, y, n_repeats=N_REPEATS, random_state=SEED
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
    shap_values_csv = tables_dir / "shap_values_gbdt.csv"
    X_transformed_csv = tables_dir / "X_transformed_gbdt.csv"
    feature_names_json = tables_dir / "feature_names.json"
    dropped_json = tables_dir / "dropped_features.json"

    if shap_csv.exists() and X_transformed_csv.exists():
        print("  从缓存加载 GBDT SHAP 数据...")
        shap_result = {
            "shap_importance": pd.read_csv(shap_csv),
            "shap_values": pd.read_csv(shap_values_csv).values,
            "X_transformed": pd.read_csv(X_transformed_csv),
        }
    else:
        print("\n[5/5] GBDT SHAP 分析 (TreeExplainer)...")
        # gbdt_pipe 已在上面加载，传入原始DataFrame
        shap_result = compute_gbdt_shap(gbdt_pipe, X_raw, feature_names)

        # 保存 SHAP 值矩阵
        shap_df = pd.DataFrame(shap_result["shap_values"], columns=feature_names)
        shap_df.to_csv(shap_values_csv, index=False, encoding="utf-8-sig")
        shap_result["shap_importance"].to_csv(shap_csv, index=False, encoding="utf-8-sig")
        print(f"  已保存: {shap_values_csv}")
        print(f"  已保存: {shap_csv}")

        # ★ 保存 X_transformed（关键中间结果，供绘图函数直接读取）
        X_t_df = shap_result["X_transformed"]
        if isinstance(X_t_df, pd.DataFrame):
            X_t_df.to_csv(X_transformed_csv, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(X_t_df, columns=feature_names).to_csv(
                X_transformed_csv, index=False, encoding="utf-8-sig"
            )
        print(f"  已保存: {X_transformed_csv}")

    # ★ 保存元数据（特征名列表 + 剔除特征列表，供绘图函数使用）
    import json
    with open(feature_names_json, "w", encoding="utf-8") as fj:
        json.dump({"feature_names": feature_names, "cat_cols": cat_cols, "num_cols": num_cols},
                  fj, ensure_ascii=False, indent=2)
    with open(dropped_json, "w", encoding="utf-8") as dj:
        json.dump({"dropped_features": dropped_features, "n_dropped": len(dropped_features)},
                  dj, ensure_ascii=False, indent=2)
    print(f"  已保存元数据: {feature_names_json}, {dropped_json}")

    print("  Top 5 重要特征 (GBDT SHAP):")
    for _, row in shap_result["shap_importance"].head(5).iterrows():
        print(f"    {row['feature']}: {row['shap_importance_mean']:.4f}")

    # ========================================================================
    # 4. 一致性检验（过滤共线特征后的有效特征集）
    # ========================================================================
    # 构建有效特征列表：从完整特征名中排除被剔除的特征
    feature_names_clean = [f for f in feature_names if f not in dropped_features]
    print(f"\n  一致性检验使用 {len(feature_names_clean)}/{len(feature_names)} 个有效特征 "
          f"(剔除 {len(dropped_features)} 个共线特征)")

    # 过滤两个重要性表
    tabpfn_imp_clean = tabpfn_imp[tabpfn_imp["feature"].isin(feature_names_clean)].copy()
    gbdt_imp_clean = shap_result["shap_importance"][
        shap_result["shap_importance"]["feature"].isin(feature_names_clean)
    ].copy()

    print("\n" + "=" * 60)
    print("一致性检验: TabPFN 排列重要性 vs GBDT SHAP (有效特征)")
    print("=" * 60)

    consistency = compute_consistency(tabpfn_imp_clean, gbdt_imp_clean)
    rho = consistency["spearman_rho"]
    pval = consistency["spearman_pval"]

    print(f"  Spearman ρ = {rho:.4f}  (p = {pval:.4f})")
    if rho > 0.6:
        print(f"  [OK] ρ > 0.6 — 两种范式归因高度一致")
    elif rho > 0.4:
        print(f"  [WARN] 0.4 < ρ ≤ 0.6 — 中等一致，需在论文中讨论差异")
    else:
        print(f"  [FAIL] ρ ≤ 0.4 — 归因不一致，基础模型与传统模型可能关注不同特征")

    # 并排对比表（含被剔除特征的标注）
    merged = consistency["merged_table"].copy()
    merged["tabpfn_rank"] = merged["importance_mean"].rank(ascending=False)
    merged["gbdt_rank"] = merged["shap_importance_mean"].rank(ascending=False)
    merged["rank_diff"] = (merged["tabpfn_rank"] - merged["gbdt_rank"]).abs()

    # 补充被剔除特征的信息（标记为共线已剔除）
    dropped_imp = tabpfn_imp[~tabpfn_imp["feature"].isin(feature_names_clean)].copy()
    if len(dropped_imp) > 0:
        dropped_imp["tabpfn_rank"] = float("nan")
        dropped_imp["gbdt_rank"] = float("nan")
        dropped_imp["shap_importance_mean"] = float("nan")
        dropped_imp["rank_diff"] = float("nan")
        # 确保列名与 merged 一致
        for col in merged.columns:
            if col not in dropped_imp.columns:
                dropped_imp[col] = float("nan")
        merged = pd.concat([merged, dropped_imp], ignore_index=True)

    print(f"\n  Top 10 特征排名对比 (有效特征):")
    print(f"  {'特征':<25} {'TabPFN排名':>10} {'GBDT排名':>10} {'排名差':>8}")
    print(f"  {'-' * 55}")
    top10 = merged[merged["tabpfn_rank"].notna()].sort_values("importance_mean", ascending=False).head(10)
    for _, row in top10.iterrows():
        print(f"  {row['feature']:<25} {row['tabpfn_rank']:10.0f} {row['gbdt_rank']:10.0f} {row['rank_diff']:8.0f}")

    if len(dropped_imp) > 0:
        print(f"\n  已剔除特征 (不计入排名):")
        for _, row in dropped_imp.iterrows():
            print(f"    {row['feature']}: TabPFN重要性={row['importance_mean']:.2f} (共线已剔除)")

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
    print(f"    原始数值特征数: {len(num_cols)}")
    print(f"    剔除共线特征数: {len(dropped_features)}")
    if dropped_features:
        print(f"    剔除列表: {dropped_features}")
    print(f"    清洗后数值特征数: {len(num_cols_clean)}")
    print(f"    VIF > 10 的特征数 (清洗后): {n_high}")
    print(f"    共线特征簇数 (清洗后): {n_clusters}")
    print(f"    一致性检验特征数: {len(feature_names_clean)}")
    print(f"    TabPFN vs GBDT Spearman ρ: {rho:.4f}")
    print(f"    验证标准 ρ > 0.6: {'[OK] 通过' if rho > 0.6 else '[FAIL] 未通过'}")


if __name__ == "__main__":
    main()
