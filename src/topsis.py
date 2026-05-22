"""TOPSIS 多准则决策 —— 9模型综合排名

支持三种赋权方法:
- 熵权法 (Entropy): 基于信息熵，指标离散度越大权重越大
- 层次熵权法 (Grouped Entropy) ★ 论文主方案:
  组间领域知识驱动 + 组内客观熵权，避免误差类指标的维度重复计数
- CRITIC: 同时考虑对比强度和冲突度（敏感性分析用）

基于嵌套CV的8项指标，计算各模型的TOPSIS相对贴近度并排序。
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import ROOT

# 评价指标及其方向: "benefit"(越大越好) 或 "cost"(越小越好)
CRITERIA = {
    "R²_mean":              "benefit",
    "R²_std":               "cost",
    "RMSE_mean":            "cost",
    "RMSE_std":             "cost",
    "MAE_mean":             "cost",
    "MAPE_mean":            "cost",
    "tail_rmse_q10_mean":   "cost",
    "tail_rmse_q90_mean":   "cost",
}


def _build_matrix(results: dict) -> tuple[np.ndarray, list[str], list[str]]:
    metrics = list(CRITERIA.keys())
    models = list(results.keys())
    X = np.array([
        [results[m]["aggregate"].get(metric, np.nan) for metric in metrics]
        for m in models
    ], dtype=float)
    return X, models, metrics


def _minmax_normalize(X: np.ndarray, directions: list[str]) -> np.ndarray:
    """Min-max 归一化到 [0,1]，成本型指标反向。"""
    m, n = X.shape
    r = np.zeros_like(X)
    for j in range(n):
        xj = X[:, j]
        x_min, x_max = xj.min(), xj.max()
        if directions[j] == "benefit":
            r[:, j] = (xj - x_min) / (x_max - x_min + 1e-12)
        else:
            r[:, j] = (x_max - xj) / (x_max - x_min + 1e-12)
    return r


# ============================================================================
# 赋权方法 1: 熵权法
# ============================================================================


def entropy_weights(X: np.ndarray, directions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """熵权法，返回 (权重w, 熵值e)。"""
    m, n = X.shape
    r = _minmax_normalize(X, directions) + 1e-12
    p = r / r.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(m)
    e = -k * np.sum(p * np.log(p), axis=0)
    e = np.clip(e, 0, 1)
    w = (1 - e) / np.sum(1 - e)
    return w, e


# ============================================================================
# 赋权方法 2: CRITIC (相关调整赋权)
# ============================================================================


def critic_weights(X: np.ndarray, directions: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CRITIC 赋权法，返回 (权重w, 对比强度S, 冲突度A)。

    C_j = σ_j × Σ_k(1 − |r_jk|)
    w_j = C_j / ΣC_j

    - σ_j: 指标j的标准化后标准差（对比强度）
    - |r_jk|: 指标j与k的Pearson相关系数的绝对值（冲突度 = 1−|r|）
    - 高度相关的指标互相惩罚，避免权重叠加放大
    """
    m, n = X.shape
    r_norm = _minmax_normalize(X, directions)

    # 1. 对比强度: 标准化后的标准差
    S = np.std(r_norm, axis=0, ddof=1)

    # 2. 冲突度: Σ_k(1 − |r_jk|)，使用 Pearson 相关
    corr = np.abs(np.corrcoef(X.T))  # (n, n) 绝对值相关矩阵
    A = np.sum(1.0 - corr, axis=1)   # 每个指标与其他指标的冲突总量

    # 3. 信息量
    C = S * A
    w = C / (np.sum(C) + 1e-12)

    return w, S, A


# ============================================================================
# 赋权方法 3: 层次熵权法 (Grouped Entropy) ★ 论文主方案
# ============================================================================


def grouped_entropy_weights(
    X: np.ndarray, metric_names: list[str], directions: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """层次赋权: 组间领域知识驱动 + 组内熵权客观驱动。

    设计逻辑:
    - 精度是模型选择的首要维度 → 50%
    - 稳定性和泛化能力是辅助维度 → 各25%
    - 组内用熵权法客观分配，避免主观偏向某个具体指标
    - MAE_mean 因与 RMSE_mean 高度共线 (r≈0.997) 被排除，权重=0

    Returns
    -------
    w_final : np.ndarray (len(metric_names),)
    weights_df : pd.DataFrame
    """
    # ── 分组定义 ──
    groups = {
        "预测精度 (50%)": ["R²_mean", "RMSE_mean", "MAPE_mean"],
        "稳定性 (25%)":  ["R²_std", "RMSE_std"],
        "泛化能力 (25%)": ["tail_rmse_q10_mean", "tail_rmse_q90_mean"],
    }
    group_weights = {
        "预测精度 (50%)": 0.50,
        "稳定性 (25%)": 0.25,
        "泛化能力 (25%)": 0.25,
    }

    metric_to_idx = {m: i for i, m in enumerate(metric_names)}
    n = len(metric_names)
    w_final = np.zeros(n)
    details = []

    for gname, gmetrics in groups.items():
        g_indices = [metric_to_idx[m] for m in gmetrics]
        g_X = X[:, g_indices]
        g_dirs = [directions[i] for i in g_indices]

        w_g, e_g = entropy_weights(g_X, g_dirs)

        for gidx, m, wg, eg in zip(g_indices, gmetrics, w_g, e_g):
            w_final[gidx] = group_weights[gname] * wg
            details.append({
                "指标": m,
                "方向": directions[gidx],
                "所属组": gname,
                "组权重": group_weights[gname],
                "组内熵值 e": round(float(eg), 4),
                "组内权重": round(float(wg), 4),
                "最终权重 w": round(float(group_weights[gname] * wg), 4),
            })

    # 未被纳入任何组的指标（如 MAE_mean）权重保持 0
    for i, m in enumerate(metric_names):
        if m not in [d["指标"] for d in details]:
            details.append({
                "指标": m,
                "方向": directions[i],
                "所属组": "排除（与组内指标高度共线）",
                "组权重": 0,
                "组内熵值 e": 0,
                "组内权重": 0,
                "最终权重 w": 0,
            })

    # 归一化
    w_final = w_final / (w_final.sum() + 1e-12)

    w_df = pd.DataFrame(details)
    w_df["最终权重 w"] = [
        round(float(w_final[metric_to_idx[d["指标"]]]), 4) if d["最终权重 w"] > 0 else 0
        for d in details
    ]
    w_df = w_df.sort_values("最终权重 w", ascending=False).reset_index(drop=True)
    return w_final, w_df


# ============================================================================
# TOPSIS 核心算法
# ============================================================================


def topsis(X: np.ndarray, weights: np.ndarray, directions: list[str]) -> np.ndarray:
    """TOPSIS 六步法，返回相对贴近度 C_i ∈ [0,1]。"""
    m, n = X.shape

    # Step 1: 向量归一化
    norm = np.sqrt(np.sum(X**2, axis=0))
    r = X / (norm + 1e-12)

    # Step 2: 加权标准化矩阵
    v = r * weights

    # Step 3: 正/负理想解
    a_plus = np.zeros(n)
    a_minus = np.zeros(n)
    for j in range(n):
        if directions[j] == "benefit":
            a_plus[j] = np.max(v[:, j])
            a_minus[j] = np.min(v[:, j])
        else:
            a_plus[j] = np.min(v[:, j])
            a_minus[j] = np.max(v[:, j])

    # Step 4: 欧氏距离
    d_plus = np.sqrt(np.sum((v - a_plus)**2, axis=1))
    d_minus = np.sqrt(np.sum((v - a_minus)**2, axis=1))

    # Step 5: 相对贴近度
    C = d_minus / (d_plus + d_minus + 1e-12)
    return C


# ============================================================================
# 综合运行
# ============================================================================


def run_topsis(results_path: Path = None) -> dict:
    """执行完整 TOPSIS 分析（层次熵权 ★ 主方案 + 熵权 + CRITIC 三方案）。

    Returns
    -------
    dict with keys:
        ranking_grouped, weights_grouped,
        ranking_entropy, weights_entropy,
        ranking_critic, weights_critic
    """
    if results_path is None:
        results_path = ROOT / "outputs" / "tables" / "cv_results.json"

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    X, models, metric_names = _build_matrix(results)
    directions = [CRITERIA[m] for m in metric_names]

    output = {}

    # --- 方案1: 熵权法 ---
    w_ent, e = entropy_weights(X, directions)
    C_ent = topsis(X, w_ent, directions)
    order_ent = np.argsort(-C_ent)
    ranks_ent = np.argsort(np.argsort(-C_ent)) + 1

    metric_df = pd.DataFrame(X, columns=metric_names).round(4)
    ranking_entropy = pd.DataFrame({
        "排名": ranks_ent,
        "模型": models,
        "贴近度 C_i": np.round(C_ent, 4),
    })
    ranking_entropy = pd.concat([ranking_entropy, metric_df], axis=1)
    ranking_entropy = ranking_entropy.iloc[order_ent].reset_index(drop=True)

    weights_entropy = pd.DataFrame({
        "指标": metric_names,
        "方向": directions,
        "熵值 e": np.round(e, 4),
        "权重 w": np.round(w_ent, 4),
    }).sort_values("权重 w", ascending=False).reset_index(drop=True)

    output["ranking_entropy"] = ranking_entropy
    output["weights_entropy"] = weights_entropy

    # --- 方案2: 层次熵权法 ★ 论文主方案 ---
    w_grouped, wg_df = grouped_entropy_weights(X, metric_names, directions)
    C_grouped = topsis(X, w_grouped, directions)
    order_grouped = np.argsort(-C_grouped)
    ranks_grouped = np.argsort(np.argsort(-C_grouped)) + 1

    ranking_grouped = pd.DataFrame({
        "排名": ranks_grouped,
        "模型": models,
        "贴近度 C_i": np.round(C_grouped, 4),
    })
    ranking_grouped = pd.concat([ranking_grouped, metric_df], axis=1)
    ranking_grouped = ranking_grouped.iloc[order_grouped].reset_index(drop=True)

    output["ranking_grouped"] = ranking_grouped
    output["weights_grouped"] = wg_df

    # --- 方案3: CRITIC（敏感性分析）---
    w_crit, S, A = critic_weights(X, directions)
    C_crit = topsis(X, w_crit, directions)
    order_crit = np.argsort(-C_crit)
    ranks_crit = np.argsort(np.argsort(-C_crit)) + 1

    ranking_critic = pd.DataFrame({
        "排名": ranks_crit,
        "模型": models,
        "贴近度 C_i": np.round(C_crit, 4),
    })
    ranking_critic = pd.concat([ranking_critic, metric_df], axis=1)
    ranking_critic = ranking_critic.iloc[order_crit].reset_index(drop=True)

    intercorr = np.abs(np.corrcoef(X.T))
    weights_critic = pd.DataFrame({
        "指标": metric_names,
        "方向": directions,
        "对比强度 σ": np.round(S, 4),
        "冲突度 A": np.round(A, 4),
        "信息量 C": np.round(S * A, 4),
        "权重 w": np.round(w_crit, 4),
    }).sort_values("权重 w", ascending=False).reset_index(drop=True)

    output["ranking_critic"] = ranking_critic
    output["weights_critic"] = weights_critic
    output["intercorr_matrix"] = intercorr

    return output


# ============================================================================
# 主入口
# ============================================================================


def main():
    import sys
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    output = run_topsis()
    tables_dir = ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ---- 层次熵权法 ★ 主方案 ----
    print("=" * 80)
    print("★ 方案1: TOPSIS 综合排名（层次熵权法 Grouped Entropy）— 论文主方案")
    print("=" * 80)
    print("组间: 精度50% / 稳定性25% / 泛化25%  |  组内: 熵权法")
    print("MAE_mean 因与 RMSE_mean 高度共线 (r≈0.997) 被排除，避免组内通胀")
    print("-" * 80)
    print(output["ranking_grouped"].to_string(index=False))

    print(f"\n层次熵权法权重分解:")
    print(output["weights_grouped"].to_string(index=False))

    # ---- 熵权法（参考）----
    print(f"\n{'=' * 80}")
    print("方案2: TOPSIS 综合排名（标准熵权法 Entropy）— 参考")
    print("=" * 80)
    print(output["ranking_entropy"].to_string(index=False))

    print(f"\n熵权法权重:")
    print(output["weights_entropy"].to_string(index=False))

    # ---- CRITIC（敏感性分析）----
    print(f"\n{'=' * 80}")
    print("方案3: TOPSIS 综合排名（CRITIC）— 敏感性分析")
    print("=" * 80)
    print("注意: n=9 时相关性估计不稳定，CRITIC 冲突度 A_j 对噪声敏感，仅供稳健性参考")
    print("-" * 80)
    print(output["ranking_critic"].to_string(index=False))

    print(f"\nCRITIC 权重:")
    print(output["weights_critic"].to_string(index=False))

    # 三方案排名对比
    print(f"\n{'=' * 80}")
    print("三方案排名对比")
    print("=" * 80)

    rank_comp = output["ranking_grouped"][["模型", "排名"]].rename(columns={"排名": "层次熵权"})
    rank_comp = rank_comp.merge(
        output["ranking_entropy"][["模型", "排名"]].rename(columns={"排名": "标准熵权"}),
        on="模型",
    )
    rank_comp = rank_comp.merge(
        output["ranking_critic"][["模型", "排名"]].rename(columns={"排名": "CRITIC"}),
        on="模型",
    )
    rank_comp["极差"] = rank_comp[["层次熵权", "标准熵权", "CRITIC"]].max(axis=1) - \
                       rank_comp[["层次熵权", "标准熵权", "CRITIC"]].min(axis=1)
    rank_comp = rank_comp.sort_values("层次熵权").reset_index(drop=True)
    print(rank_comp.to_string(index=False))

    # 保存
    output["ranking_grouped"].to_csv(
        tables_dir / "topsis_grouped.csv", index=False, encoding="utf-8-sig")
    output["weights_grouped"].to_csv(
        tables_dir / "grouped_weights.csv", index=False, encoding="utf-8-sig")
    output["ranking_entropy"].to_csv(
        tables_dir / "topsis_entropy.csv", index=False, encoding="utf-8-sig")
    output["weights_entropy"].to_csv(
        tables_dir / "entropy_weights.csv", index=False, encoding="utf-8-sig")
    output["ranking_critic"].to_csv(
        tables_dir / "topsis_critic.csv", index=False, encoding="utf-8-sig")
    output["weights_critic"].to_csv(
        tables_dir / "critic_weights.csv", index=False, encoding="utf-8-sig")

    print(f"\n已保存 6 个文件至 {tables_dir}/")
    print("  topsis_grouped.csv / grouped_weights.csv  ★ 主方案")
    print("  topsis_entropy.csv / entropy_weights.csv")
    print("  topsis_critic.csv  / critic_weights.csv")


if __name__ == "__main__":
    main()
