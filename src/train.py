"""步骤5: 嵌套交叉验证 + Optuna超参调优 + 最终模型训练

- 外层5折 StratifiedKFold，内层3折 Optuna TPE
- XGBoost/LightGBM: KDE样本权重（尾部增强）
- TabPFN: 零超参，直接评估
- 嵌套CV完成后，在全量数据上训练最终模型并保存
"""

import sys
import io
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KernelDensity
import optuna
import joblib

from src.config import SEED, TARGET, ROOT
from src.models import get_all_models, TABPFN_READY
from src.evaluate import compute_metrics, compute_tail_rmse, aggregate_fold_results

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# 分层
# ============================================================================


def create_stratification_bins(y: pd.Series, n_bins: int = 5) -> pd.Series:
    """按分位数将目标变量分为 n_bins 个箱，返回整数分层标签（0..n_bins-1）。"""
    y_vals = y.values if isinstance(y, pd.Series) else np.asarray(y)
    q_edges = np.linspace(0, 1, n_bins + 1)[1:-1]
    thresholds = np.quantile(y_vals, q_edges)
    bins = np.digitize(y_vals, thresholds)
    return pd.Series(bins, index=y.index if isinstance(y, pd.Series) else None)


# ============================================================================
# KDE 样本权重（仅 XGBoost / LightGBM）
# ============================================================================


def compute_kde_weights(y: pd.Series) -> np.ndarray:
    """KDE密度倒数权重，归一化到均值为1。尾部稀有样本获得更高权重。"""
    y_vals = (
        y.values.reshape(-1, 1)
        if isinstance(y, pd.Series)
        else np.asarray(y).reshape(-1, 1)
    )
    bw = 1.06 * np.std(y_vals) * len(y_vals) ** (-0.2)
    bw = max(bw, 1e-4)
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(y_vals)
    log_dens = kde.score_samples(y_vals)
    density = np.exp(np.clip(log_dens, -50, 50))
    weights = 1.0 / np.maximum(density, 1e-10)
    weights = weights / weights.mean()
    return weights


def _make_fit_params(model_name: str, y_train: pd.Series) -> dict:
    """为 XGBoost / LightGBM 生成 model__sample_weight 字典。"""
    if model_name in ("XGBoost", "LightGBM"):
        sw = compute_kde_weights(y_train)
        return {"model__sample_weight": sw}
    return {}


# ============================================================================
# 参数前缀映射（管道C: TransformedTargetRegressor → regressor__model__xxx）
# ============================================================================


def _add_param_prefix(params: dict, pipeline_label: str) -> dict:
    """将 model__xxx 参数映射为 pipeline.set_params 可用的实际路径。"""
    if pipeline_label == "C":
        return {f"regressor__{k}": v for k, v in params.items()}
    return params


# ============================================================================
# 嵌套CV核心
# ============================================================================


def run_nested_cv(
    model_name: str,
    pipeline,
    pipeline_label: str,
    param_fn,
    X: pd.DataFrame,
    y: pd.Series,
    stratify_bins: pd.Series,
    n_outer: int = 5,
    n_inner: int = 3,
    n_trials: int = 100,
    random_state: int = 42,
) -> dict:
    """对单个模型执行嵌套交叉验证（StratifiedKFold 外层5折 / 内层3折）。

    Returns
    -------
    dict with keys: model, pipeline, aggregate, fold_metrics, y_true, y_pred
    """

    print(f"\n{'=' * 60}")
    print(f"[管道{pipeline_label}] {model_name} — 嵌套CV (外{n_outer}/内{n_inner}, n_trials={n_trials})")
    print(f"{'=' * 60}")

    # ---- 外层划分：StratifiedKFold ----
    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    outer_splits = list(skf.split(X, stratify_bins))
    train_sizes = [len(tr) for tr, _ in outer_splits]
    test_sizes = [len(te) for _, te in outer_splits]
    print(f"  StratifiedKFold — "
          f"训练=[{', '.join(map(str, train_sizes))}], "
          f"测试=[{', '.join(map(str, test_sizes))}]")

    all_y_true: list[float] = []
    all_y_pred: list[float] = []
    fold_metrics: list[dict] = []
    all_best_params: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_splits):
        print(f"\n-- 外层折 {fold_i + 1}/{n_outer} "
              f"(训练{len(train_idx)} 测试{len(test_idx)}) --")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # ---- 内层 Optuna（TabPFN 跳过）----
        if model_name == "TabPFN":
            print("  跳过 Optuna 超参调优（零超参模型）")
            best_params = {}
        else:
            inner_bins = create_stratification_bins(y_train, n_bins=5)

            def _objective(trial: optuna.Trial) -> float:
                params = _add_param_prefix(param_fn(trial), pipeline_label)
                pipe = clone(pipeline)
                pipe.set_params(**params)

                inner_skf = StratifiedKFold(
                    n_splits=n_inner, shuffle=True,
                    random_state=random_state + fold_i,
                )
                inner_splits = list(inner_skf.split(X_train, inner_bins))

                scores = []
                for in_tr, in_val in inner_splits:
                    X_it, y_it = X_train.iloc[in_tr], y_train.iloc[in_tr]
                    X_iv, y_iv = X_train.iloc[in_val], y_train.iloc[in_val]

                    fit_params = _make_fit_params(model_name, y_it)
                    pipe.fit(X_it, y_it, **fit_params)
                    pred = pipe.predict(X_iv)
                    scores.append(np.sqrt(mean_squared_error(y_iv, pred)))

                return float(np.mean(scores))

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=random_state + fold_i),
            )
            study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

            completed = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if completed:
                best_params = _add_param_prefix(study.best_params, pipeline_label)
                print(f"  内层最优RMSE={study.best_value:.1f}  "
                      f"（{len(completed)}/{n_trials} 试验完成）")
            else:
                print("  ⚠ 所有 Optuna 试验均失败，使用默认参数")
                best_params = {}

        all_best_params.append(best_params)

        # ---- 用最优参数在外层训练集上重新训练 ----
        final_pipe = clone(pipeline)
        if best_params:
            final_pipe.set_params(**best_params)

        fit_params = _make_fit_params(model_name, y_train)
        final_pipe.fit(X_train, y_train, **fit_params)

        # ---- 在外层测试集上预测 ----
        y_pred = final_pipe.predict(X_test)

        all_y_true.extend(y_test.values.tolist())
        all_y_pred.extend(y_pred.tolist())

        # ---- 指标 ----
        m = compute_metrics(y_test.values, y_pred)
        m.update(compute_tail_rmse(y_test.values, y_pred))
        fold_metrics.append(m)

        print(f"  R²={m['R²']:.4f}  RMSE={m['RMSE']:.1f}  MAE={m['MAE']:.1f}  "
              f"MAPE={m['MAPE']:.1f}%  q10_RMSE={m['tail_rmse_q10']:.1f}  "
              f"q90_RMSE={m['tail_rmse_q90']:.1f}")

    # ---- 汇总 ----
    agg = aggregate_fold_results(fold_metrics)

    r2_per_fold = [f"{m['R²']:.4f}" for m in fold_metrics]
    r2_str = "  ".join(f"F{i+1}={v}" for i, v in enumerate(r2_per_fold))
    print(f"\n  📊 逐折 R²: {r2_str}")

    r2_mean = agg.get("R²_mean", np.nan)
    r2_std = agg.get("R²_std", np.nan)
    rmse_mean = agg.get("RMSE_mean", np.nan)
    rmse_std = agg.get("RMSE_std", np.nan)
    print(
        f"\n  → 汇总: R²={r2_mean:.4f}±{r2_std:.4f}, "
        f"RMSE={rmse_mean:.1f}±{rmse_std:.1f}"
    )

    return {
        "model": model_name,
        "pipeline": pipeline_label,
        "aggregate": agg,
        "fold_metrics": fold_metrics,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "best_params_per_fold": all_best_params,
    }


# ============================================================================
# 最终模型训练（全量数据）
# ============================================================================


def train_final_model(
    model_name: str,
    pipeline,
    pipeline_label: str,
    param_fn,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    random_state: int = 42,
) -> dict:
    """在全量训练集上运行 Optuna（5折CV）+ 用最优参数训练最终模型。

    Returns
    -------
    dict with keys: pipeline, best_params, train_r2, train_rmse
    """

    print(f"\n{'=' * 60}")
    print(f"[管道{pipeline_label}] {model_name} — 最终模型训练（全量数据）")
    print(f"{'=' * 60}")

    if model_name == "TabPFN":
        print("  零超参模型，直接在全量数据上拟合...")
        final_pipe = clone(pipeline)
        final_pipe.fit(X, y)
        preds = final_pipe.predict(X)
        r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"  训练集 R²={r2:.4f}  RMSE={rmse:.1f}")
        return {"pipeline": final_pipe, "best_params": {}, "train_r2": r2, "train_rmse": rmse}

    stratify_bins = create_stratification_bins(y, n_bins=5)

    def _objective(trial: optuna.Trial) -> float:
        params = _add_param_prefix(param_fn(trial), pipeline_label)
        pipe = clone(pipeline)
        pipe.set_params(**params)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, val_idx in skf.split(X, stratify_bins):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            fit_params = _make_fit_params(model_name, y_tr)
            pipe.fit(X_tr, y_tr, **fit_params)
            pred = pipe.predict(X_val)
            scores.append(np.sqrt(mean_squared_error(y_val, pred)))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    best_params = _add_param_prefix(study.best_params, pipeline_label)
    print(f"  最优CV RMSE={study.best_value:.1f}")
    print(f"  最优参数: {best_params}")

    final_pipe = clone(pipeline)
    if best_params:
        final_pipe.set_params(**best_params)

    fit_params = _make_fit_params(model_name, y)
    final_pipe.fit(X, y, **fit_params)

    preds = final_pipe.predict(X)
    r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"  训练集 R²={r2:.4f}  RMSE={rmse:.1f}")

    return {"pipeline": final_pipe, "best_params": best_params, "train_r2": r2, "train_rmse": rmse}


# ============================================================================
# 主入口
# ============================================================================


def main():
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    from src.data_loader import load_and_clean
    from src.preprocessing import MissingValueImputer
    from src.feature_engineering import FeatureEngineer

    # ---- 数据准备 ----
    print("=" * 60)
    print("步骤5: 嵌套交叉验证 + Optuna 超参调优（正式运行）")
    print("=" * 60)

    print("\n加载并处理数据...")
    df = load_and_clean()
    df = MissingValueImputer().fit_transform(df)
    df = FeatureEngineer().fit_transform(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    n_samples, n_features = X.shape
    print(f"特征矩阵: {n_samples} × {n_features}")
    print(f"目标变量: {TARGET} — "
          f"均值={y.mean():.1f}, 标准差={y.std():.1f}, "
          f"范围=[{y.min():.1f}, {y.max():.1f}]")

    stratify_bins = create_stratification_bins(y, n_bins=5)
    print(f"目标分箱分布: {dict(sorted(stratify_bins.value_counts().to_dict().items()))}")

    # 所有模型
    all_models = get_all_models(X)
    print(f"\n模型数: {len(all_models)}")
    if not TABPFN_READY:
        print("(TabPFN 未认证，已自动排除)")

    # ---- 阶段1: 嵌套CV ----
    N_TRIALS = 100
    print(f"\nOptuna n_trials = {N_TRIALS} (正式模式)")
    print("CV 策略: StratifiedKFold (外层5折 / 内层3折)")

    results: dict[str, dict] = {}

    for model_name, pipeline, pipe_label, param_fn in all_models:
        result = run_nested_cv(
            model_name=model_name,
            pipeline=pipeline,
            pipeline_label=pipe_label,
            param_fn=param_fn,
            X=X,
            y=y,
            stratify_bins=stratify_bins,
            n_outer=5,
            n_inner=3,
            n_trials=N_TRIALS,
            random_state=SEED,
        )
        results[model_name] = result

    # ---- 阶段2: 最终模型训练（全量数据）----
    print(f"\n{'=' * 60}")
    print("阶段2: 在全量训练集上训练最终模型")
    print(f"{'=' * 60}")

    models_dir = ROOT / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    final_models: dict = {}

    for model_name, pipeline, pipe_label, param_fn in all_models:
        fm = train_final_model(
            model_name=model_name,
            pipeline=pipeline,
            pipeline_label=pipe_label,
            param_fn=param_fn,
            X=X,
            y=y,
            n_trials=N_TRIALS,
            random_state=SEED,
        )
        final_models[model_name] = fm

        model_path = models_dir / f"{model_name}_final.pkl"
        joblib.dump(fm["pipeline"], model_path)
        print(f"  已保存: {model_path}")

    # ========================================================================
    # 最终排名表
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("9 模型最终排名表（嵌套CV 5×3 StratifiedKFold, n_trials=100）")
    print(f"{'=' * 80}")

    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["aggregate"].get("R²_mean", -999),
        reverse=True,
    )

    print(f"{'排名':<6} {'模型':<12} {'管道':<4} {'R²_mean':>10} {'R²_std':>8} "
          f"{'RMSE_mean':>10} {'RMSE_std':>8} {'MAE':>8} {'MAPE%':>8} "
          f"{'q10_RMSE':>10} {'q90_RMSE':>10}")
    print("-" * 100)

    for rank, (name, r) in enumerate(ranked, 1):
        agg = r["aggregate"]
        print(
            f"{rank:<6} {name:<12} {r['pipeline']:<4} "
            f"{agg.get('R²_mean', 0):10.4f} {agg.get('R²_std', 0):8.4f} "
            f"{agg.get('RMSE_mean', 0):10.1f} {agg.get('RMSE_std', 0):8.1f} "
            f"{agg.get('MAE_mean', 0):8.1f} {agg.get('MAPE_mean', 0):8.1f} "
            f"{agg.get('tail_rmse_q10_mean', 0):10.1f} "
            f"{agg.get('tail_rmse_q90_mean', 0):10.1f}"
        )

    # ---- 最终模型训练集表现 ----
    print(f"\n{'=' * 80}")
    print("最终模型全量训练集表现")
    print(f"{'=' * 80}")
    print(f"{'模型':<12} {'训练R²':>10} {'训练RMSE':>10} {'最优参数':>30}")
    print("-" * 70)
    for name, fm in final_models.items():
        params_str = str(fm.get("best_params", {}))[:60] if fm.get("best_params") else "(零超参)"
        print(f"{name:<12} {fm['train_r2']:10.4f} {fm['train_rmse']:10.1f}  {params_str}")

    # ---- 关键发现 ----
    print(f"\n{'=' * 60}")
    print("关键发现")
    print(f"{'=' * 60}")

    best_name = ranked[0][0]
    best_r2 = ranked[0][1]["aggregate"]["R²_mean"]
    r2_ridge = results.get("Ridge", {}).get("aggregate", {}).get("R²_mean", 0)
    gap = best_r2 - r2_ridge
    print(f"  最优模型: {best_name} (嵌套CV R²={best_r2:.4f})")
    print(f"  非线性-线性差距: {gap:.4f} {'✓ > 0.05 — 非线性建模必要性得到支撑' if gap >= 0.05 else '⚠ < 0.05 — 特征本身即具有强预测力'}")

    if "TabPFN" in results:
        r2_tabpfn = results["TabPFN"]["aggregate"].get("R²_mean", 0)
        r2_rf = results.get("RF", {}).get("aggregate", {}).get("R²_mean", 0)
        r2_xgb = results.get("XGBoost", {}).get("aggregate", {}).get("R²_mean", 0)
        print(f"  TabPFN(零超参) 嵌套CV R²={r2_tabpfn:.4f}")
        if r2_rf:
            flag = "✓ 超过" if r2_tabpfn >= r2_rf else "✗ 低于"
            print(f"    vs RF: {flag} (RF R²={r2_rf:.4f})")
        if r2_xgb:
            flag = "✓ 超过" if r2_tabpfn >= r2_xgb else "✗ 低于"
            print(f"    vs XGBoost: {flag} (XGBoost R²={r2_xgb:.4f})")

    print(f"\n最终模型已保存至: {models_dir}")
    print("\n步骤5正式运行完成。")


if __name__ == "__main__":
    main()
