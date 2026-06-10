#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冒烟测试（Smoke Test）—— 全流程微缩验证

参数覆写:
  - Optuna n_trials = 2
  - CV outer/inner = 2/2 (KFold)
  - SHAP n_repeats = 2

输出目录: outputs_smoketest/（绝不覆盖官方 outputs/）
"""

import sys
import io
import json
import os
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ══════════════════════════════════════════════════════════════════════════════
# 0. 路径覆写：重定向全部输出到 outputs_smoketest/
# ══════════════════════════════════════════════════════════════════════════════

import src.config as cfg

# 保存原始路径
_ORIG_OUTPUTS = cfg.OUTPUTS
_ORIG_FIGURES = cfg.FIGURES
_ORIG_TABLES = cfg.TABLES
_ORIG_MODELS = cfg.ROOT / "outputs" / "models"

# 覆写
SMOKE_OUTPUTS = cfg.ROOT / "outputs_smoketest"
cfg.OUTPUTS = SMOKE_OUTPUTS
cfg.FIGURES = SMOKE_OUTPUTS / "figures"
cfg.TABLES = SMOKE_OUTPUTS / "tables"

SMOKE_MODELS_DIR = SMOKE_OUTPUTS / "models"
SMOKE_FIGURES = SMOKE_OUTPUTS / "figures"
SMOKE_TABLES = SMOKE_OUTPUTS / "tables"
SMOKE_PRED_DIR = SMOKE_TABLES / "prediction_tables"

for d in [SMOKE_MODELS_DIR, SMOKE_FIGURES, SMOKE_TABLES, SMOKE_PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. 导入项目模块（config 已覆写）
# ══════════════════════════════════════════════════════════════════════════════

from src.config import SEED, TARGET, ROOT
from src.data_loader import load_and_clean
from src.models import get_all_models, TABPFN_READY, _get_column_lists
from src.evaluate import compute_metrics, compute_tail_rmse, aggregate_fold_results
from src.train import (
    create_stratification_bins, compute_kde_weights, _make_fit_params,
    _add_param_prefix, run_nested_cv, train_final_model,
)
from src.topsis import run_topsis

# ══════════════════════════════════════════════════════════════════════════════
# 2. 冒烟测试参数
# ══════════════════════════════════════════════════════════════════════════════

N_TRIALS = 2
N_OUTER = 2
N_INNER = 2
N_REPEATS_SHAP = 2
REPR_SEED = 91  # 80/20 代表性种子

# ══════════════════════════════════════════════════════════════════════════════
# 3. 主流程
# ══════════════════════════════════════════════════════════════════════════════

def smoke_test():
    total_start = time.time()
    errors = []

    print("=" * 70)
    print(" 冒烟测试 — 全流程微缩验证")
    print(f" 参数: n_trials={N_TRIALS}, CV={N_OUTER}x{N_INNER}, SHAP n_repeats={N_REPEATS_SHAP}")
    print(f" 输出: {SMOKE_OUTPUTS}")
    print("=" * 70)

    # ── 3a. 加载数据 ─────────────────────────────────────────────────────
    print("\n[1/8] 加载数据...")
    df = load_and_clean()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"  样本: {len(df)}, 特征: {X.shape[1]}, 目标: {TARGET} (μ={y.mean():.1f}, σ={y.std():.1f})")

    all_models = get_all_models()
    print(f"  模型数: {len(all_models)}")
    for name, _, label, has_params in all_models:
        print(f"    [{label}] {name} {'(含超参)' if has_params else '(零超参)'}")

    # ── 3b. 嵌套 CV（覆写参数）───────────────────────────────────────────
    print(f"\n[2/8] 嵌套 CV (KFold {N_OUTER}x{N_INNER}, n_trials={N_TRIALS})...")

    cv_outer = KFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    outer_splits = list(cv_outer.split(X))

    results: dict[str, dict] = {}
    cv_predictions: dict[str, dict] = {}

    for model_name, pipeline, pipe_label, param_fn in all_models:
        print(f"\n  --- [{pipe_label}] {model_name} ---")
        t0 = time.time()

        try:
            # 用简化CV直接跑
            fold_metrics = []
            all_y_true, all_y_pred = [], []
            all_best_params = []

            for fold_i, (train_idx, test_idx) in enumerate(outer_splits):
                print(f"    折 {fold_i+1}/{N_OUTER} (train={len(train_idx)}, test={len(test_idx)})")

                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

                # 内层 Optuna（TabPFN 跳过）
                if model_name == "TabPFN":
                    best_params_dict = {}
                else:
                    inner_bins = create_stratification_bins(y_train, n_bins=5)
                    # 内层也用 KFold
                    inner_cv = KFold(n_splits=N_INNER, shuffle=True, random_state=SEED + fold_i)
                    inner_splits_list = list(inner_cv.split(X_train))

                    import optuna
                    optuna.logging.set_verbosity(optuna.logging.WARNING)

                    def _objective(trial):
                        params = _add_param_prefix(param_fn(trial), pipe_label)
                        p = clone(pipeline)
                        p.set_params(**params)
                        scores = []
                        for in_tr, in_val in inner_splits_list:
                            X_it, y_it = X_train.iloc[in_tr], y_train.iloc[in_tr]
                            X_iv, y_iv = X_train.iloc[in_val], y_train.iloc[in_val]
                            fit_params = _make_fit_params(model_name, y_it)
                            p.fit(X_it, y_it, **fit_params)
                            pred = p.predict(X_iv)
                            scores.append(np.sqrt(mean_squared_error(y_iv, pred)))
                        return float(np.mean(scores))

                    study = optuna.create_study(
                        direction="minimize",
                        sampler=optuna.samplers.TPESampler(seed=SEED + fold_i),
                    )
                    study.optimize(_objective, n_trials=N_TRIALS, show_progress_bar=False)
                    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    best_params_dict = _add_param_prefix(study.best_params, pipe_label) if completed else {}

                all_best_params.append(best_params_dict)

                # 在外层训练集上训练
                final_pipe = clone(pipeline)
                if best_params_dict:
                    final_pipe.set_params(**best_params_dict)
                fit_params = _make_fit_params(model_name, y_train)
                final_pipe.fit(X_train, y_train, **fit_params)
                y_pred = final_pipe.predict(X_test)

                all_y_true.extend(y_test.values.tolist())
                all_y_pred.extend(y_pred.tolist())

                m = compute_metrics(y_test.values, y_pred)
                m.update(compute_tail_rmse(y_test.values, y_pred))
                fold_metrics.append(m)
                print(f"      R²={m['R²']:.4f}, RMSE={m['RMSE']:.1f}")

            agg = aggregate_fold_results(fold_metrics)
            r2_mean = agg.get("R²_mean", np.nan)
            rmse_mean = agg.get("RMSE_mean", np.nan)
            elapsed = time.time() - t0
            print(f"    汇总: R²={r2_mean:.4f}, RMSE={rmse_mean:.1f} ({elapsed:.0f}s)")

            results[model_name] = {
                "model": model_name,
                "pipeline": pipe_label,
                "aggregate": agg,
                "fold_metrics": fold_metrics,
                "best_params_per_fold": all_best_params,
            }
            cv_predictions[model_name] = {
                "y_true": all_y_true,
                "y_pred": all_y_pred,
            }

        except Exception as e:
            msg = f"{model_name}: {str(e)[:200]}"
            print(f"    ✗ FAILED: {msg}")
            errors.append(msg)
            results[model_name] = None

    # ── 保存 cv_results.json ─────────────────────────────────────────────
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    results_json = {}
    for name, r in results.items():
        if r:
            results_json[name] = {
                "model": r["model"], "pipeline": r["pipeline"],
                "aggregate": r["aggregate"], "fold_metrics": r["fold_metrics"],
                "best_params_per_fold": r.get("best_params_per_fold", []),
            }
    cv_results_path = SMOKE_TABLES / "cv_results.json"
    with open(cv_results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, cls=NpEncoder, indent=2, ensure_ascii=False)
    print(f"\n  ✓ cv_results.json ({len(results_json)} models)")

    # ── 保存 cv_predictions.json ─────────────────────────────────────────
    cv_pred_path = SMOKE_TABLES / "cv_predictions.json"
    with open(cv_pred_path, "w", encoding="utf-8") as f:
        json.dump(cv_predictions, f, cls=NpEncoder, indent=2, ensure_ascii=False)
    print(f"  ✓ cv_predictions.json ({len(cv_predictions)} models)")

    # ── 3c. 训练最终模型（全量数据）───────────────────────────────────────
    print(f"\n[3/8] 训练最终模型（全量数据, n_trials={N_TRIALS}）...")

    final_models = {}
    for model_name, pipeline, pipe_label, param_fn in all_models:
        try:
            fm = train_final_model(
                model_name=model_name, pipeline=pipeline, pipeline_label=pipe_label,
                param_fn=param_fn, X=X, y=y, n_trials=N_TRIALS, random_state=SEED,
            )
            final_models[model_name] = fm
            model_path = SMOKE_MODELS_DIR / f"{model_name}_final.pkl"
            joblib.dump(fm["pipeline"], model_path)
            print(f"  ✓ {model_name}: R²={fm['train_r2']:.4f}, RMSE={fm['train_rmse']:.1f} → {model_path.name}")
        except Exception as e:
            msg = f"Final {model_name}: {str(e)[:200]}"
            print(f"  ✗ {msg}")
            errors.append(msg)

    # ── 3d. 80/20 分割导出 ───────────────────────────────────────────────
    print(f"\n[4/8] 80/20 分割预测导出 (SEED={REPR_SEED})...")

    from sklearn.model_selection import train_test_split

    q_edges = np.linspace(0, 1, 6)[1:-1]
    thresholds = np.quantile(y.values, q_edges)
    bins = np.digitize(y.values, thresholds)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.20,
        random_state=REPR_SEED, stratify=bins,
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    for model_name in ["TabPFN", "GBDT", "RF", "XGBoost", "LightGBM", "GPR", "SVR"]:
        try:
            pipe = joblib.load(SMOKE_MODELS_DIR / f"{model_name}_final.pkl")
            tr_preds = pipe.predict(X_train)
            if tr_preds.ndim == 2 and tr_preds.shape[1] == 1:
                tr_preds = tr_preds.ravel()
            te_preds = pipe.predict(X_test)
            if te_preds.ndim == 2 and te_preds.shape[1] == 1:
                te_preds = te_preds.ravel()

            df_train = pd.DataFrame({
                "True_Value": y_train.values, "Predicted_Value": tr_preds,
                "Residual": tr_preds - y_train.values, "Set": "Train",
            })
            df_test = pd.DataFrame({
                "True_Value": y_test.values, "Predicted_Value": te_preds,
                "Residual": te_preds - y_test.values, "Set": "Test",
            })
            df_out = pd.concat([df_train, df_test], ignore_index=True)
            csv_path = SMOKE_PRED_DIR / f"Figure_4_{model_name}_Predictions.csv"
            df_out.to_csv(csv_path, index=False, float_format="%.6f")
            r2_te = r2_score(y_test, te_preds)
            print(f"  ✓ {model_name}: Test R²={r2_te:.4f} → {csv_path.name}")
        except Exception as e:
            msg = f"Export {model_name}: {str(e)[:200]}"
            print(f"  ✗ {msg}")
            errors.append(msg)

    # ── 3e. TOPSIS ───────────────────────────────────────────────────────
    print(f"\n[5/8] TOPSIS 排名（层次熵权 ★ 主方案）...")

    try:
        topsis_output = run_topsis(results_path=cv_results_path)

        topsis_output["ranking_grouped"].to_csv(
            SMOKE_TABLES / "topsis_grouped.csv", index=False, encoding="utf-8-sig")
        topsis_output["weights_grouped"].to_csv(
            SMOKE_TABLES / "grouped_weights.csv", index=False, encoding="utf-8-sig")
        topsis_output["ranking_entropy"].to_csv(
            SMOKE_TABLES / "topsis_entropy.csv", index=False, encoding="utf-8-sig")
        topsis_output["weights_entropy"].to_csv(
            SMOKE_TABLES / "entropy_weights.csv", index=False, encoding="utf-8-sig")
        topsis_output["ranking_critic"].to_csv(
            SMOKE_TABLES / "topsis_critic.csv", index=False, encoding="utf-8-sig")
        topsis_output["weights_critic"].to_csv(
            SMOKE_TABLES / "critic_weights.csv", index=False, encoding="utf-8-sig")

        top1 = topsis_output["ranking_grouped"]["模型"].iloc[0]
        top1_ci = topsis_output["ranking_grouped"]["贴近度 C_i"].iloc[0]
        print(f"  ✓ Top 1: {top1} (C_i={top1_ci:.4f})")
    except Exception as e:
        msg = f"TOPSIS: {str(e)[:200]}"
        print(f"  ✗ {msg}")
        errors.append(msg)

    # ── 3f. SHAP 分析（覆写 n_repeats=2）────────────────────────────────
    print(f"\n[6/8] SHAP 分析 (n_repeats={N_REPEATS_SHAP})...")

    try:
        import shap
        from sklearn.impute import SimpleImputer
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        from scipy.stats import spearmanr

        # 加载 GBDT pipeline
        gbdt_path = SMOKE_MODELS_DIR / "GBDT_final.pkl"
        gbdt_pipe = joblib.load(gbdt_path)
        imputer = gbdt_pipe.named_steps["impute"]
        engineer = gbdt_pipe.named_steps["features"]
        preprocessor = gbdt_pipe.named_steps["preprocessor"]
        gbdt_model = gbdt_pipe.named_steps["model"]

        X_imputed = imputer.transform(X)
        X_fe = engineer.transform(X_imputed)
        cat_cols, num_cols = _get_column_lists(X_fe)
        feature_names = cat_cols + num_cols

        # VIF
        X_num_fe = X_fe[num_cols].copy()
        X_num_imp = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(X_num_fe),
            columns=X_num_fe.columns, index=X_num_fe.index)

        from src.shap_analysis import diagnose_and_remove_collinear, compute_vif
        X_num_clean, dropped_features = diagnose_and_remove_collinear(X_num_imp)
        vif_table = compute_vif(X_num_clean)
        vif_table.to_csv(SMOKE_TABLES / "vif_table.csv", index=False, encoding="utf-8-sig")

        # Spearman clustering
        from src.shap_analysis import compute_spearman_clusters
        cluster_table, Z, spearman_corr = compute_spearman_clusters(X_num_clean, threshold=0.3)
        cluster_table.to_csv(SMOKE_TABLES / "feature_clusters.csv", index=False, encoding="utf-8-sig")
        spearman_corr.to_csv(SMOKE_TABLES / "spearman_correlation.csv", encoding="utf-8-sig")

        # Save dropped features
        with open(SMOKE_TABLES / "dropped_features.json", "w", encoding="utf-8") as dj:
            json.dump({"dropped_features": dropped_features, "n_dropped": len(dropped_features)}, dj,
                      ensure_ascii=False, indent=2)

        # TabPFN permutation importance (n_repeats=2, fast)
        print("  计算 TabPFN 排列重要性...")
        tabpfn_path = SMOKE_MODELS_DIR / "TabPFN_final.pkl"
        tabpfn_pipe = joblib.load(tabpfn_path)
        tabpfn_imputer = tabpfn_pipe.named_steps["impute"]
        tabpfn_engineer = tabpfn_pipe.named_steps["features"]
        tabpfn_preprocessor = tabpfn_pipe.named_steps["preprocessor"]
        tabpfn_model = tabpfn_pipe.named_steps["model"]

        from sklearn.pipeline import Pipeline as SkPipeline
        X_tabpfn_fe = tabpfn_engineer.transform(tabpfn_imputer.transform(X))
        X_tabpfn_fe.columns = [str(c) for c in X_tabpfn_fe.columns]
        slim_pipe = SkPipeline([
            ("preprocessor", tabpfn_preprocessor),
            ("model", tabpfn_model),
        ])

        from src.shap_analysis import compute_tabpfn_permutation_importance
        tabpfn_imp = compute_tabpfn_permutation_importance(
            slim_pipe, X_tabpfn_fe, y, n_repeats=N_REPEATS_SHAP, random_state=SEED)
        tabpfn_imp.to_csv(SMOKE_TABLES / "permutation_importance_tabpfn.csv", index=False, encoding="utf-8-sig")

        # GBDT SHAP
        print("  计算 GBDT SHAP 值...")
        X_transformed = preprocessor.transform(X_fe)
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

        explainer = shap.TreeExplainer(gbdt_model)
        shap_vals = explainer.shap_values(X_transformed)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_df = pd.DataFrame(shap_vals, columns=feature_names)
        shap_df.to_csv(SMOKE_TABLES / "shap_values_gbdt.csv", index=False, encoding="utf-8-sig")

        importance = np.abs(shap_vals).mean(axis=0)
        imp_df = pd.DataFrame({"feature": feature_names, "shap_importance_mean": importance}
                             ).sort_values("shap_importance_mean", ascending=False).reset_index(drop=True)
        imp_df.to_csv(SMOKE_TABLES / "shap_importance_gbdt.csv", index=False, encoding="utf-8-sig")

        X_transformed.to_csv(SMOKE_TABLES / "X_transformed_gbdt.csv", index=False, encoding="utf-8-sig")

        with open(SMOKE_TABLES / "feature_names.json", "w", encoding="utf-8") as fj:
            json.dump({"feature_names": feature_names, "cat_cols": cat_cols, "num_cols": num_cols},
                      fj, ensure_ascii=False, indent=2)

        # Consistency
        feature_names_clean = [f for f in feature_names if f not in dropped_features]
        tabpfn_imp_clean = tabpfn_imp[tabpfn_imp["feature"].isin(feature_names_clean)]
        gbdt_imp_clean = imp_df[imp_df["feature"].isin(feature_names_clean)]
        merged = tabpfn_imp_clean.merge(gbdt_imp_clean, on="feature", how="inner")
        rho, pval = spearmanr(merged["importance_mean"], merged["shap_importance_mean"])
        print(f"  ✓ Spearman ρ = {rho:.4f} (p={pval:.4f}, n={len(merged)} features)")
        merged.to_csv(SMOKE_TABLES / "consistency_comparison.csv", index=False, encoding="utf-8-sig")

        # Store for plotting
        _shap_dropped = dropped_features

    except Exception as e:
        msg = f"SHAP: {str(e)[:300]}"
        print(f"  ✗ {msg}")
        errors.append(msg)
        _shap_dropped = []

    # ── 3g. 生成全部图表 ─────────────────────────────────────────────────
    print(f"\n[7/8] 生成全部图表...")

    # ── 路径修正: plotting.py 硬编码了 ROOT/"outputs"/"models" 和 "tables"，需要临时指向冒烟测试目录 ──
    _REAL_MODELS_DIR = cfg.ROOT / "outputs" / "models"
    _REAL_MODELS_BACKUP = cfg.ROOT / "outputs" / "models_backup_smoketest"
    _REAL_TABLES_DIR = cfg.ROOT / "outputs" / "tables"
    _REAL_TABLES_BACKUP = cfg.ROOT / "outputs" / "tables_backup_smoketest"

    if _REAL_MODELS_DIR.exists():
        _REAL_MODELS_DIR.rename(_REAL_MODELS_BACKUP)
    if _REAL_TABLES_DIR.exists():
        _REAL_TABLES_DIR.rename(_REAL_TABLES_BACKUP)

    import shutil as _shutil
    _shutil.copytree(str(SMOKE_MODELS_DIR), str(_REAL_MODELS_DIR))
    _shutil.copytree(str(SMOKE_TABLES), str(_REAL_TABLES_DIR))
    print(f"  [路径] 临时挂载冒烟测试 models + tables → outputs/")

    from src import plotting as plt_mod

    figure_funcs = [
        ("Figure_S1", plt_mod.figure_S1_precursor_distribution),
        ("Figure_1", plt_mod.figure_1_clustermap),
        ("Figure_2", plt_mod.figure_2_boxplot),
        ("Figure_3", plt_mod.figure_3_model_comparison),
        ("Figure_4_TabPFN", plt_mod.figure_4_tabpfn_marginal),
        ("Figure_5", plt_mod.figure_5_shap_beeswarm),
        ("Figure_6", plt_mod.figure_6_shap_dependence),
        ("Figure_7", plt_mod.figure_7_residuals),
        ("Figure_8", plt_mod.figure_8_pdp_ice),
        ("shap_bar", plt_mod.figure_shap_bar),
    ]

    for fig_name, fig_func in figure_funcs:
        try:
            print(f"  {fig_name}...", end=" ", flush=True)
            t0 = time.time()
            fig_func()
            elapsed = time.time() - t0
            print(f"✓ ({elapsed:.0f}s)")
        except Exception as e:
            msg = f"{fig_name}: {str(e)[:200]}"
            print(f"✗ {msg}")
            errors.append(msg)

    # ── Figure 4 (7模型 80/20) ──────────────────────────────────────────
    print("\n  Figure 4 (7-model 80/20 marginal)...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt2
        import seaborn as sns
        from scipy.stats import linregress

        plt2.rcParams['font.family'] = 'sans-serif'
        plt2.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt2.rcParams['axes.linewidth'] = 1.2

        for model_name in ["TabPFN", "GBDT", "RF", "XGBoost", "LightGBM", "GPR", "SVR"]:
            csv_path = SMOKE_PRED_DIR / f"Figure_4_{model_name}_Predictions.csv"
            if not csv_path.exists():
                print(f"    {model_name}: CSV 不存在, 跳过")
                continue
            df_pred = pd.read_csv(csv_path)
            train_df = df_pred[df_pred['Set'] == 'Train']
            test_df = df_pred[df_pred['Set'] == 'Test']

            r2_train = r2_score(train_df['True_Value'], train_df['Predicted_Value'])
            r2_test = r2_score(test_df['True_Value'], test_df['Predicted_Value'])
            rmse_test = np.sqrt(mean_squared_error(test_df['True_Value'], test_df['Predicted_Value']))

            fig = plt2.figure(figsize=(7, 8), dpi=300)
            gs = fig.add_gridspec(5, 5, hspace=0.0, wspace=0.0)

            ax_main = fig.add_subplot(gs[1:4, 0:4])
            ax_top = fig.add_subplot(gs[0, 0:4], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1:4, 4], sharey=ax_main)
            ax_res = fig.add_subplot(gs[4, 0:4], sharex=ax_main)

            # Main scatter (hollow circles)
            ax_main.scatter(train_df['True_Value'], train_df['Predicted_Value'],
                            facecolors='none', edgecolors='#D32F2F', linewidths=1.0,
                            alpha=0.55, marker='o', s=50, label='Train', zorder=3)
            ax_main.scatter(test_df['True_Value'], test_df['Predicted_Value'],
                            facecolors='none', edgecolors='#1976D2', linewidths=1.2,
                            alpha=1.0, marker='o', s=55, label='Test', zorder=5)

            min_val = min(df_pred['True_Value'].min(), df_pred['Predicted_Value'].min())
            max_val = max(df_pred['True_Value'].max(), df_pred['Predicted_Value'].max())
            ax_main.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

            x_test_v = test_df['True_Value'].values
            y_test_v = test_df['Predicted_Value'].values
            x_train_v = train_df['True_Value'].values
            y_train_v = train_df['Predicted_Value'].values

            if len(x_test_v) > 1:
                sl_te, ic_te, _, _, _ = linregress(x_test_v, y_test_v)
                x_fit = np.linspace(min_val, max_val, 200)
                ax_main.plot(x_fit, sl_te * x_fit + ic_te, color='#1976D2', lw=1.5)
            if len(x_train_v) > 1:
                sl_tr, ic_tr, _, _, _ = linregress(x_train_v, y_train_v)
                x_fit = np.linspace(min_val, max_val, 200)
                ax_main.plot(x_fit, sl_tr * x_fit + ic_tr, color='#D32F2F', lw=1.5)

            r2_text = f"{model_name}\nTrain R² = {r2_train:.4f}\nTest R² = {r2_test:.4f}\nTest RMSE = {rmse_test:.2f}"
            ax_main.text(0.95, 0.05, r2_text, transform=ax_main.transAxes,
                         ha='right', va='bottom', fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                   edgecolor='#BDBDBD', linewidth=0.8, alpha=0.9))
            ax_main.set_ylabel('Predicted CO$_2$ Uptake (mg/g)', fontsize=14, fontweight='bold')
            ax_main.legend(loc='upper left', fontsize=12, frameon=True)
            ax_main.tick_params(labelbottom=False)

            # Top histogram
            sns.histplot(data=df_pred, x='True_Value', hue='Set',
                         palette={'Train': '#E57373', 'Test': '#64B5F6'},
                         ax=ax_top, kde=True, element='step', common_norm=False, legend=False)
            ax_top.axis('off')

            # Right histogram
            sns.histplot(data=df_pred, y='Predicted_Value', hue='Set',
                         palette={'Train': '#E57373', 'Test': '#64B5F6'},
                         ax=ax_right, kde=True, element='step', common_norm=False, legend=False)
            ax_right.axis('off')

            # Residuals
            ax_res.scatter(x_train_v, train_df['Residual'].values,
                           facecolors='none', edgecolors='#D32F2F', linewidths=0.8,
                           alpha=0.55, marker='o', s=30, zorder=3)
            ax_res.scatter(x_test_v, test_df['Residual'].values,
                           facecolors='none', edgecolors='#1976D2', linewidths=1.0,
                           alpha=1.0, marker='o', s=35, zorder=5)
            ax_res.axhline(0, color='gray', linestyle='--', lw=1.5)
            ax_res.set_xlabel('Experimental CO$_2$ Uptake (mg/g)', fontsize=14, fontweight='bold')
            ax_res.set_ylabel('Residuals (mg/g)', fontsize=12, fontweight='bold')

            out_path = SMOKE_FIGURES / f"Figure_4_{model_name}_Marginal.png"
            fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt2.close(fig)
            print(f"    ✓ {model_name}")

    except Exception as e:
        msg = f"Figure 4 (7-model): {str(e)[:200]}"
        print(f"  ✗ {msg}")
        errors.append(msg)

    # ── 3h. 生成报告 ─────────────────────────────────────────────────────
    print(f"\n[8/8] 生成报告...")

    try:
        import src.generate_report as gen_report
        gen_report.OUTPUTS_DIR = SMOKE_OUTPUTS  # patch output path
        gen_report.main()
        print("  ✓ 综合进展报告")
    except Exception as e:
        msg = f"generate_report: {str(e)[:200]}"
        print(f"  ✗ {msg}")
        errors.append(msg)

    try:
        import src.generate_performance_report as gen_perf
        # This module references hardcoded relative paths — try to patch
        gen_perf.ROOT = cfg.ROOT  # fallback
        gen_perf.main()
        print("  ✓ 性能报告（中英文）")
    except Exception as e:
        msg = f"generate_performance_report: {str(e)[:200]}"
        print(f"  ✗ {msg}")
        errors.append(msg)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. 路径清理：恢复原始 outputs/models/ 和 outputs/tables/
    # ══════════════════════════════════════════════════════════════════════════
    if _REAL_MODELS_DIR.exists():
        _shutil.rmtree(str(_REAL_MODELS_DIR))
    if _REAL_TABLES_DIR.exists():
        _shutil.rmtree(str(_REAL_TABLES_DIR))
    if _REAL_MODELS_BACKUP.exists():
        _REAL_MODELS_BACKUP.rename(_REAL_MODELS_DIR)
    if _REAL_TABLES_BACKUP.exists():
        _REAL_TABLES_BACKUP.rename(_REAL_TABLES_DIR)
    if not (_REAL_MODELS_BACKUP.exists() or _REAL_TABLES_BACKUP.exists()):
        print(f"\n  [路径] 已恢复原始 models + tables 目录")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. 输出验证
    # ══════════════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print(f" 冒烟测试完成  (总耗时 {total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")

    # 统计生成的文件
    all_files = list(SMOKE_OUTPUTS.rglob("*"))
    figures_list = list(SMOKE_FIGURES.rglob("*.png")) if SMOKE_FIGURES.exists() else []
    tables_list = list(SMOKE_TABLES.rglob("*")) if SMOKE_TABLES.exists() else []
    models_list = list(SMOKE_MODELS_DIR.rglob("*.pkl")) if SMOKE_MODELS_DIR.exists() else []

    print(f"\n  Figures: {len(figures_list)} 张 PNG 图")
    for f in sorted(figures_list):
        print(f"    {f.name}")

    print(f"\n  Tables: {len(tables_list)} 个文件")
    for f in sorted(tables_list):
        print(f"    {f.relative_to(SMOKE_OUTPUTS)}")

    print(f"\n  Models: {len(models_list)} 个 .pkl")
    for f in sorted(models_list):
        print(f"    {f.name}")

    print(f"\n  总文件数: {len(all_files)}")

    # 期望检查
    expected_figs_min = 17  # S1 + F1-F3 + F4(8 variants) + F5-F8 + bar = 17
    if len(figures_list) >= expected_figs_min:
        print(f"\n  ✓ 图表数量符合预期 (≥{expected_figs_min})")
    else:
        print(f"\n  ⚠ 图表数量不足 (期望 ≥{expected_figs_min}, 实际 {len(figures_list)})")

    if errors:
        print(f"\n  ⚠ {len(errors)} 个错误:")
        for e in errors:
            print(f"    - {e}")
        print("\n  冒烟测试: 部分通过 (有错误)")
        return 1
    else:
        print(f"\n  ✓ 冒烟测试: 全部通过")
        return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    exit_code = smoke_test()
    sys.exit(exit_code)
