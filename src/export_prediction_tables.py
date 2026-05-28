"""导出7个非线性模型的预测数据CSV（80/20 Train/Test 划分）

每个模型:
1. 随机划分 80% Train / 20% Test（Stratified by target bins）
2. 在 Train 上训练，对两部分分别预测
3. 导出 CSV: True_Value, Predicted_Value, Residual, Set
"""

import sys
import io
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone

from src.config import ROOT, TARGET, TABLES, SEED
from src.data_loader import load_and_clean
from src.models import get_all_models

NONLINEAR_MODELS = ["TabPFN", "GBDT", "RF", "XGBoost", "LightGBM", "GPR", "SVR"]
REPR_SEED = 91  # 代表性种子: TabPFN 80/20 Test R²=0.9688 ≈ 嵌套CV 0.9692
TEST_SIZE = 0.20


def export_all():
    print("加载数据...")
    df = load_and_clean()
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # 分层 bins (5分位)
    q_edges = np.linspace(0, 1, 6)[1:-1]
    thresholds = np.quantile(y.values, q_edges)
    bins = np.digitize(y.values, thresholds)

    # ── 80/20 随机划分 ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=TEST_SIZE,
        random_state=REPR_SEED,
        stratify=bins,
    )
    print(f"Train: {len(X_train)} 样本, Test: {len(X_test)} 样本 "
          f"(随机种子={REPR_SEED}, stratified)")

    # ── 获取模型定义 ───────────────────────────────────────────────────
    all_models = get_all_models()
    model_map = {}
    for name, pipeline, pipe_label, param_fn in all_models:
        model_map[name] = (pipeline, pipe_label, param_fn)

    export_dir = TABLES / "prediction_tables"
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"{'Model':<12} {'Train R²':>10} {'Train RMSE':>12} {'Test R²':>10} {'Test RMSE':>12}")
    print(f"{'='*70}")

    for name in NONLINEAR_MODELS:
        pipeline, pipe_label, param_fn = model_map[name]

        # 使用已保存的最优参数（若有），跳过 Optuna
        params_path = ROOT / "outputs" / "tables" / "cv_results.json"
        with open(params_path, "r", encoding="utf-8") as f:
            cv_results = json.load(f)

        best_params = {}
        if name in cv_results and "best_params_per_fold" in cv_results[name]:
            fold_params = cv_results[name]["best_params_per_fold"]
            if fold_params and len(fold_params) > 0 and fold_params[0]:
                best_params = fold_params[0]  # 使用第一折的最优参数

        pipe = clone(pipeline)
        if best_params:
            try:
                pipe.set_params(**best_params)
            except (ValueError, KeyError):
                pass  # 参数不兼容则使用默认值

        # ── 在训练集上拟合 ──────────────────────────────────────────────
        if name in ("XGBoost", "LightGBM"):
            from src.train import compute_kde_weights
            sw = compute_kde_weights(y_train)
            pipe.fit(X_train, y_train, model__sample_weight=sw)
        else:
            pipe.fit(X_train, y_train)

        # ── 预测 ────────────────────────────────────────────────────────
        tr_preds = pipe.predict(X_train)
        if tr_preds.ndim == 2 and tr_preds.shape[1] == 1:
            tr_preds = tr_preds.ravel()
        te_preds = pipe.predict(X_test)
        if te_preds.ndim == 2 and te_preds.shape[1] == 1:
            te_preds = te_preds.ravel()

        tr_resid = tr_preds - y_train.values
        te_resid = te_preds - y_test.values

        r2_train = r2_score(y_train, tr_preds)
        rmse_train = np.sqrt(mean_squared_error(y_train, tr_preds))
        r2_test = r2_score(y_test, te_preds)
        rmse_test = np.sqrt(mean_squared_error(y_test, te_preds))

        # ── 构建 DataFrame ──────────────────────────────────────────────
        df_train = pd.DataFrame({
            "True_Value": y_train.values,
            "Predicted_Value": tr_preds,
            "Residual": tr_resid,
            "Set": "Train",
        })
        df_test = pd.DataFrame({
            "True_Value": y_test.values,
            "Predicted_Value": te_preds,
            "Residual": te_resid,
            "Set": "Test",
        })
        df_out = pd.concat([df_train, df_test], ignore_index=True)

        # ── 保存 CSV ────────────────────────────────────────────────────
        csv_path = export_dir / f"Figure_4_{name}_Predictions.csv"
        df_out.to_csv(csv_path, index=False, float_format="%.6f")

        print(f"{name:<12} {r2_train:>10.4f} {rmse_train:>12.2f} {r2_test:>10.4f} {rmse_test:>12.2f}")
        print(f"  → {csv_path} ({len(df_out)} 行: {len(df_train)} Train + {len(df_test)} Test)")

    print(f"{'='*70}")
    print(f"\n全部导出完成。文件位于: {export_dir}")
    print(f"划分方式: 80/20 stratified split, random_state={REPR_SEED} (代表性种子)")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    export_all()
