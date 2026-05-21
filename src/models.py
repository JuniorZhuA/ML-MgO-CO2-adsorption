"""步骤4: ML模型 —— 9个模型，4条预处理管道，Optuna超参搜索空间"""

import sys
import io
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import SEED, TARGET

# ---------------------------------------------------------------------------
# TabPFN 可选导入与就绪检测
# ---------------------------------------------------------------------------

HAS_TABPFN = False
TABPFN_READY = False
TabPFNRegressor = None

try:
    from tabpfn import TabPFNRegressor as _TabPFN

    TabPFNRegressor = _TabPFN
    HAS_TABPFN = True
except ImportError:  # pragma: no cover
    pass


def _check_tabpfn_ready() -> bool:
    """检测 TabPFN 是否已完成许可接受（auth token 已缓存即可，模型权重会在首次 fit 时自动下载）。"""
    if not HAS_TABPFN:
        return False
    token_file = Path.home() / ".cache" / "tabpfn" / "auth_token"
    if token_file.exists():
        return True
    alt_token = Path.home() / ".tabpfn" / "token"
    return alt_token.exists()


TABPFN_READY = _check_tabpfn_ready()

# ---------------------------------------------------------------------------
# 特征集定义
# ---------------------------------------------------------------------------

CATEGORICAL_COLUMNS = [
    "carbon_precursors",
    "MgO_precursors",
    "Mg_loading_method",
    "act1_type",
    "act2_type",
    "carb1_type",
    "carb2_type",
    "post_treatment_type",
]


def _get_column_lists(X):
    """从DataFrame中检测分类列和数值列（排除目标列）。"""
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols and c != TARGET]
    return cat_cols, num_cols


# ---------------------------------------------------------------------------
# 四条预处理管道
# ---------------------------------------------------------------------------


def build_pipeline_a(model, cat_cols, num_cols):
    """管道A（线性模型）: OneHotEncoder + StandardScaler + median填补."""
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        [
            ("impute_median", SimpleImputer(strategy="median")),
            ("impute_zero", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_pipeline_b(model, cat_cols, num_cols):
    """管道B（树模型）: OrdinalEncoder + median填补，无缩放."""
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    num_pipe = Pipeline(
        [
            ("impute_median", SimpleImputer(strategy="median")),
            ("impute_zero", SimpleImputer(strategy="constant", fill_value=0.0)),
        ]
    )
    preprocessor = ColumnTransformer(
        [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_pipeline_c(model, cat_cols, num_cols):
    """管道C（SVR/GPR）: TargetEncoder + StandardScaler + log1p(y)变换."""
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("target", TargetEncoder(target_type="continuous", smooth="auto")),
        ]
    )
    num_pipe = Pipeline(
        [
            ("impute_median", SimpleImputer(strategy="median")),
            ("impute_zero", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)]
    )
    inner = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return TransformedTargetRegressor(
        regressor=inner, func=np.log1p, inverse_func=np.expm1
    )


def build_pipeline_d(cat_cols, num_cols):
    """管道D（TabPFN）: OrdinalEncoder，无缩放/填补/调优."""
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [("cat", cat_pipe, cat_cols), ("num", "passthrough", num_cols)]
    )
    model = TabPFNRegressor(random_state=SEED)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# Optuna 超参搜索空间
# ---------------------------------------------------------------------------


def suggest_params(trial, model_name: str) -> dict:
    """为给定模型返回一个参数字典，参数名带 `model__` 前缀以便 Pipeline 访问。

    Optuna 内部参数名也带 `model__` 前缀，确保 study.best_params 可直接传递给
    pipeline.set_params()。
    """

    if model_name == "Ridge":
        return {"model__alpha": trial.suggest_float("model__alpha", 1e-3, 1e4, log=True)}

    if model_name == "Lasso":
        return {"model__alpha": trial.suggest_float("model__alpha", 1e-4, 1e2, log=True)}

    if model_name == "RF":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 600),
            "model__max_depth": trial.suggest_int("model__max_depth", 5, 50),
            "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 20),
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 10),
            "model__max_features": trial.suggest_float("model__max_features", 0.3, 1.0),
        }

    if model_name == "XGBoost":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 600),
            "model__max_depth": trial.suggest_int("model__max_depth", 3, 12),
            "model__learning_rate": trial.suggest_float(
                "model__learning_rate", 0.01, 0.3, log=True
            ),
            "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0),
            "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.6, 1.0),
            "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-4, 10, log=True),
            "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-4, 10, log=True),
        }

    if model_name == "LightGBM":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 600),
            "model__num_leaves": trial.suggest_int("model__num_leaves", 15, 255),
            "model__learning_rate": trial.suggest_float(
                "model__learning_rate", 0.01, 0.3, log=True
            ),
            "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0),
            "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.6, 1.0),
            "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-4, 10, log=True),
            "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-4, 10, log=True),
            "model__min_child_samples": trial.suggest_int("model__min_child_samples", 5, 50),
        }

    if model_name == "GBDT":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 600),
            "model__max_depth": trial.suggest_int("model__max_depth", 3, 12),
            "model__learning_rate": trial.suggest_float(
                "model__learning_rate", 0.01, 0.3, log=True
            ),
            "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0),
            "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 20),
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 10),
            "model__max_features": trial.suggest_float("model__max_features", 0.3, 1.0),
        }

    if model_name == "SVR":
        return {
            "model__C": trial.suggest_float("model__C", 0.1, 100, log=True),
            "model__gamma": trial.suggest_float("model__gamma", 1e-4, 1, log=True),
            "model__epsilon": trial.suggest_float("model__epsilon", 0.01, 1, log=True),
        }

    if model_name == "GPR":
        return {
            "model__alpha": trial.suggest_float("model__alpha", 1e-8, 1e-1, log=True),
        }

    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# 模型汇总工厂
# ---------------------------------------------------------------------------


def get_all_models(X):
    """返回全部9个模型的 (名称, 管道, 管道标签, Optuna参数函数或None) 列表。"""
    cat_cols, num_cols = _get_column_lists(X)

    models = []

    # --- 管道A: 线性基线 ---
    models.append(
        (
            "Ridge",
            build_pipeline_a(Ridge(random_state=SEED), cat_cols, num_cols),
            "A",
            lambda t: suggest_params(t, "Ridge"),
        )
    )
    models.append(
        (
            "Lasso",
            build_pipeline_a(Lasso(random_state=SEED, max_iter=5000), cat_cols, num_cols),
            "A",
            lambda t: suggest_params(t, "Lasso"),
        )
    )

    # --- 管道B: 树模型 ---
    models.append(
        (
            "RF",
            build_pipeline_b(
                RandomForestRegressor(random_state=SEED), cat_cols, num_cols
            ),
            "B",
            lambda t: suggest_params(t, "RF"),
        )
    )
    models.append(
        (
            "XGBoost",
            build_pipeline_b(
                XGBRegressor(random_state=SEED, verbosity=0), cat_cols, num_cols
            ),
            "B",
            lambda t: suggest_params(t, "XGBoost"),
        )
    )
    models.append(
        (
            "LightGBM",
            build_pipeline_b(
                LGBMRegressor(random_state=SEED, verbose=-1), cat_cols, num_cols
            ),
            "B",
            lambda t: suggest_params(t, "LightGBM"),
        )
    )
    models.append(
        (
            "GBDT",
            build_pipeline_b(
                GradientBoostingRegressor(random_state=SEED), cat_cols, num_cols
            ),
            "B",
            lambda t: suggest_params(t, "GBDT"),
        )
    )

    # --- 管道C: SVR / GPR ---
    models.append(
        (
            "SVR",
            build_pipeline_c(SVR(kernel="rbf"), cat_cols, num_cols),
            "C",
            lambda t: suggest_params(t, "SVR"),
        )
    )
    models.append(
        (
            "GPR",
            build_pipeline_c(
                GaussianProcessRegressor(
                    kernel=ConstantKernel() * RBF() + WhiteKernel(),
                    normalize_y=True,
                    n_restarts_optimizer=10,
                    random_state=SEED,
                ),
                cat_cols,
                num_cols,
            ),
            "C",
            lambda t: suggest_params(t, "GPR"),
        )
    )

    # --- 管道D: TabPFN（零超参）---
    if TABPFN_READY:
        models.append(
            ("TabPFN", build_pipeline_d(cat_cols, num_cols), "D", None)
        )

    return models


# ---------------------------------------------------------------------------
# 独立验证
# ---------------------------------------------------------------------------


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from src.data_loader import load_and_clean
    from src.preprocessing import MissingValueImputer
    from src.feature_engineering import FeatureEngineer

    # ---- 数据准备 ----
    print("加载数据 ...")
    df = load_and_clean()
    df = MissingValueImputer().fit_transform(df)
    df = FeatureEngineer().fit_transform(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    cat_cols, num_cols = _get_column_lists(X)
    print(f"\n特征维度: {X.shape}")
    print(f"  分类列 ({len(cat_cols)}): {cat_cols}")
    print(f"  数值列 ({len(num_cols)}): {num_cols}")
    print(f"  目标变量: {TARGET}  (均值={y.mean():.1f}, 标准差={y.std():.1f})")

    # ---- 获取所有模型 ----
    all_models = get_all_models(X)
    print(f"\n{'='*60}")
    print(f"共 {len(all_models)} 个模型")
    if HAS_TABPFN and not TABPFN_READY:
        print("注意: TabPFN 已安装但未完成许可接受，暂未加入模型列表")
        print("  请运行以下命令完成一次性认证（需要浏览器）:")
        print("  python -c \"from tabpfn import TabPFNRegressor; import numpy as np;")
        print("  TabPFNRegressor().fit(np.array([[1.0]]), np.array([1.0]))\"")
    print(f"{'='*60}")

    for name, pipe, label, param_fn in all_models:
        try:
            pipe_clone = clone(pipe)
            pipe_clone.fit(X, y)
            preds = pipe_clone.predict(X)
            r2_train = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
            print(f"  [{label}] {name:10s}  → 训练R²={r2_train:.4f}  (参数函数={'有' if param_fn else '无'})")
        except Exception as e:
            err_msg = str(e)[:120]
            print(f"  [{label}] {name:10s}  → 失败: {err_msg}")

    print("\nmodels.py 验证完成。")


if __name__ == "__main__":
    main()
