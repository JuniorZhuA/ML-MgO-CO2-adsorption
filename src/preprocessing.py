"""步骤2: 缺失值填补"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """对数值列执行缺失值填补，遵循计划书的策略。

    - SBET, Vtotal → KNNImputer (k=5)
    - Vmicro → IterativeImputer (BayesianRidge, max_iter=20)
      + 后处理约束: Vmicro = min(Vmicro_imputed, Vtotal)
    - MgO_crystallite_size → 直接删除（双轨策略A方案）
    """

    def __init__(self, drop_crystallite_size: bool = True):
        self.drop_crystallite_size = drop_crystallite_size

        # 定义各列填补器
        self._knn_imputer = KNNImputer(n_neighbors=5)
        self._iter_imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
        )

        # 记录哪些列需要哪种填补
        self._knn_cols = ["SBET_m2_g", "Vtotal_cm3_g"]
        self._iter_col = "Vmicro_cm3_g"
        self._cryst_col = "MgO_crystallite_size_nm"

    def fit(self, X: pd.DataFrame, y=None):
        # 1. KNN 填补 SBET, Vtotal
        self._knn_imputer.fit(X[self._knn_cols])

        # 2. IterativeImputer 填补 Vmicro（使用完整数值列作为特征）
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        # 排除 crystallite size（缺失率太高，干扰填补）
        if self._cryst_col in num_cols:
            num_cols.remove(self._cryst_col)
        self._iter_feature_cols = num_cols
        self._iter_imputer.fit(X[num_cols])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. KNN 填补 SBET, Vtotal
        X[self._knn_cols] = self._knn_imputer.transform(X[self._knn_cols])

        # 2. IterativeImputer 填补 Vmicro
        iter_features = X[self._iter_feature_cols]
        iter_filled = self._iter_imputer.transform(iter_features)
        vmicro_idx = self._iter_feature_cols.index(self._iter_col)
        X[self._iter_col] = iter_filled[:, vmicro_idx]

        # 3. 物理约束: 0 ≤ Vmicro ≤ Vtotal
        neg = X[self._iter_col] < 0
        n_neg = neg.sum()
        if n_neg > 0:
            print(f"  [约束] 修正 {n_neg} 条 Vmicro < 0 的记录 → 0")
            X.loc[neg, self._iter_col] = 0.0

        violated = X[self._iter_col] > X["Vtotal_cm3_g"]
        n_violated = violated.sum()
        if n_violated > 0:
            print(f"  [约束] 修正 {n_violated} 条 Vmicro > Vtotal 的记录")
            X.loc[violated, self._iter_col] = X.loc[violated, "Vtotal_cm3_g"]

        # 4. 删除 MgO_crystallite_size（双轨策略A）
        if self.drop_crystallite_size and self._cryst_col in X.columns:
            X = X.drop(columns=[self._cryst_col])

        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def robustness_check(df: pd.DataFrame) -> dict:
    """Vmicro完整样本 vs 全量填补样本的基本统计对比。

    返回完整样本的子集 df_vmicro_complete 和 key stats，供后续模型对比使用。
    """
    complete = df[df["Vmicro_cm3_g"].notna()].copy()
    print(f"Vmicro 完整样本: {len(complete)} 条")
    print(f"  CO2 uptake 均值: {complete['CO2_uptake_mg_g'].mean():.1f} mg/g")
    print(f"  CO2 uptake 标准差: {complete['CO2_uptake_mg_g'].std():.1f} mg/g")
    return complete
