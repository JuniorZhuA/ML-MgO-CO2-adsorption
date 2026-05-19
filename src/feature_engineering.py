"""步骤3: 特征工程 —— 正则解析 + 复合特征"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """从工艺文本列中提取温度/时间数值特征和分类特征，构建领域复合特征。

    输入列:
    - 工艺文本: Activation1, Activation2, Carbonization1, Carbonization2, Post_treatment
    - 数值列: SBET_m2_g, Vtotal_cm3_g, Vmicro_cm3_g, MgO_mass_ratio,
              temperature_C, pressure_bar

    输出列（新增）:
    - 解析数值: act1_temp_C, act1_duration_h, act2_temp_C, act2_duration_h,
                carb1_temp_C, carb1_duration_h, carb2_temp_C, carb2_duration_h
    - 工艺分类: act1_type, act2_type, carb1_type, carb2_type, post_treatment_type
    - 复合特征: Vmeso_cm3_g, microporosity, MgO_surface_density, T_lnP, inv_T_K
    """

    # ---- 正则模式 ----
    _TEMP_RE = re.compile(r"(\d+\.?\d*)\s*°C")
    _DURATION_RE = re.compile(r"(\d+\.?\d*)\s*(h|hour|min)")

    # ---- 工艺分类映射 ----
    @staticmethod
    def _classify_act1(text: str) -> str:
        if text == "none":
            return "none"
        t = text.lower()
        if "steam" in t:
            return "steam"
        if "koh" in t:
            return "KOH"
        if "hydrothermal" in t:
            return "hydrothermal"
        return "other"

    @staticmethod
    def _classify_act2(text: str) -> str:
        if text == "none":
            return "none"
        if "h3po4" in text.lower():
            return "H3PO4"
        return "other"

    @staticmethod
    def _classify_carb1(text: str) -> str:
        if text == "none":
            return "none"
        t = text.lower()
        if "microwave" in t:
            return "microwave"
        if "fast pyrolysis" in t:
            return "fast_pyrolysis"
        # "carbonized at ..." / plain "XX °C for XX h" → conventional
        return "conventional"

    @staticmethod
    def _classify_carb2(text: str) -> str:
        if text == "none":
            return "none"
        if "15 min" in text:
            return "500C_15min"
        if "2 h" in text:
            return "600C_2h"
        return "other"

    @staticmethod
    def _classify_post_treatment(text: str) -> str:
        if text == "none":
            return "none"
        t = text.lower()
        if "without rinsing" in t:
            return "without_rinsing"
        if "with rinsing" in t:
            return "with_rinsing"
        if "sonication" in t:
            return "sonication"
        return "other"

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ---- 3a. 正则提取温度和时间 ----
        for col, prefix in [
            ("Activation1", "act1"),
            ("Activation2", "act2"),
            ("Carbonization1", "carb1"),
            ("Carbonization2", "carb2"),
        ]:
            X = self._parse_temp_duration(X, col, prefix)

        # ---- 3a. 工艺分类 ----
        X["act1_type"] = X["Activation1"].apply(self._classify_act1)
        X["act2_type"] = X["Activation2"].apply(self._classify_act2)
        X["carb1_type"] = X["Carbonization1"].apply(self._classify_carb1)
        X["carb2_type"] = X["Carbonization2"].apply(self._classify_carb2)
        X["post_treatment_type"] = X["Post_treatment"].apply(
            self._classify_post_treatment
        )

        # ---- 3b. 领域复合特征 ----
        X["Vmeso_cm3_g"] = X["Vtotal_cm3_g"] - X["Vmicro_cm3_g"]
        X["microporosity"] = X["Vmicro_cm3_g"] / X["Vtotal_cm3_g"].replace(0, np.nan)
        X["MgO_surface_density"] = X["MgO_mass_ratio"] / (
            X["SBET_m2_g"] / 1000.0
        ).replace(0, np.nan)
        X["T_lnP"] = X["temperature_C"] * np.log(X["pressure_bar"] + 0.01)
        X["inv_T_K"] = 1.0 / (X["temperature_C"] + 273.15)

        # ---- 丢弃原始工艺文本列（信息已提取完毕）----
        X = X.drop(
            columns=[
                "Activation1",
                "Activation2",
                "Carbonization1",
                "Carbonization2",
                "Post_treatment",
            ]
        )

        return X

    # ------------------------------------------------------------------
    def _parse_temp_duration(
        self, df: pd.DataFrame, col: str, prefix: str
    ) -> pd.DataFrame:
        """从单列中提取温度(°C)和时长(h)，返回修改后的DataFrame。"""
        temps, durations = [], []
        for text in df[col]:
            text_norm = " ".join(str(text).split())  # 合并连续空格
            if text_norm == "none":
                temps.append(np.nan)
                durations.append(np.nan)
            else:
                t_match = self._TEMP_RE.search(text_norm)
                temps.append(float(t_match.group(1)) if t_match else np.nan)

                d_match = self._DURATION_RE.search(text_norm)
                if d_match:
                    val = float(d_match.group(1))
                    unit = d_match.group(2)
                    durations.append(val if unit in ("h", "hour") else val / 60.0)
                else:
                    durations.append(np.nan)

        df[f"{prefix}_temp_C"] = temps
        df[f"{prefix}_duration_h"] = durations
        return df
