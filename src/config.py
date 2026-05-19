"""项目路径、常量与随机种子配置"""

from pathlib import Path

# --- 项目根目录 ---
ROOT = Path(__file__).resolve().parent.parent

# --- 数据路径 ---
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RAW_EXCEL = DATA_RAW / "0514biochar-MgOdata.xlsx"

# --- 输出路径 ---
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

# --- 随机种子 ---
SEED = 42

# --- 列名定义（顺序与原始Excel列对应）---
RAW_COLUMNS = [
    "carbon_precursors",
    "MgO_precursors",
    "Mg_loading_method",
    "Activation1",
    "Activation2",
    "Carbonization1",
    "Carbonization2",
    "Post_treatment",
    "SBET_m2_g",
    "Vtotal_cm3_g",
    "Vmicro_cm3_g",
    "MgO_crystallite_size_nm",
    "MgO_mass_ratio",
    "temperature_C",
    "pressure_bar",
    "CO2_uptake_mg_g",
    "References",
]

# --- 工艺列（NaN = 未执行该步骤）---
PROCESS_COLUMNS = [
    "Activation1",
    "Activation2",
    "Carbonization1",
    "Carbonization2",
    "Post_treatment",
]

# --- 数值列（需要 pd.to_numeric 转换）---
NUMERIC_COLUMNS = [
    "SBET_m2_g",
    "Vtotal_cm3_g",
    "Vmicro_cm3_g",
    "MgO_crystallite_size_nm",
    "MgO_mass_ratio",
    "temperature_C",
    "pressure_bar",
    "CO2_uptake_mg_g",
]

# --- 目标变量 ---
TARGET = "CO2_uptake_mg_g"
