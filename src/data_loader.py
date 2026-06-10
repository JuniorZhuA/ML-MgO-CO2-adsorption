"""步骤1: 数据加载与清洗"""

import sys
import unicodedata

import pandas as pd

from src.config import RAW_COLUMNS, PROCESS_COLUMNS, NUMERIC_COLUMNS, RAW_EXCEL

# ── 必需列清单（加载后立即校验）────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "carbon_precursors", "MgO_precursors", "Mg_loading_method",
    "Activation1", "Activation2", "Carbonization1", "Carbonization2",
    "Post_treatment", "SBET_m2_g", "Vtotal_cm3_g", "Vmicro_cm3_g",
    "MgO_mass_ratio", "temperature_C", "pressure_bar",
    "CO2_uptake_mg_g", "References",
]


def load_and_clean(excel_path: str | None = None) -> pd.DataFrame:
    """加载原始Excel，执行步骤1全部清洗逻辑，返回干净DataFrame。"""
    path = excel_path or str(RAW_EXCEL)

    # --- 0. 文件读取保护 ──────────────────────────────────────────────────
    try:
        df = pd.read_excel(path, header=None, skiprows=2)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"数据文件不存在: {path}\n"
            f"  请确认 Excel 文件已放置于 {RAW_EXCEL}"
        ) from None
    except Exception as e:
        raise RuntimeError(
            f"读取 Excel 文件失败: {path}\n"
            f"  原始错误: {type(e).__name__}: {e}"
        ) from e

    # --- 0.1 列名数量校验 ────────────────────────────────────────────────
    n_actual = df.shape[1]
    n_expected = len(RAW_COLUMNS)
    if n_actual != n_expected:
        raise ValueError(
            f"Excel 列数 ({n_actual}) 与配置列数 ({n_expected}) 不匹配。\n"
            f"  期望列: {RAW_COLUMNS}\n"
            f"  实际列: {list(df.columns)}\n"
            f"  请检查 Excel 文件结构或 src/config.py 中的 RAW_COLUMNS 定义。"
        )
    df.columns = RAW_COLUMNS

    # --- 0.2 关键列名存在性检查 ──────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Excel 缺少必需列: {missing_cols}\n"
            f"  当前列: {list(df.columns)}\n"
            f"  请检查 RAW_COLUMNS 映射是否正确。"
        )

    # --- 1. 丢弃 References ---
    df = df.drop(columns=["References"])

    # --- 3. 数值列转换 ---
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 4. 统一 MgO_precursors 的 Unicode 变体 ---
    df["MgO_precursors"] = (
        df["MgO_precursors"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)               # 多余空格合并
        .str.replace("(NO3)2 ", "(NO3)2", regex=False)       # Mg (NO3)2 → Mg(NO3)2
        .apply(lambda s: unicodedata.normalize("NFKC", s))
        .str.replace("·", "⋅", regex=False)        # 统一点字符 · → ⋅
    )

    # --- 5. 工艺列 NaN → "none" ---
    for col in PROCESS_COLUMNS:
        df[col] = df[col].fillna("none").astype(str)

    # --- 6. Mg_loading_method: 合并 impregnation 与 wetness impregnation ---
    df["Mg_loading_method"] = (
        df["Mg_loading_method"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"wetness impregnation": "impregnation"})
    )

    return df


def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    df = load_and_clean()
    print("=" * 60)
    print("数据加载与清洗完成")
    print("=" * 60)
    print()
    df.info()
    print()
    print("前5行预览:")
    print(df.head().to_string())
    print()
    print("工艺列唯一值检查:")
    for col in PROCESS_COLUMNS:
        vals = df[col].unique()
        print(f"  {col}: {len(vals)} 种取值 — {vals[:10]}")
    print()
    print(f"MgO_precursors 唯一值 ({df['MgO_precursors'].nunique()} 种):")
    print(df["MgO_precursors"].value_counts().to_string())
    print()
    print(f"Mg_loading_method 唯一值: {df['Mg_loading_method'].unique()}")
    print()
    print("数值列缺失统计:")
    print(df[NUMERIC_COLUMNS].isnull().sum().to_string())


if __name__ == "__main__":
    main()
