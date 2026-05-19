"""步骤1: 数据加载与清洗"""

import unicodedata

import pandas as pd

from src.config import RAW_COLUMNS, PROCESS_COLUMNS, NUMERIC_COLUMNS, RAW_EXCEL


def load_and_clean(excel_path: str | None = None) -> pd.DataFrame:
    """加载原始Excel，执行步骤1全部清洗逻辑，返回干净DataFrame。"""
    path = excel_path or str(RAW_EXCEL)

    # --- 1. 读取Excel，跳过前2行合并表头 ---
    df = pd.read_excel(path, header=None, skiprows=2)
    df.columns = RAW_COLUMNS

    # --- 2. 丢弃 References ---
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
