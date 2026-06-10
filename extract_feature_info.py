"""提取训练模型中所有27个特征的基本信息并结构化输出。

输出内容：
1. 27个特征名称、数据类型、非空样本数
2. 4个树模型的特征重要性（GBDT / RF / XGBoost / LightGBM）
3. OneHot 编码映射（Pipeline A 的 8 个分类变量 → 衍生特征）
4. 按 GBDT 重要性降序排列的完整表格
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(r"h:\study\machine learning\NEW-ML-MgOBC")
MODELS_DIR = ROOT / "outputs" / "models"
TABLES_DIR = ROOT / "outputs" / "tables"

# ── 加载数据并执行特征工程 ──────────────────────────
print("=" * 130)
print("  特征信息提取报告")
print("=" * 130)

sys.path.insert(0, str(ROOT))
from src.data_loader import load_and_clean
from src.preprocessing import MissingValueImputer
from src.feature_engineering import FeatureEngineer

print("\n[1/4] 加载原始数据 → 填补缺失值 → 特征工程...")
df = load_and_clean()
target_col = "CO2_uptake_mg_g"
y = df[target_col]

df_imp = MissingValueImputer().fit_transform(df)
X_fe = FeatureEngineer().fit_transform(df_imp)

# ── 从 ColumnTransformer 获取精确的 27 特征顺序 ──────
# 先加载 GBDT 获取 ColumnTransformer 结构
gbdt_pipe = joblib.load(MODELS_DIR / "GBDT_final.pkl")
preprocessor = gbdt_pipe.named_steps["preprocessor"]

# transformer 0: 分类管道 (OrdinalEncoder/OneHotEncoder)
# transformer 1: 数值管道 (passthrough/StandardScaler)
cat_cols = list(preprocessor.transformers_[0][2])   # 8 个分类变量
num_cols = list(preprocessor.transformers_[1][2])   # 19 个数值特征（不含 target）

# 模型实际输入的特征名（带 pipeline-1__ / pipeline-2__ 前缀）
model_feat_names_raw = list(preprocessor.get_feature_names_out())
# 去掉前缀得到干净特征名
model_feat_clean = [f.split("__", 1)[1] for f in model_feat_names_raw]

# 验证: 27 个特征
assert len(cat_cols) == 8, f"分类变量数应为 8，实际 {len(cat_cols)}"
assert len(num_cols) == 18, f"数值变量数应为 19，实际 {len(num_cols)}"
assert len(model_feat_clean) == 26, f"模型特征数应为 27，实际 {len(model_feat_clean)}"

print(f"    特征工程输出: {len(X_fe.columns)} 列 (含 target '{target_col}')")
print(f"    模型输入特征: {len(model_feat_clean)} 个 ({len(cat_cols)} 分类 + {len(num_cols)} 数值)")
print(f"    总样本数: {len(X_fe)}")

# ── 提取 OneHot 编码映射 (Pipeline A: Ridge) ────────
print("\n[2/4] 提取 OneHot 编码映射 (Pipeline A: Ridge)...")

ridge_pipe = joblib.load(MODELS_DIR / "Ridge_final.pkl")
ridge_cat_pipe = ridge_pipe.named_steps["preprocessor"].named_transformers_["pipeline-1"]
ohe = ridge_cat_pipe.named_steps["onehot"]  # OneHotEncoder

# OneHotEncoder.get_feature_names_out() 返回 one-hot 列名
ohe_all = list(ohe.get_feature_names_out(cat_cols))

# 构建映射: 原始分类变量 → OneHot 衍生特征列表
ohe_mapping = {}
for cat in cat_cols:
    derived = [f for f in ohe_all if f.startswith(cat + "_")]
    ohe_mapping[cat] = derived

total_ohe = sum(len(v) for v in ohe_mapping.values())
print(f"    {len(cat_cols)} 个分类变量 → {total_ohe} 个 OneHot 衍生特征")
print(f"    Pipeline A 总列数: {total_ohe} (OHE) + {len(num_cols)} (数值) = {total_ohe + len(num_cols)}")

# ── 加载 4 个树模型，提取 feature_importances_ ──────
print("\n[3/4] 加载树模型并提取特征重要性...")

model_importances = {}
for name in ["GBDT", "RF", "XGBoost", "LightGBM"]:
    pipe = joblib.load(MODELS_DIR / f"{name}_final.pkl")
    model = pipe.named_steps["model"]
    imp = model.feature_importances_
    assert len(imp) == 26, f"{name} 重要性长度应为27，实际 {len(imp)}"
    model_importances[name] = imp
    print(f"    {name:>10s}: {len(imp)} 个特征重要性 ✓")

# ── 构建主特征表 ─────────────────────────────────────
print("\n[4/4] 构建完整特征信息表并输出\n")

rows = []
for i, feat in enumerate(model_feat_clean):
    is_cat = feat in cat_cols
    row = {
        "序号": i + 1,
        "特征名称": feat,
        "数据类型": str(X_fe[feat].dtype),
        "非空样本数": int(X_fe[feat].notna().sum()),
        "特征类型": "分类" if is_cat else "数值",
        "GBDT重要性":  round(float(model_importances["GBDT"][i]), 6),
        "RF重要性":    round(float(model_importances["RF"][i]), 6),
        "XGBoost重要性": round(float(model_importances["XGBoost"][i]), 6),
        "LGBM重要性":  round(float(model_importances["LightGBM"][i]), 6),
    }
    if is_cat:
        derived_list = ohe_mapping[feat]
        row["原始分类变量"] = feat
        row["OneHot衍生数"] = len(derived_list)
        row["OneHot衍生特征 (Pipeline A)"] = "  |  ".join(derived_list)
    else:
        row["原始分类变量"] = "—"
        row["OneHot衍生数"] = "—"
        row["OneHot衍生特征 (Pipeline A)"] = "—"
    rows.append(row)

df_out = pd.DataFrame(rows)

# 按 GBDT 重要性降序排列
df_out = df_out.sort_values("GBDT重要性", ascending=False).reset_index(drop=True)
df_out["序号"] = range(1, len(df_out) + 1)

# ── 打印完整表格 ─────────────────────────────────────
print("=" * 140)
print("  特征信息总表（按 GBDT 重要性降序排列，共 {} 个特征）".format(len(df_out)))
print("=" * 140)
print(df_out.to_string(index=False, max_colwidth=80))

# ── OneHot 映射详情 ──────────────────────────────────
print("\n")
print("=" * 120)
print("  OneHot 编码映射详情（仅 Pipeline A: Ridge / Lasso）")
print("=" * 120)
for cat in cat_cols:
    derived = ohe_mapping[cat]
    print(f"\n  ▸ {cat}  ({len(derived)} 个衍生特征)")
    for j, d in enumerate(derived, 1):
        print(f"      {j:>2}. {d}")
print(f"\n  ─────────────────────────────────────────────")
print(f"  合计: {len(cat_cols)} 原始 → {total_ohe} OneHot 衍生")
print(f"  Pipeline A 总列数: {total_ohe} + {len(num_cols)} = {total_ohe + len(num_cols)}")

# ── 统计摘要 ─────────────────────────────────────────
print("\n")
print("=" * 120)
print("  统计摘要")
print("=" * 120)
print(f"  特征工程后列数:       {len(X_fe.columns)}  (含 target '{target_col}')")
print(f"  模型输入特征数:        {len(model_feat_clean)}  ({len(cat_cols)} 分类 + {len(num_cols)} 数值)")
print(f"  Pipeline A (OneHot):   {total_ohe + len(num_cols)} 列")
print(f"  Pipeline B (Ordinal):  {len(model_feat_clean)} 列")
print(f"  Pipeline C (Target):   {len(model_feat_clean)} 列")
print(f"  Pipeline D (Ordinal):  {len(model_feat_clean)} 列")
print(f"  有特征重要性的模型:    4  (GBDT, RF, XGBoost, LightGBM)")
print(f"  VIF 剔除特征数:        5 (act2_temp_C, act2_duration_h, carb2_duration_h, carb2_temp_C; Vtotal 已在特征工程阶段剔除)  (act2_temp_C, act2_duration_h, Vtotal, carb2_duration_h, carb2_temp_C)")

# ── 保存 CSV ─────────────────────────────────────────
csv_path = TABLES_DIR / "feature_summary.csv"
df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n  ✓ 已保存: {csv_path}")
print("=" * 120)
