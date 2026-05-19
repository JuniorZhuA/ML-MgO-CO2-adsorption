# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

利用文献数据(341条)预测生物质衍生MgO改性多孔碳的CO2吸附量(mg/g)，产出一篇可发表的ML论文。9个模型(含TabPFN基础模型)，4条预处理管道，SHAP可解释性。

完整研究计划: `~/.claude/plans/mgo-co2-imperative-diffie.md`

## 环境

- Windows 11, Git Bash 终端
- Python 3.x, 依赖: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, optuna, tabpfn
- 尚未创建 requirements.txt——请在步骤4实施时根据实际导入创建
- 终端编码问题: 脚本中使用 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')` 规避 GBK 编码错误

## 架构

流水线为 `load → impute → features → models → evaluate → SHAP → plot`，每个模块是一个 sklearn Transformer：

```
data_loader.py          → raw DataFrame (341 行, 16 列含目标)
preprocessing.py        → MissingValueImputer (fit_transform)
feature_engineering.py  → FeatureEngineer (fit_transform) → 28 列
models.py               → 4条管道 × 9个模型
train.py                → 嵌套CV + Optuna
evaluate.py             → 指标计算汇总
shap_analysis.py        → VIF + Spearman聚类 + SHAP/排列重要性
plotting.py             → 8张论文图, 300DPI
```

### 数据流

- **步骤1** `data_loader.load_and_clean()`: 读取 Excel(skiprows=2), 赋予规范列名, 数值列转换, Unicode统一, 工艺NaN→"none", 合并 impregnation/wetness impregnation
- **步骤2** `preprocessing.MissingValueImputer`: SBET/Vtotal→KNN(k=5), Vmicro→IterativeImputer(BayesianRidge, max_iter=20), 物理约束 NaN < 0→0 和 Vmicro>Vtotal→Vtotal, 删除 MgO_crystallite_size
- **步骤3** `feature_engineering.FeatureEngineer`: 正则提取温度(°C)/时长(h), 工艺分类, 构建5个领域复合特征(Vmeso, microporosity, MgO_surface_density, T_lnP, inv_T_K), 丢弃原始工艺文本列
- **步骤4** `models.py` (待实现): 9个模型, 4条管道

### 四条预处理管道

| 管道 | 适用模型 | 分类变量 | 数值缩放 | NaN处理 |
|---|---|---|---|---|
| A | Ridge, Lasso | OneHotEncoder | StandardScaler | SimpleImputer(median) |
| B | RF, XGBoost, LightGBM, GBDT | OrdinalEncoder | 无 | 原生支持 |
| C | SVR, GPR | TargetEncoder | StandardScaler + log1p(y) | SimpleImputer(median) |
| D | TabPFN | OrdinalEncoder | 无(内置预处理) | 原生支持 |

### 配置

`src/config.py`: `ROOT` 基于 `__file__` 推导; `SEED = 42`; 目标列 `TARGET = "CO2_uptake_mg_g"`; 数据路径通过 `DATA_RAW / "0514biochar-MgOdata.xlsx"` 获取。

## 当前进度

- [x] 步骤1: data_loader.py — 完成并验证
- [x] 步骤2: preprocessing.py — 完成并验证
- [x] 步骤3: feature_engineering.py — 完成并验证
- [ ] 步骤4: models.py — 待实现
- [ ] 步骤5: train.py + evaluate.py
- [ ] 步骤6: shap_analysis.py
- [ ] 步骤7: plotting.py
- [ ] Notebooks 01-07

## 运行方式

各模块均可独立运行以验证输出:

```bash
cd "h:/study/machine learning/NEW-ML-MgOBC"
python -m src.data_loader
python -m src.preprocessing
python -m src.feature_engineering
```

当前状态可串联验证:

```python
from src.data_loader import load_and_clean
from src.preprocessing import MissingValueImputer
from src.feature_engineering import FeatureEngineer
df = load_and_clean()
df = MissingValueImputer().fit_transform(df)
df = FeatureEngineer().fit_transform(df)
```

## 验证标准

- XGBoost/LightGBM 嵌套CV R² > 0.80
- 最优非线性 - Ridge R² ≥ 0.05
- TabPFN(零超参) ≥ RF(调优后)
- SHAP top特征在XGBoost/LightGBM/RF间一致
- Vmicro填补 vs 完整样本 R²差距 < 0.10
- TabPFN排列重要性 vs XGBoost SHAP Spearman ρ > 0.6
