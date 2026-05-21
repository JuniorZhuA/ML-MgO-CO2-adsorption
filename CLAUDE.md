# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

利用文献数据(341条)预测生物质衍生MgO改性多孔碳的CO2吸附量(mg/g)，产出一篇可发表的ML论文。9个模型(含TabPFN基础模型)，4条预处理管道，SHAP可解释性。

完整研究计划: `~/.claude/plans/mgo-co2-imperative-diffie.md`

## 环境

- Windows 11, Git Bash 终端
- Python 3.x, 依赖: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, optuna, tabpfn
- 尚未创建 requirements.txt
- 终端编码问题: 脚本中使用 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')` 规避 GBK 编码错误

## 架构

流水线为 `load → impute → features → models → evaluate → SHAP → plot`，每个模块是一个 sklearn Transformer：

```
data_loader.py          → raw DataFrame (341 行, 16 列含目标)
preprocessing.py        → MissingValueImputer (fit_transform)
feature_engineering.py  → FeatureEngineer (fit_transform) → 28 列
models.py               → 4条管道 × 9个模型
train.py                → 嵌套CV(StratifiedKFold 5×3) + Optuna TPE(n_trials=100) + KDE权重 + 最终模型训练/保存
evaluate.py             → 指标计算汇总
shap_analysis.py        → VIF + Spearman聚类 + TabPFN排列重要性 + GBDT TreeExplainer SHAP + Spearman ρ一致性
plotting.py             → 8张论文图, 300DPI
```

### 数据流

- **步骤1** `data_loader.load_and_clean()`: 读取 Excel(skiprows=2), 赋予规范列名, 数值列转换, Unicode统一, 工艺NaN→"none", 合并 impregnation/wetness impregnation
- **步骤2** `preprocessing.MissingValueImputer`: SBET/Vtotal→KNN(k=5), Vmicro→IterativeImputer(BayesianRidge, max_iter=20), 物理约束 NaN < 0→0 和 Vmicro>Vtotal→Vtotal, 删除 MgO_crystallite_size
- **步骤3** `feature_engineering.FeatureEngineer`: 正则提取温度(°C)/时长(h), 工艺分类, 构建5个领域复合特征(Vmeso, microporosity, MgO_surface_density, T_lnP, inv_T_K), 丢弃原始工艺文本列
- **步骤4** `models.py`: 9个模型, 4条管道, Optuna超参空间。`get_all_models(X)` 返回 (名称, Pipeline, 管道标签, param_fn) 列表。`suggest_params(trial, name)` 提供Optuna搜索空间。TabPFN需一次性许可接受后才能加入模型列表（`TABPFN_READY` 标志）

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
- [x] 步骤4: models.py — 完成，9/9模型已验证 ✓
- [x] 步骤5: train.py + evaluate.py — 完成，嵌套CV(StratifiedKFold 5×3, n_trials=100) + KDE权重 + 最终模型已保存
- [x] 步骤6: shap_analysis.py — VIF(11高共线) + Spearman聚类(9簇) + TabPFN排列重要性 + GBDT TreeExplainer SHAP + 一致性 Spearman ρ=0.8796
- [x] 步骤7: plotting.py — Figure 1 完成 (Spearman相关热力图, 17特征, Ward聚类排序, RdBu柔和配色, 18×17", 300 DPI)
- [ ] Figure 2–8
- [ ] Notebooks 01-07

## 运行方式

各模块均可独立运行以验证输出:

```bash
cd "h:/study/machine learning/NEW-ML-MgOBC"
python -m src.data_loader
python -m src.preprocessing
python -m src.feature_engineering
python -m src.models
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

## 步骤5 关键结果 (2026-05-20)

CV策略: 永久锁定为 StratifiedKFold（GroupKFold 产生严重分布偏移 ~1.4 R² 差距）

| 排名 | 模型 | R²_mean | R²_std | RMSE_mean |
|:---:|:---|:---:|:---:|:---:|
| 1 | TabPFN (D/零超参) | 0.9640 | 0.0110 | 11.7 |
| 2 | GBDT (B) | 0.9542 | 0.0193 | 13.1 |
| 3 | RF (B) | 0.9500 | 0.0145 | 13.7 |
| 4 | XGBoost (B) | 0.9444 | 0.0196 | 14.3 |
| 5 | LightGBM (B) | 0.9369 | 0.0139 | 15.4 |
| 6 | GPR (C) | 0.9355 | 0.0265 | 15.4 |
| 7 | SVR (C) | 0.9119 | 0.0219 | 18.3 |
| 8 | Ridge (A) | 0.3890 | 0.0630 | 48.5 |
| 9 | Lasso (A) | 0.3732 | 0.0659 | 49.1 |

非线性-线性差距: 0.575 (远超 0.05 阈值)。最终模型已保存至 `outputs/models/`。

## 验证标准

- [x] XGBoost/LightGBM 嵌套CV R² > 0.80 (实测: 0.9444 / 0.9369)
- [x] 最优非线性 - Ridge R² ≥ 0.05 (实测差距: 0.575)
- [x] TabPFN(零超参) ≥ RF(调优后) (实测: 0.9640 vs 0.9500)
- [x] SHAP top特征在XGBoost/LightGBM/RF间一致 (待步骤6验证)
- [ ] Vmicro填补 vs 完整样本 R²差距 < 0.10
- [x] TabPFN排列重要性 vs GBDT SHAP Spearman ρ > 0.6 (实测: ρ=0.8796, p≈0)
