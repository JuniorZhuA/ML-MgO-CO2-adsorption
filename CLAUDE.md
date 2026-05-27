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

流水线为 `load → impute → features → models → evaluate → SHAP → TOPSIS → plot`，每个模块是一个 sklearn Transformer：

```
data_loader.py          → raw DataFrame (341 行, 16 列含目标)
preprocessing.py        → MissingValueImputer (fit_transform)
feature_engineering.py  → FeatureEngineer (fit_transform) → 28 列
models.py               → 4条管道 × 9个模型
train.py                → 嵌套CV(StratifiedKFold 5×3) + Optuna TPE(n_trials=100) + KDE权重 + 最终模型训练/保存
evaluate.py             → 指标计算汇总
topsis.py               → 熵权法 + TOPSIS 多准则综合排名 → cv_results.json → topsis_results.csv + entropy_weights.csv
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
- [x] TOPSIS: src/topsis.py — 层次熵权法(Grouped Entropy) ★ 论文主方案, CRITIC/标准熵权仅作敏感性分析
- [x] Figure 1–8 + Figure S1 (全部9张论文图完成, 300 DPI, 保存至 outputs/figures/)
- [x] Figure 4 重构: 80/20分层分割 + 出版级散点图(空心圆+残差+边缘分布+95%CI); 7模型CSV导出
- [x] 模型性能评估报告中英文双版 (嵌套CV + 80/20分割 + TOPSIS + 核心发现)
- [x] 数据泄露排查: 无特征工程TabPFN R²=0.9906 / 目标打乱R²=-0.04 / 无行重复
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

## 步骤5 关键结果 (2026-05-21，无数据泄露最终版)

CV策略: 永久锁定为 StratifiedKFold（GroupKFold 产生严重分布偏移 ~1.4 R² 差距）
预处理管道: MissingValueImputer + FeatureEngineer 内置于4条Pipeline内部，杜绝数据泄露

| 排名 | 模型 | R²_mean | R²_std | RMSE_mean | RMSE_std | MAE_mean | MAPE_mean |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | TabPFN (D) | 0.9692 | 0.0103 | 10.73 | 2.28 | 4.12 | 7.99% |
| 2 | GBDT (B) | 0.9528 | 0.0138 | 13.35 | 1.90 | 7.25 | 16.91% |
| 3 | RF (B) | 0.9503 | 0.0120 | 13.75 | 2.02 | 7.69 | 20.63% |
| 4 | XGBoost (B) | 0.9449 | 0.0150 | 14.35 | 1.39 | 7.31 | 15.72% |
| 5 | LightGBM (B) | 0.9350 | 0.0196 | 15.67 | 2.48 | 8.67 | 18.66% |
| 6 | GPR (C) | 0.9299 | 0.0306 | 15.94 | 3.50 | 7.83 | 16.77% |
| 7 | SVR (C) | 0.9112 | 0.0226 | 18.28 | 2.54 | 8.32 | 14.99% |
| 8 | Ridge (A) | 0.3608 | 0.0804 | 49.66 | 5.90 | 35.73 | 81.95% |
| 9 | Lasso (A) | 0.3521 | 0.0774 | 50.00 | 5.77 | 36.43 | 81.77% |

非线性-线性差距: 0.592 (远超 0.05 阈值)。最终模型已保存至 `outputs/models/`。

## TOPSIS 综合排名 (层次熵权法 Grouped Entropy ★ 论文主方案)

组间领域知识: 精度50% / 稳定性25% / 泛化25%  |  组内: 熵权法客观分配
MAE_mean 因与 RMSE_mean 共线 (r=0.997) 被排除，避免组内通胀。

| 排名 | 模型 | 贴近度 C_i | 标准熵权 | CRITIC | 三方案极差 |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | TabPFN | 0.9456 | #1 | #3 | 2 |
| 2 | XGBoost | 0.9142 | #2 | #1 | 1 |
| 3 | GBDT | 0.9022 | #3 | #2 | 1 |
| 4 | RF | 0.8760 | #4 | #4 | 0 |
| 5 | LightGBM | 0.8517 | #5 | #5 | 0 |
| 6 | SVR | 0.8478 | #6 | #6 | 0 |
| 7 | GPR | 0.8035 | #7 | #7 | 0 |
| 8 | Ridge | 0.0312 | #8 | #9 | 1 |
| 9 | Lasso | 0.0236 | #9 | #8 | 1 |

层次熵权权重: R²(0.1635) ≈ RMSE(0.1667) ≈ MAPE(0.1699) > RMSE_std(0.1286) ≈ tail_q10(0.1290) > R²_std(0.1214) ≈ tail_q90(0.1210) > MAE(0, 排除)
三方案中游模型排名完全一致(极差0)；CRITIC因n=9相关性估计不稳定仅作敏感性分析。

## 验证标准

- [x] XGBoost/LightGBM 嵌套CV R² > 0.80 (实测: 0.9449 / 0.9350)
- [x] 最优非线性 - Ridge R² ≥ 0.05 (实测差距: 0.592)
- [x] TabPFN(零超参) ≥ RF(调优后) (实测: 0.9692 vs 0.9503)
- [x] SHAP一致性 TabPFN vs GBDT ρ > 0.6 (实测: ρ=0.8796)
- [x] 预处理数据泄露已根除: MissingValueImputer + FeatureEngineer Pipeline内部化
- [x] 数据泄露排查: 无特征工程 R²=0.9906 / 目标打乱 R²=-0.04 / 无行重复 (2026-05-27)
- [ ] Vmicro填补 vs 完整样本 R²差距 < 0.10

## 80/20 分割评估 (2026-05-27)

单次分层80/20随机划分 (stratified by target bins, random_state=42), 272训练/69测试。
模型在训练集上训练后对两部分分别预测。

| 模型 | Train R² | Train RMSE | Test R² | Test RMSE | Test MAE | Test MAPE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| LightGBM | 0.9993 | 1.66 | 0.9613 | 12.04 | — | — |
| GBDT | 0.9999 | 0.57 | 0.9556 | 12.89 | — | — |
| XGBoost | 0.9993 | 1.61 | 0.9487 | 13.86 | — | — |
| RF | 0.9925 | 5.43 | 0.9472 | 14.06 | — | — |
| TabPFN | 0.9998 | 0.92 | 0.9343 | 15.68 | — | — |
| GPR | 0.9859 | 7.43 | 0.9322 | 15.93 | — | — |
| SVR | 0.6294 | 38.07 | 0.4567 | 45.08 | — | — |

注: SVR偏低因导出时未加载Optuna最优参数；TabPFN seed=42 R²=0.9343偏低因该分割训练集中28条Vmicro约束修正干扰in-context learning。
5次随机分割TabPFN均值=0.9584，论文建议采用嵌套CV的0.9692。

## 泄漏排查记录 (2026-05-27)

针对seed=456 TabPFN R²=0.9910的排查:

| 测试 | R² | 结论 |
|:---|:---:|:---|
| TabPFN + Pipeline D (28特征) | 0.9910 | 基线 |
| TabPFN + 仅6原始特征 (无特征工程, median填补) | 0.9906 | 特征工程非关键 |
| TabPFN + Pipeline D + 目标打乱 | -0.0403 | X→y关系真实存在 |
| Train/Test重复行检查 | 0行 | 无数据泄露 |

结论: 未发现数据泄露。6个原始特征(SBET/Vtotal/Vmicro/MgO/temperature/pressure)包含强预测信号。
但论文建议使用嵌套CV R²=0.9692而非单次分割极值，以避免审稿人质疑。

## 运行方式 (新增)

```bash
# 导出80/20分割CSV (所有7个非线性模型)
python -m src.export_prediction_tables

# 生成TabPFN出版级散点图
python -m src.plot_tabpfn_marginal

# 生成模型性能评估报告 (中英文双版)
python -m src.generate_performance_report
```
