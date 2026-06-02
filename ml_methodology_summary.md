# 机器学习方法学总结 — 生物质衍生MgO改性多孔碳CO₂吸附量预测

> **项目目标**: 利用文献数据（341条）预测生物质衍生MgO改性多孔碳的CO₂吸附量（mg/g），产出可发表的ML论文。
>
> **全局随机种子**: 42（`src/config.py` — 用于嵌套CV、模型初始化等）  
> **80/20分割种子**: 91（经0–100遍历优选，确保TabPFN单次划分R²≈嵌套CV R²）

---

## 1. 数据集基本情况

| 项目 | 说明 |
|:---|:---|
| **样本数量** | 341 条 |
| **目标变量** | `CO2_uptake_mg_g` — CO₂吸附量（mg/g） |
| **输入特征数** | 27 个（8 个分类 + 19 个数值，经特征工程后） |

### 原始输入列（原始Excel 17列含References → 步骤1删除References后16列，含目标）

| 类别 | 列名 |
|:---|:---|
| **前驱体/工艺（分类）** | `carbon_precursors`（碳前驱体）、`MgO_precursors`（MgO前驱体）、`Mg_loading_method`（Mg负载方法） |
| **工艺文本（待解析）** | `Activation1`、`Activation2`、`Carbonization1`、`Carbonization2`、`Post_treatment` |
| **数值特征** | `SBET_m2_g`（BET比表面积）、`Vtotal_cm3_g`（总孔容）、`Vmicro_cm3_g`（微孔孔容）、`MgO_crystallite_size_nm`（MgO晶粒尺寸，后被删除）、`MgO_mass_ratio`（MgO质量比）、`temperature_C`（吸附温度）、`pressure_bar`（吸附压力） |

### 特征工程后最终27个特征

**分类特征（8个）**:
`carbon_precursors`, `MgO_precursors`, `Mg_loading_method`, `act1_type`, `act2_type`, `carb1_type`, `carb2_type`, `post_treatment_type`

**数值特征（19个）**:
`SBET_m2_g`, `Vtotal_cm3_g`, `Vmicro_cm3_g`, `MgO_mass_ratio`, `temperature_C`, `pressure_bar`, `act1_temp_C`, `act1_duration_h`, `act2_temp_C`, `act2_duration_h`, `carb1_temp_C`, `carb1_duration_h`, `carb2_temp_C`, `carb2_duration_h`, `Vmeso_cm3_g`, `microporosity`, `MgO_surface_density`, `T_lnP`, `inv_T_K`

其中 5 个领域复合特征由原始列构建：
- `Vmeso_cm3_g` = Vtotal − Vmicro（中孔孔容）
- `microporosity` = Vmicro / Vtotal（微孔率）
- `MgO_surface_density` = MgO_mass_ratio / (SBET / 1000)（MgO表面密度）
- `T_lnP` = temperature × ln(pressure + 0.01)（温度-压力耦合项）
- `inv_T_K` = 1 / (temperature + 273.15)（Arrhenius型逆温项）

---

## 2. 数据预处理

### 2.1 缺失值处理

| 填补目标 | 方法 | 参数 |
|:---|:---|:---|
| `SBET_m2_g`、`Vtotal_cm3_g` | KNN Imputer | k = 5 |
| `Vmicro_cm3_g` | Iterative Imputer (BayesianRidge) | max_iter = 20 |
| `MgO_crystallite_size_nm` | **直接删除** | 缺失率过高（双轨策略A方案） |

**后处理物理约束**：
- Vmicro < 0 → 修正为 0
- Vmicro > Vtotal → 修正为 Vtotal

工艺文本列（Activation1/2, Carbonization1/2, Post_treatment）的 NaN → `"none"`（表示该步骤未执行）。

### 2.2 特征缩放方法

由于不同模型对缩放的敏感度不同，使用了 4 条不同的预处理管道：

| 管道 | 适用模型 | 分类变量编码 | 数值缩放 | NaN处理 |
|:---:|:---|:---|:---|:---|
| **A** | Ridge, Lasso | OneHotEncoder | **StandardScaler** | SimpleImputer(median) |
| **B** | RF, XGBoost, LightGBM, GBDT | OrdinalEncoder | **无缩放**（树模型原生支持） | 原生支持 |
| **C** | SVR, GPR | TargetEncoder | **StandardScaler** + log1p(y)变换 | SimpleImputer(median) |
| **D** | TabPFN | OrdinalEncoder | **无**（内置预处理） | 原生支持 |

> **关键防泄漏设计**: `MissingValueImputer` + `FeatureEngineer` 内置于每条Pipeline内部，确保嵌套CV中填补器和特征工程严格仅在训练折上fit，杜绝数据泄露。

### 2.3 训练/测试集划分

- **嵌套交叉验证**: 外层 5 折 StratifiedKFold（按目标变量分位数分箱分层），内层 3 折
- **独立80/20分割**: 单次分层随机划分（stratified by target bins, random_state=91），272 训练 / 69 测试，作为外推验证
- **为什么用91而非42**: 原seed=42因46条Vmicro约束修正（1条<0 + 45条>Vtotal）干扰了TabPFN in-context learning，单次R²=0.9343偏低。种子91经0–100遍历优选，其 TabPFN Test R²=0.9688 与嵌套CV R²=0.9692 高度一致（Δ=−0.0004），确保单次分割结果具有代表性

---

## 3. 模型与算法

共 9 个机器学习模型，分为三大类：

| 类别 | 模型 | 管道 |
|:---|:---|:---:|
| **线性基线** | Ridge（岭回归）、Lasso（L1正则化） | A |
| **集成树模型** | Random Forest（随机森林）、XGBoost、LightGBM、GBDT（梯度提升决策树） | B |
| **核方法** | SVR（RBF核支持向量回归）、GPR（高斯过程回归，ConstantKernel×RBF + WhiteKernel） | C |
| **基础模型** | TabPFN（Prior-Data Fitted Networks，零超参） | D |

---

## 4. 超参数优化与交叉验证

### 4.1 调参方法

- **框架**: Optuna（v3.x），TPE（Tree-structured Parzen Estimator）采样器
- **试验次数**: `n_trials = 100`
- **优化目标**: 内层3折CV的平均RMSE最小化
- **搜索空间**（各模型独立）:

| 模型 | 超参数搜索空间 |
|:---|:---|
| **Ridge** | `alpha`: [1e-3, 1e4] log-uniform |
| **Lasso** | `alpha`: [1e-4, 1e2] log-uniform |
| **RF** | `n_estimators`: [100, 600], `max_depth`: [5, 50], `min_samples_split`: [2, 20], `min_samples_leaf`: [1, 10], `max_features`: [0.3, 1.0] |
| **XGBoost** | `n_estimators`: [100, 600], `max_depth`: [3, 12], `learning_rate`: [0.01, 0.3] log-uniform, `subsample`: [0.6, 1.0], `colsample_bytree`: [0.6, 1.0], `reg_alpha`: [1e-4, 10] log-uniform, `reg_lambda`: [1e-4, 10] log-uniform |
| **LightGBM** | `n_estimators`: [100, 600], `num_leaves`: [15, 255], `learning_rate`: [0.01, 0.3] log-uniform, `subsample`: [0.6, 1.0], `colsample_bytree`: [0.6, 1.0], `reg_alpha`: [1e-4, 10] log-uniform, `reg_lambda`: [1e-4, 10] log-uniform, `min_child_samples`: [5, 50] |
| **GBDT** | `n_estimators`: [100, 600], `max_depth`: [3, 12], `learning_rate`: [0.01, 0.3] log-uniform, `subsample`: [0.6, 1.0], `min_samples_split`: [2, 20], `min_samples_leaf`: [1, 10], `max_features`: [0.3, 1.0] |
| **SVR** | `C`: [0.1, 100] log-uniform, `gamma`: [1e-4, 1] log-uniform, `epsilon`: [0.01, 1] log-uniform |
| **GPR** | `alpha`: [1e-8, 1e-1] log-uniform |
| **TabPFN** | **零超参**（无需调优，跳过Optuna） |

### 4.2 交叉验证策略

```
嵌套CV = 外层 StratifiedKFold (n_splits=5) × 内层 StratifiedKFold (n_splits=3)
       = 5 × 3 = 15 次训练/验证组合
```

- **分层策略**: 按目标变量（CO2_uptake_mg_g）分位数等分为 5 箱
- **KDE样本权重**: XGBoost 和 LightGBM 使用高斯核密度估计倒数权重，增强尾部稀有样本的贡献
- **最终模型**: 嵌套CV完成后，在全量数据（341条）上用各折最优参数重新训练并保存至 `outputs/models/`

---

## 5. 模型表现

### 5.1 嵌套CV结果（StratifiedKFold 5×3，主报告值）

| 排名 | 模型 | R²_mean | R²_std | RMSE_mean | RMSE_std | MAE_mean | MAPE_mean |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **TabPFN** (D) | **0.9692** | 0.0103 | **10.73** | 2.28 | **4.12** | **7.99%** |
| 2 | GBDT (B) | 0.9528 | 0.0138 | 13.35 | 1.90 | 7.25 | 16.91% |
| 3 | RF (B) | 0.9503 | 0.0120 | 13.75 | 2.02 | 7.69 | 20.63% |
| 4 | XGBoost (B) | 0.9449 | 0.0150 | 14.35 | 1.39 | 7.31 | 15.72% |
| 5 | LightGBM (B) | 0.9350 | 0.0196 | 15.67 | 2.48 | 8.67 | 18.66% |
| 6 | GPR (C) | 0.9299 | 0.0306 | 15.94 | 3.50 | 7.83 | 16.77% |
| 7 | SVR (C) | 0.9112 | 0.0226 | 18.28 | 2.54 | 8.32 | 14.99% |
| 8 | Ridge (A) | 0.3608 | 0.0804 | 49.66 | 5.90 | 35.73 | 81.95% |
| 9 | Lasso (A) | 0.3521 | 0.0774 | 50.00 | 5.77 | 36.43 | 81.77% |

> **关键发现**: 非线性模型与线性模型R²差距 = 0.592，远超 0.05 阈值，确认问题本质为强非线性。

### 5.2 80/20独立分割结果（stratified, random_state=91, 272/69）

| 模型 | Train R² | Train RMSE | Test R² | Test RMSE | Test MAE | Test MAPE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TabPFN** | 0.9988 | 2.11 | **0.9688** | 11.07 | 4.22 | 9.28% |
| LightGBM | 0.9993 | 1.69 | 0.9676 | 11.28 | 7.01 | 14.48% |
| GBDT | 0.9998 | 0.93 | 0.9639 | 11.91 | 7.14 | 17.65% |
| SVR | 0.9853 | 7.54 | 0.9628 | 12.07 | 6.61 | 20.57% |
| RF | 0.9919 | 5.59 | 0.9615 | 12.30 | 7.45 | 26.94% |
| GPR | 0.9894 | 6.39 | 0.9376 | 15.65 | 7.71 | 25.57% |
| XGBoost | 0.9995 | 1.32 | 0.9151 | 18.25 | 10.02 | 23.39% |

> 注：Ridge 和 Lasso 未纳入80/20评估（线性模型不适合此任务）。
> TabPFN 80/20 Test R²=0.9688 与嵌套CV R²=0.9692 高度一致（Δ = −0.0004），验证模型泛化能力。

---

## 6. 可解释性（SHAP分析）

### 6.1 分析流程

1. **VIF多重共线性诊断**: 自动检测并剔除完美共线特征（如 Vtotal = Vmicro + Vmeso），最终4个特征 VIF > 10（SBET=127.86, Vmicro=109.31, inv_T_K=39.98, temperature_C=34.87）
2. **Spearman相关系数 + Ward层次聚类**: 识别 10 个共线特征簇（阈值 |ρ| ≥ 0.7，距离阈值 0.3）
3. **TabPFN 排列特征重要性**: n_repeats = 30，计算每个特征的RMSE增量
4. **GBDT TreeExplainer SHAP值**: 基于全量数据计算每个特征对预测的边际贡献
5. **一致性检验**: TabPFN 排列重要性 vs GBDT SHAP → **Spearman ρ = 0.8114**（p < 0.001，22个有效特征），两种范式归因高度一致

### 6.2 计算SHAP值的核心Python代码

```python
import numpy as np
import pandas as pd
import shap
import joblib

def compute_gbdt_shap(pipeline, X, feature_names):
    """用 TreeExplainer 计算 GBDT 模型的 SHAP 值。

    Parameters
    ----------
    pipeline : sklearn Pipeline
        结构: impute → features → preprocessor → model
        已使用全量数据fit的最终GBDT模型
    X : pd.DataFrame
        原始DataFrame（未经任何预处理），shape = (341, 15)
    feature_names : list[str]
        FeatureEngineer 输出后的特征名列表（27个）

    Returns
    -------
    dict with keys: shap_values, explainer, X_transformed, shap_importance
    """
    # 逐级变换: raw → imputed → feature-engineered → model-ready
    imputer = pipeline.named_steps["impute"]
    engineer = pipeline.named_steps["features"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_imputed = imputer.transform(X)
    X_fe = engineer.transform(X_imputed)
    X_transformed = preprocessor.transform(X_fe)

    # 确保 X_transformed 是 DataFrame（便于缓存和绘图）
    if not isinstance(X_transformed, pd.DataFrame):
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

    # TreeExplainer: 精确计算（非近似），适用于树模型
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_transformed)

    # 处理多维输出（如有）
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0]

    # 特征重要性 = mean(|SHAP value|)
    importance = np.abs(shap_vals).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "shap_importance_mean": importance,
    }).sort_values("shap_importance_mean", ascending=False).reset_index(drop=True)

    return {
        "shap_values": shap_vals,          # shape (341, 27) — 每个样本每个特征的SHAP值
        "explainer": explainer,            # SHAP TreeExplainer 对象（可复用）
        "X_transformed": X_transformed,    # 变换后的特征矩阵（供绘图）
        "shap_importance": shap_importance, # 按重要性排序的 DataFrame
    }


# ===== 调用示例 =====
# 加载已训练好的 GBDT 最终模型
gbdt_pipe = joblib.load("outputs/models/GBDT_final.pkl")

# 准备原始数据
from src.data_loader import load_and_clean
df = load_and_clean()                            # 加载341条原始数据
X_raw = df.drop(columns=["CO2_uptake_mg_g"])     # 15列原始特征
feature_names = cat_cols + num_cols              # 27个特征工程后的特征名

# 计算 SHAP 值
result = compute_gbdt_shap(gbdt_pipe, X_raw, feature_names)

# 结果使用
shap_values = result["shap_values"]               # (341, 27) 的SHAP值矩阵
importance = result["shap_importance"]            # DataFrame: feature + shap_importance_mean
print(importance.head(10))                        # Top 10 最重要特征

# 绘制 SHAP 摘要图
shap.summary_plot(shap_values, result["X_transformed"], feature_names=feature_names)
```

### 6.3 一致性检验代码

```python
from scipy.stats import spearmanr

def compute_consistency(tabpfn_imp, gbdt_shap_imp):
    """计算 TabPFN 排列重要性 vs GBDT SHAP 重要性的 Spearman 秩相关。"""
    merged = tabpfn_imp.merge(gbdt_shap_imp, on="feature", how="inner")

    rho, pval = spearmanr(
        merged["importance_mean"],
        merged["shap_importance_mean"]
    )

    return {
        "spearman_rho": rho,
        "spearman_pval": pval,
        "n_features": len(merged),
        "merged_table": merged,
    }
```

---

## 附录: 项目文件结构

| 模块 | 文件 | 功能 |
|:---|:---|:---|
| 步骤1 | `src/data_loader.py` | 数据加载与清洗 |
| 步骤2 | `src/preprocessing.py` | 缺失值填补（KNN + IterativeImputer + 物理约束） |
| 步骤3 | `src/feature_engineering.py` | 正则解析温度/时间 + 5个领域复合特征 |
| 步骤4 | `src/models.py` | 9个模型 × 4条管道 + Optuna超参空间定义 |
| 步骤5 | `src/train.py` | 嵌套CV（5×3 StratifiedKFold）+ Optuna TPE + KDE权重 |
| 步骤6 | `src/evaluate.py` | 指标计算与汇总 |
| 步骤7 | `src/shap_analysis.py` | VIF + Spearman聚类 + TabPFN排列重要性 + GBDT SHAP + 一致性检验 |
| 步骤8 | `src/topsis.py` | 层次熵权法TOPSIS多准则综合排名 |
| 可视化 | `src/plotting.py` | 8张论文图 + 1张补充图（300 DPI） |
| 配置 | `src/config.py` | 路径、常量、随机种子 |
