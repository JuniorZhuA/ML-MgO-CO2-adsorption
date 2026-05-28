"""80/20 Train/Test 分割出版级图表 — 全部7个非线性模型

5×5 GridSpec 布局:
- 主图: 空心圆散点 + y=x 虚线 + Train/Test 回归线
- 顶部: 边缘直方图 (True_Value)
- 右侧: 边缘直方图 (Predicted_Value)
- 底部: 残差空心圆散点

文本框 (4行): 模型名称 / Train R² / Test R² / Test RMSE
"""

import sys
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_squared_error

from src.config import TABLES, FIGURES

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── 出版级全局设置 ────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# ── 配色 ──────────────────────────────────────────────────────────────────
TRAIN_EDGE = '#D32F2F'
TEST_EDGE  = '#1976D2'

TRAIN_FILL_HIST = '#E57373'
TEST_FILL_HIST  = '#64B5F6'

# ── 7个非线性模型 ─────────────────────────────────────────────────────────
MODELS = ["TabPFN", "GBDT", "RF", "XGBoost", "LightGBM", "GPR", "SVR"]


def plot_one_model(model_name: str):
    csv_path = TABLES / "prediction_tables" / f"Figure_4_{model_name}_Predictions.csv"
    df = pd.read_csv(csv_path)
    train_df = df[df['Set'] == 'Train']
    test_df = df[df['Set'] == 'Test']

    r2_train = r2_score(train_df['True_Value'], train_df['Predicted_Value'])
    r2_test = r2_score(test_df['True_Value'], test_df['Predicted_Value'])
    rmse_test = np.sqrt(mean_squared_error(test_df['True_Value'], test_df['Predicted_Value']))

    print(f"{model_name}: Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:.2f}")

    # ── 回归参数 ─────────────────────────────────────────────────────────
    x_test = test_df['True_Value'].values
    y_test = test_df['Predicted_Value'].values
    sl_te, ic_te, _, _, _ = linregress(x_test, y_test)

    x_train = train_df['True_Value'].values
    y_train = train_df['Predicted_Value'].values
    sl_tr, ic_tr, _, _, _ = linregress(x_train, y_train)

    min_val = min(df['True_Value'].min(), df['Predicted_Value'].min())
    max_val = max(df['True_Value'].max(), df['Predicted_Value'].max())
    x_fit = np.linspace(min_val, max_val, 200)

# ── GridSpec 布局 ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 8), dpi=300)
    gs = fig.add_gridspec(5, 5, hspace=0.0, wspace=0.0)

    ax_main  = fig.add_subplot(gs[1:4, 0:4])
    ax_top   = fig.add_subplot(gs[0, 0:4], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 4], sharey=ax_main)
    ax_res   = fig.add_subplot(gs[4, 0:4], sharex=ax_main)

    # ═══════════════════════════════════════════════════════════════════
    # 1. 主散点图 (空心圆)
    # ═══════════════════════════════════════════════════════════════════
    ax_main.scatter(x_train, y_train,
                    facecolors='none', edgecolors=TRAIN_EDGE, linewidths=1.0,
                    alpha=0.55, marker='o', s=50, label='Train', zorder=3)
    ax_main.scatter(x_test, y_test,
                    facecolors='none', edgecolors=TEST_EDGE, linewidths=1.2,
                    alpha=1.0, marker='o', s=55, label='Test', zorder=5)

    # y=x 虚线
    ax_main.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

    # 训练集回归线 (红色, 无 CI)
    ax_main.plot(x_fit, sl_tr * x_fit + ic_tr, color=TRAIN_EDGE, lw=1.5)

    # 测试集回归线 (蓝色)
    ax_main.plot(x_fit, sl_te * x_fit + ic_te, color=TEST_EDGE, lw=1.5)

    # 4行文本框: 模型名称 / Train R² / Test R² / Test RMSE
    r2_text = (f"{model_name}\n"
               f"Train R² = {r2_train:.4f}\n"
               f"Test R² = {r2_test:.4f}\n"
               f"Test RMSE = {rmse_test:.2f}")
    ax_main.text(0.95, 0.05, r2_text,
                 transform=ax_main.transAxes, ha='right', va='bottom',
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='#BDBDBD', linewidth=0.8, alpha=0.9))

    ax_main.set_ylabel('Predicted CO$_2$ Uptake (mg/g)', fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper left', fontsize=12, frameon=True)
    ax_main.tick_params(labelbottom=False)

    # ═══════════════════════════════════════════════════════════════════
    # 2. 顶部直方图
    # ═══════════════════════════════════════════════════════════════════
    sns.histplot(data=df, x='True_Value', hue='Set',
                 palette={'Train': TRAIN_FILL_HIST, 'Test': TEST_FILL_HIST},
                 ax=ax_top, kde=True, element='step', common_norm=False, legend=False)
    ax_top.axis('off')

    # ═══════════════════════════════════════════════════════════════════
    # 3. 右侧直方图
    # ═══════════════════════════════════════════════════════════════════
    sns.histplot(data=df, y='Predicted_Value', hue='Set',
                 palette={'Train': TRAIN_FILL_HIST, 'Test': TEST_FILL_HIST},
                 ax=ax_right, kde=True, element='step', common_norm=False, legend=False)
    ax_right.axis('off')

    # ═══════════════════════════════════════════════════════════════════
    # 4. 底部残差图 (空心圆)
    # ═══════════════════════════════════════════════════════════════════
    ax_res.scatter(x_train, train_df['Residual'].values,
                   facecolors='none', edgecolors=TRAIN_EDGE, linewidths=0.8,
                   alpha=0.55, marker='o', s=30, zorder=3)
    ax_res.scatter(x_test, test_df['Residual'].values,
                   facecolors='none', edgecolors=TEST_EDGE, linewidths=1.0,
                   alpha=1.0, marker='o', s=35, zorder=5)
    ax_res.axhline(0, color='gray', linestyle='--', lw=1.5)
    ax_res.set_xlabel('Experimental CO$_2$ Uptake (mg/g)', fontsize=14, fontweight='bold')
    ax_res.set_ylabel('Residuals (mg/g)', fontsize=12, fontweight='bold')

    # ── 保存 ─────────────────────────────────────────────────────────────
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES / f"Figure_4_{model_name}_Marginal.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"  已保存: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("生成 80/20 Train/Test 分割边际分布图 (7个模型)")
    print("=" * 60)
    for model in MODELS:
        plot_one_model(model)
    print(f"\n全部完成，共 {len(MODELS)} 张图，保存至 {FIGURES}")
