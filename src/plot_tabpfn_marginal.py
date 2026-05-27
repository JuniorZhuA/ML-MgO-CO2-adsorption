"""基于用户模板生成 TabPFN Predicted vs Experimental 出版级图表

5×5 GridSpec 布局:
- 主图: 空心圆散点 + y=x 虚线 + 测试集回归线 + 95% CI 蓝色阴影
- 顶部: 边缘直方图 (True_Value)
- 右侧: 边缘直方图 (Predicted_Value)
- 底部: 残差空心圆散点
"""

import sys
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, t as t_dist
from sklearn.metrics import r2_score

from src.config import TABLES, FIGURES

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 出版级全局字体和样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

MODEL_NAME = "TabPFN"
CSV_PATH = TABLES / "prediction_tables" / f"Figure_4_{MODEL_NAME}_Predictions.csv"

# ── 读取数据 ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
train_df = df[df['Set'] == 'Train']
test_df = df[df['Set'] == 'Test']

r2_train = r2_score(train_df['True_Value'], train_df['Predicted_Value'])
r2_test = r2_score(test_df['True_Value'], test_df['Predicted_Value'])

print(f"{MODEL_NAME}: Train R² = {r2_train:.4f}, Test R² = {r2_test:.4f}")

# ── 配色 ──────────────────────────────────────────────────────────────
TRAIN_EDGE = '#D32F2F'   # 训练集描边 (深红)
TEST_EDGE  = '#1976D2'   # 测试集描边 (深蓝)
TEST_FILL  = '#90CAF9'   # 测试集 CI 填充 (浅蓝)
TRAIN_FILL_HIST = '#E57373'  # 训练集直方图
TEST_FILL_HIST  = '#64B5F6'  # 测试集直方图

# ── 拟合测试集回归参数 (用于 CI 计算) ──────────────────────────────────
x_test = test_df['True_Value'].values
y_test = test_df['Predicted_Value'].values
sl_te, ic_te, _, _, std_err = linregress(x_test, y_test)

x_train = train_df['True_Value'].values
y_train = train_df['Predicted_Value'].values
sl_tr, ic_tr, _, _, _ = linregress(x_train, y_train)

min_val = min(df['True_Value'].min(), df['Predicted_Value'].min())
max_val = max(df['True_Value'].max(), df['Predicted_Value'].max())

x_fit = np.linspace(min_val, max_val, 200)

# 参数化 95% CI: y_pred ± t_0.025 * SE
n_test = len(x_test)
t_crit = t_dist.ppf(0.975, n_test - 2)
x_mean = np.mean(x_test)
ssx = np.sum((x_test - x_mean) ** 2)
se_fit = std_err * np.sqrt(1 / n_test + (x_fit - x_mean) ** 2 / ssx)
ci_upper = sl_te * x_fit + ic_te + t_crit * se_fit
ci_lower = sl_te * x_fit + ic_te - t_crit * se_fit

# ── GridSpec 布局 ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 8), dpi=300)
gs = fig.add_gridspec(5, 5, hspace=0.0, wspace=0.0)

ax_main  = fig.add_subplot(gs[1:4, 0:4])            # 主图
ax_top   = fig.add_subplot(gs[0, 0:4], sharex=ax_main)    # 顶部边缘
ax_right = fig.add_subplot(gs[1:4, 4], sharey=ax_main)    # 右侧边缘
ax_res   = fig.add_subplot(gs[4, 0:4], sharex=ax_main)    # 底部残差

# ══════════════════════════════════════════════════════════════════════
# 1. 主散点图 (空心圆)
# ══════════════════════════════════════════════════════════════════════
ax_main.scatter(x_train, y_train,
                facecolors='none', edgecolors=TRAIN_EDGE, linewidths=1.0,
                alpha=1.0, marker='o', s=50, label='Train', zorder=3)
ax_main.scatter(x_test, y_test,
                facecolors='none', edgecolors=TEST_EDGE, linewidths=1.0,
                alpha=0.55, marker='o', s=50, label='Test', zorder=4)

# y=x 虚线
ax_main.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

# 训练集回归线 (红色, 无 CI)
ax_main.plot(x_fit, sl_tr * x_fit + ic_tr, color=TRAIN_EDGE, lw=1.5)

# 测试集回归线 + 95% CI 蓝色阴影
ax_main.plot(x_fit, sl_te * x_fit + ic_te, color=TEST_EDGE, lw=1.5)
ax_main.fill_between(x_fit, ci_lower, ci_upper,
                     color=TEST_FILL, alpha=0.30, edgecolor='none')

# 主图标注: 3 行文本框 (模型名 + Train R² + Test R²)
r2_text = (f"{MODEL_NAME}\n"
           f"Train R² = {r2_train:.4f}\n"
           f"Test R² = {r2_test:.4f}")
ax_main.text(0.95, 0.05, r2_text,
             transform=ax_main.transAxes, ha='right', va='bottom',
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='#BDBDBD', linewidth=0.8, alpha=0.9))

ax_main.set_ylabel('Predicted CO$_2$ Uptake (mg/g)', fontsize=14, fontweight='bold')
ax_main.legend(loc='upper left', fontsize=12, frameon=True)
ax_main.tick_params(labelbottom=False)

# ══════════════════════════════════════════════════════════════════════
# 2. 顶部直方图 (True_Value 边缘分布)
# ══════════════════════════════════════════════════════════════════════
sns.histplot(data=df, x='True_Value', hue='Set',
             palette={'Train': TRAIN_FILL_HIST, 'Test': TEST_FILL_HIST},
             ax=ax_top, kde=True, element='step', common_norm=False, legend=False)
ax_top.axis('off')

# ══════════════════════════════════════════════════════════════════════
# 3. 右侧直方图 (Predicted_Value 边缘分布)
# ══════════════════════════════════════════════════════════════════════
sns.histplot(data=df, y='Predicted_Value', hue='Set',
             palette={'Train': TRAIN_FILL_HIST, 'Test': TEST_FILL_HIST},
             ax=ax_right, kde=True, element='step', common_norm=False, legend=False)
ax_right.axis('off')

# ══════════════════════════════════════════════════════════════════════
# 4. 底部残差图 (空心圆)
# ══════════════════════════════════════════════════════════════════════
ax_res.scatter(x_train, train_df['Residual'].values,
               facecolors='none', edgecolors=TRAIN_EDGE, linewidths=0.8,
               alpha=1.0, marker='o', s=30, zorder=3)
ax_res.scatter(x_test, test_df['Residual'].values,
               facecolors='none', edgecolors=TEST_EDGE, linewidths=0.8,
               alpha=0.55, marker='o', s=30, zorder=4)
ax_res.axhline(0, color='gray', linestyle='--', lw=1.5)
ax_res.set_xlabel('Experimental CO$_2$ Uptake (mg/g)', fontsize=14, fontweight='bold')
ax_res.set_ylabel('Residuals (mg/g)', fontsize=12, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════════════════════════
FIGURES.mkdir(parents=True, exist_ok=True)
output_path = FIGURES / f"Figure_4_{MODEL_NAME}_Marginal.png"
fig.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
print(f"已保存: {output_path}")
plt.close(fig)
