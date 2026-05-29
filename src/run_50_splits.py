"""50次随机数据集分割评估 (Repeated Random Sub-sampling)

对7个非线性模型执行50次独立80/20分层随机划分,
记录每次的Train/Test R², RMSE, MAE,
汇总为均值±标准差表格,并绘制带误差棒的出版级柱状图。
"""

import sys
import io
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import ROOT, TARGET, TABLES, FIGURES, SEED
from src.data_loader import load_and_clean
from src.models import get_all_models

warnings.filterwarnings("ignore")

# ── 配置 ─────────────────────────────────────────────────────────────────
N_SPLITS = 50
TEST_SIZE = 0.20
SEEDS = list(range(N_SPLITS))  # 0, 1, 2, ..., 49

NONLINEAR_MODELS = ["TabPFN", "GBDT", "RF", "XGBoost", "LightGBM", "GPR", "SVR"]
METRICS = ["R2", "RMSE", "MAE"]

# 学术柔和配色: Train=浅湖蓝, Test=暖珊瑚
COLOR_TRAIN = "#7ec8c8"
COLOR_TEST = "#e8a090"


def _compute_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def main():
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 60)
    print(f"50次随机分割评估 — {N_SPLITS} splits × 7 models")
    print("=" * 60)

    # ── 1. 加载数据 ──────────────────────────────────────────────────────
    print("\n[1/4] 加载数据...")
    df = load_and_clean()
    y_all = df[TARGET]
    X_all = df.drop(columns=[TARGET])
    print(f"  样本: {len(df)} 行, 特征: {X_all.shape[1]} 列")

    # ── 2. 加载已保存的最优参数 ──────────────────────────────────────────
    print("\n[2/4] 加载嵌套CV最优参数...")
    cv_path = TABLES / "cv_results.json"
    best_params_map = {}
    if cv_path.exists():
        with open(cv_path, "r", encoding="utf-8") as f:
            cv_results = json.load(f)
        for name in NONLINEAR_MODELS:
            if name in cv_results and "best_params_per_fold" in cv_results[name]:
                fold_params = cv_results[name]["best_params_per_fold"]
                if fold_params and len(fold_params) > 0 and fold_params[0]:
                    best_params_map[name] = fold_params[0]
                    print(f"  {name}: 已加载最优参数")
                else:
                    best_params_map[name] = {}
            else:
                best_params_map[name] = {}
    else:
        print("  ⚠ cv_results.json 未找到，将使用默认参数")
        for name in NONLINEAR_MODELS:
            best_params_map[name] = {}

    # ── 3. 获取模型定义 ──────────────────────────────────────────────────
    all_models = get_all_models()
    model_defs = {}
    for name, pipeline, pipe_label, param_fn in all_models:
        if name in NONLINEAR_MODELS:
            model_defs[name] = (pipeline, pipe_label)

    # ── 4. 50次循环 ──────────────────────────────────────────────────────
    print(f"\n[3/4] 运行 {N_SPLITS} 次随机分割...")

    # 存储结构: results[model_name][set_name][metric] = list of N_SPLITS values
    results: dict = {name: {"train": {m: [] for m in METRICS},
                            "test":  {m: [] for m in METRICS}}
                     for name in NONLINEAR_MODELS}

    for i, seed in enumerate(SEEDS):
        # 分层划分
        y_vals = y_all.values
        q_edges = np.linspace(0, 1, 6)[1:-1]
        thresholds = np.quantile(y_vals, q_edges)
        bins = np.digitize(y_vals, thresholds)

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=bins,
        )

        for name in NONLINEAR_MODELS:
            pipeline, pipe_label = model_defs[name]
            pipe = clone(pipeline)
            params = best_params_map.get(name, {})
            if params:
                try:
                    pipe.set_params(**params)
                except (ValueError, KeyError):
                    pass

            # KDE 权重 (XGBoost/LightGBM)
            if name in ("XGBoost", "LightGBM"):
                from src.train import compute_kde_weights
                sw = compute_kde_weights(y_train)
                pipe.fit(X_train, y_train, model__sample_weight=sw)
            else:
                pipe.fit(X_train, y_train)

            # 预测
            tr_pred = pipe.predict(X_train)
            if tr_pred.ndim == 2 and tr_pred.shape[1] == 1:
                tr_pred = tr_pred.ravel()
            te_pred = pipe.predict(X_test)
            if te_pred.ndim == 2 and te_pred.shape[1] == 1:
                te_pred = te_pred.ravel()

            tr_m = _compute_metrics(y_train, tr_pred)
            te_m = _compute_metrics(y_test, te_pred)

            for m in METRICS:
                results[name]["train"][m].append(tr_m[m])
                results[name]["test"][m].append(te_m[m])

        if (i + 1) % 10 == 0:
            print(f"  已完成 {i + 1}/{N_SPLITS} 次分割...")

    print(f"  全部 {N_SPLITS} 次分割完成。")

    # ── 4b. 导出原始逐次结果 CSV ────────────────────────────────────────
    raw_rows = []
    for name in NONLINEAR_MODELS:
        for subset in ["train", "test"]:
            dataset_label = "Train" if subset == "train" else "Test"
            for split_i in range(N_SPLITS):
                for metric in METRICS:
                    raw_rows.append({
                        "Model": name,
                        "Dataset": dataset_label,
                        "Metric": metric,
                        "Value": results[name][subset][metric][split_i],
                        "Split": split_i,
                    })
    df_raw = pd.DataFrame(raw_rows)
    raw_csv_path = TABLES / "50_splits_raw_results.csv"
    df_raw.to_csv(raw_csv_path, index=False, float_format="%.6f")
    print(f"  原始逐次结果已保存: {raw_csv_path} ({len(raw_rows)} 行)")

    # ── 5. 汇总统计 ──────────────────────────────────────────────────────
    print("\n[4/4] 汇总统计并导出...")

    rows = []
    for name in NONLINEAR_MODELS:
        row = {"Model": name}
        for subset in ["train", "test"]:
            for metric in METRICS:
                vals = results[name][subset][metric]
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                col_mean = f"{subset.capitalize()}_{metric}_mean"
                col_std = f"{subset.capitalize()}_{metric}_std"
                row[col_mean] = mean_val
                row[col_std] = std_val
        rows.append(row)

    df_summary = pd.DataFrame(rows)

    # 按 Test_R2_mean 降序排列
    df_summary = df_summary.sort_values("Test_R2_mean", ascending=False).reset_index(drop=True)

    # 导出 CSV
    csv_path = TABLES / "50_splits_summary_table.csv"
    df_summary.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV 已保存: {csv_path}")

    # 打印终端表格
    print(f"\n{'=' * 110}")
    print("50次随机分割汇总表 (均值 ± 标准差)")
    print(f"{'=' * 110}")
    header = (f"{'Model':<12} {'Train R²':>18} {'Test R²':>18} "
              f"{'Train RMSE':>18} {'Test RMSE':>18} {'Train MAE':>18} {'Test MAE':>18}")
    print(header)
    print("-" * 110)
    for _, row in df_summary.iterrows():
        print(f"{row['Model']:<12} "
              f"{row['Train_R2_mean']:>8.4f}±{row['Train_R2_std']:.4f} "
              f"{row['Test_R2_mean']:>8.4f}±{row['Test_R2_std']:.4f} "
              f"{row['Train_RMSE_mean']:>8.2f}±{row['Train_RMSE_std']:.2f} "
              f"{row['Test_RMSE_mean']:>8.2f}±{row['Test_RMSE_std']:.2f} "
              f"{row['Train_MAE_mean']:>8.2f}±{row['Train_MAE_std']:.2f} "
              f"{row['Test_MAE_mean']:>8.2f}±{row['Test_MAE_std']:.2f}")
    print(f"{'=' * 110}")

    # ── 6. 绘制同轴箱线图（Train/Test 重叠在同一 x 位置）──────────────
    print("\n绘制出版级同轴箱线图...")

    plot_order = df_summary["Model"].tolist()
    n_models = len(plot_order)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    box_w = 0.42
    inner_w = 0.22
    x_pos = np.arange(n_models)

    # 统一的线宽 —— Train 和 Test 完全一致
    LW = 1.5
    MW = 2.0

    box_train = dict(color=COLOR_TRAIN, linewidth=LW)
    whisker_train = dict(color=COLOR_TRAIN, linewidth=LW)
    cap_train = dict(color=COLOR_TRAIN, linewidth=LW)
    median_train = dict(color="#2d6a6a", linewidth=MW)

    box_test = dict(color=COLOR_TEST, linewidth=LW)
    whisker_test = dict(color=COLOR_TEST, linewidth=LW)
    cap_test = dict(color=COLOR_TEST, linewidth=LW)
    median_test = dict(color="#7a2a1a", linewidth=MW)

    flier_none = dict(marker="", markersize=0)

    # 收集所有模型的 Test R² 用于动态 Y 轴范围
    all_test_r2 = [v for name in plot_order for v in results[name]["test"]["R2"]]

    for ax_i, (metric, ylabel) in enumerate(
        [("R2", "R²"), ("RMSE", "RMSE (mg/g)"), ("MAE", "MAE (mg/g)")]
    ):
        ax = axes[ax_i]

        for i, model_name in enumerate(plot_order):
            train_data = results[model_name]["train"][metric]
            test_data = results[model_name]["test"][metric]

            # Train 箱 (宽, 半透明, 后层)
            ax.boxplot(
                train_data, positions=[x_pos[i]], widths=box_w,
                patch_artist=True, manage_ticks=False,
                boxprops=box_train, whiskerprops=whisker_train,
                capprops=cap_train, medianprops=median_train,
                flierprops=flier_none,
            )
            for patch in ax.patches[-1:]:
                patch.set_facecolor("#c5e8e8")
                patch.set_alpha(0.65)
                patch.set_zorder(2)

            # Test 箱 (窄, 不透明, 前层)
            ax.boxplot(
                test_data, positions=[x_pos[i]], widths=inner_w,
                patch_artist=True, manage_ticks=False,
                boxprops=box_test, whiskerprops=whisker_test,
                capprops=cap_test, medianprops=median_test,
                flierprops=flier_none,
            )
            for patch in ax.patches[-1:]:
                patch.set_facecolor("#f0c4b8")
                patch.set_alpha(0.90)
                patch.set_zorder(4)

        # 学术美化
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_order, fontsize=11, rotation=15,
                           ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

        # R²: 动态下限，上限固定 1.0
        if metric == "R2":
            y_min = max(0, min(all_test_r2) - 0.05)
            ax.set_ylim(bottom=y_min, top=1.0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c5e8e8", edgecolor=COLOR_TRAIN, linewidth=LW,
              alpha=0.65, label="Train"),
        Patch(facecolor="#f0c4b8", edgecolor=COLOR_TEST, linewidth=LW,
              alpha=0.90, label="Test"),
    ]
    fig.legend(
        handles=legend_elements, fontsize=12, frameon=True,
        edgecolor="#cccccc", facecolor="white",
        loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = FIGURES / "model_performance_bar_chart.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  同轴箱线图已保存: {fig_path}")

    print(f"\n{'=' * 60}")
    print("全部完成。")
    print(f"  CSV: {csv_path}")
    print(f"  PNG: {fig_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
