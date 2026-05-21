"""步骤5b: 评估指标计算 —— R², RMSE, MAE, MAPE, 尾部RMSE"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算标准回归指标。"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return {"R²": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def compute_tail_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算低吸附区(<q10)和高吸附区(>q90)的RMSE。"""
    q10 = np.quantile(y_true, 0.10)
    q90 = np.quantile(y_true, 0.90)
    low_mask = y_true <= q10
    high_mask = y_true >= q90
    tail_low = (
        np.sqrt(mean_squared_error(y_true[low_mask], y_pred[low_mask]))
        if low_mask.sum() > 1
        else np.nan
    )
    tail_high = (
        np.sqrt(mean_squared_error(y_true[high_mask], y_pred[high_mask]))
        if high_mask.sum() > 1
        else np.nan
    )
    return {"tail_rmse_q10": tail_low, "tail_rmse_q90": tail_high}


def aggregate_fold_results(fold_metrics: list[dict]) -> dict:
    """汇总多折指标：mean ± std。"""
    agg = {}
    keys = ["R²", "RMSE", "MAE", "MAPE", "tail_rmse_q10", "tail_rmse_q90"]
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m and not np.isnan(m[k])]
        if vals:
            agg[f"{k}_mean"] = np.mean(vals)
            agg[f"{k}_std"] = np.std(vals)
    return agg
