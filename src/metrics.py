"""Forecast metrics for multi-step (sequence-to-sequence) regression.

y_true, y_pred have shape (N, H): N forecast windows, H future steps.
Every metric is computed per horizon step (column) and for all steps pooled.
"""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def nse(y, p):
    """Nash-Sutcliffe efficiency = 1 - SSE / SST (SST around the observed mean).
    1 = perfect, 0 = no better than predicting the mean, <0 = worse than the mean.
    Identical to sklearn.metrics.r2_score."""
    sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / sst)


def r2(y, p):
    """Coefficient of determination as the squared Pearson correlation between
    observations and predictions. Unlike NSE it ignores bias / scale errors, so
    R2 >= NSE; a large gap means the model tracks the shape but is biased."""
    if np.std(y) == 0 or np.std(p) == 0:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1] ** 2)


METRICS = {"R2": r2, "NSE": nse, "MAE": mae, "RMSE": rmse}


def evaluate(y_true, y_pred):
    """Return {'overall': {...}, 'per_step': {1: {...}, ...}}."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {"overall": {k: f(y_true.ravel(), y_pred.ravel()) for k, f in METRICS.items()},
           "per_step": {}}
    for h in range(y_true.shape[1]):
        out["per_step"][h + 1] = {k: f(y_true[:, h], y_pred[:, h]) for k, f in METRICS.items()}
    return out
