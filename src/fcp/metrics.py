from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(y_true.values - y_pred.values)))


def smape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-8) -> float:
    """
    Symmetric mean absolute percentage error (in %).
    """
    y_true, y_pred = _align(y_true, y_pred)
    denom = np.maximum(np.abs(y_true.values) + np.abs(y_pred.values), eps)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred.values - y_true.values) / denom))


def _align(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    common_idx = y_true.index.intersection(y_pred.index)
    if len(common_idx) == 0:
        raise ValueError("y_true and y_pred have no overlapping timestamps to compare.")
    return y_true.loc[common_idx], y_pred.loc[common_idx]
