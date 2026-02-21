from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class RidgeForecasterConfig:
    """
    Simple regression forecaster using lag/rolling features + recursive multi-step prediction.

    Notes:
    - This is intentionally lightweight and explainable.
    - Forecasts are produced recursively (each predicted step becomes available as a lag).
    """
    lags: tuple[int, ...] = (1, 7, 14)
    rolling_windows: tuple[int, ...] = (7,)
    alpha: float = 1.0  # Ridge regularization strength


class RidgeForecaster:
    def __init__(self, cfg: RidgeForecasterConfig) -> None:
        if not cfg.lags:
            raise ValueError("lags must not be empty")
        if any(l <= 0 for l in cfg.lags):
            raise ValueError("all lags must be > 0")
        if any(w <= 1 for w in cfg.rolling_windows):
            raise ValueError("rolling windows must be > 1")
        if cfg.alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.cfg = cfg
        self.model = Ridge(alpha=cfg.alpha)

    def fit(self, y: pd.Series) -> "RidgeForecaster":
        y = y.astype(float).copy()
        X, target = self._build_training_matrix(y)

        if len(target) == 0:
            raise ValueError("Not enough history to build training features for RidgeForecaster.")

        self.model.fit(X, target)
        self._y_train = y  # keep history for recursive prediction
        return self

    def predict(self, start: pd.Timestamp, horizon: int, freq: str) -> pd.Series:
        """
        Recursive forecast:
        - for each step in horizon, compute features from the latest available series
          (which includes prior predictions), then predict next value.
        """
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if not hasattr(self, "_y_train"):
            raise RuntimeError("Model must be fit before predicting.")

        history = self._y_train.copy()
        preds = []

        idx = pd.date_range(start=start, periods=horizon, freq=freq)

        for ts in idx:
            x = self._build_single_feature_row(history)
            yhat = float(self.model.predict(x)[0])
            preds.append(yhat)

            # Append prediction to history for next-step features
            history.loc[ts] = yhat

        return pd.Series(preds, index=idx, name="yhat")

    # ---------- feature engineering helpers ----------

    def _build_training_matrix(self, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """
        Build X, y_target aligned so each row predicts y[t] from features at t-1 (lags/rollups).
        """
        df = pd.DataFrame({"y": y})

        # Lag features
        for lag in self.cfg.lags:
            df[f"lag_{lag}"] = df["y"].shift(lag)

        # Rolling mean features (based on past values)
        for w in self.cfg.rolling_windows:
            df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(window=w).mean()

        df = df.dropna()

        feature_cols = [c for c in df.columns if c != "y"]
        X = df[feature_cols].to_numpy(dtype=float)
        target = df["y"].to_numpy(dtype=float)
        return X, target

    def _build_single_feature_row(self, history: pd.Series) -> np.ndarray:
        """
        Build a single-row feature vector from the most recent history.
        """
        vals = []

        # lags
        for lag in self.cfg.lags:
            if len(history) < lag:
                raise ValueError("Not enough history to compute required lag features.")
            vals.append(float(history.iloc[-lag]))

        # rolling means (use last w values, excluding 'current' step)
        for w in self.cfg.rolling_windows:
            if len(history) < w:
                raise ValueError("Not enough history to compute required rolling features.")
            vals.append(float(history.iloc[-w:].mean()))

        return np.array([vals], dtype=float)