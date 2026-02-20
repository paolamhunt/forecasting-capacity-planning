from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeasonalNaiveConfig:
    season_length: int


class SeasonalNaiveForecaster:
    """
    Seasonal naive forecast:
    y_hat[t] = y[t - season_length]

    For a horizon H, we repeat the last observed season.
    """

    def __init__(self, config: SeasonalNaiveConfig) -> None:
        if config.season_length <= 0:
            raise ValueError("season_length must be > 0")
        self.config = config

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        if len(y) < self.config.season_length:
            raise ValueError("Not enough history to fit seasonal naive model.")
        self._y = y.astype(float)
        return self

    def predict(self, start: pd.Timestamp, horizon: int, freq: str) -> pd.Series:
        """
        Predict for [start, start + horizon).
        start: first timestamp to forecast
        horizon: number of steps
        freq: pandas offset alias (e.g., 'D')
        """
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if not hasattr(self, "_y"):
            raise RuntimeError("Model must be fit before predicting.")

        idx = pd.date_range(start=start, periods=horizon, freq=freq)

        # We forecast by repeating the last season_length values
        last_season = self._y.iloc[-self.config.season_length :].to_numpy()
        reps = int(np.ceil(horizon / self.config.season_length))
        vals = np.tile(last_season, reps)[:horizon]

        return pd.Series(vals, index=idx, name="yhat")
