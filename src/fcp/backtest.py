from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

from fcp.metrics import mae, smape
from fcp.models import SeasonalNaiveConfig, SeasonalNaiveForecaster
from fcp.planning import PlanningConfig, evaluate_capacity_cost, recommend_capacity

class Forecaster(Protocol):
    def fit(self, y: pd.Series) -> "Forecaster": ...
    def predict(self, start: pd.Timestamp, horizon: int, freq: str) -> pd.Series: ...

@dataclass(frozen=True)
class BacktestConfig:
    horizon: int
    step: int
    min_train_points: int


@dataclass(frozen=True)
class BacktestResult:
    fold: int
    cutoff: pd.Timestamp
    horizon: int
    mae: float
    smape: float
    recommended_capacity: int
    planning_cost: float



def rolling_origin_backtest(
    y: pd.Series,
    freq: str,
    cfg: BacktestConfig,
    model_factory: Callable[[], Forecaster],
    service_level: float,
    units_per_capacity: float,
    over_capacity_cost: float,
    under_capacity_cost: float,
) -> list[BacktestResult]:

    """
    Rolling-origin evaluation.

    For each fold:
    - Train on y[:cutoff]
    - Predict horizon steps starting at cutoff + freq
    - Compare to actuals
    """
    if cfg.horizon <= 0 or cfg.step <= 0:
        raise ValueError("horizon and step must be > 0")
    required = cfg.min_train_points + cfg.horizon
    available = len(y)
    if available < required:
        raise ValueError(
            f"Not enough data for backtest. Required >= {required} points "
            f"(min_train_points={cfg.min_train_points} + horizon={cfg.horizon}), "
            f"but got {available}."
        )


    results: list[BacktestResult] = []

    # Determine cutoff points in integer index space
    start_cutoff_idx = cfg.min_train_points - 1
    last_possible_cutoff_idx = len(y) - cfg.horizon - 1

    fold = 0
    cutoff_idx = start_cutoff_idx

    while cutoff_idx <= last_possible_cutoff_idx:
        fold += 1
        cutoff_ts = y.index[cutoff_idx]

        train = y.iloc[: cutoff_idx + 1]
        test_start = y.index[cutoff_idx] + pd.tseries.frequencies.to_offset(freq)
        test_idx = pd.date_range(start=test_start, periods=cfg.horizon, freq=freq)
        test = y.reindex(test_idx)

        # Fit & predict
        model = model_factory()
        model.fit(train)
        preds = model.predict(start=test_start, horizon=cfg.horizon, freq=freq)

        # Metrics (handle potential missing)
        if test.isna().any():
            # In real settings, we'd handle missing more carefully. For now: drop missing.
            test_clean = test.dropna()
            preds_clean = preds.reindex(test_clean.index)
        else:
            test_clean = test
            preds_clean = preds

        # Planning evaluation: forecast -> capacity -> cost vs actual demand
        p_cfg = PlanningConfig(
            service_level=service_level,
            units_per_capacity=units_per_capacity,
            over_capacity_cost=over_capacity_cost,
            under_capacity_cost=under_capacity_cost,
        )
        rec = recommend_capacity(preds_clean, p_cfg)
        recommended_capacity = int(rec["recommended_capacity"].iloc[0])
        planning_cost = evaluate_capacity_cost(test_clean, recommended_capacity, p_cfg)


        results.append(
            BacktestResult(
                fold=fold,
                cutoff=cutoff_ts,
                horizon=cfg.horizon,
                mae=mae(test_clean, preds_clean),
                smape=smape(test_clean, preds_clean),
                recommended_capacity=recommended_capacity,
                planning_cost=planning_cost,

            )
        )

        cutoff_idx += cfg.step

    return results


def build_seasonal_naive_factory(season_length: int) -> Callable[[], SeasonalNaiveForecaster]:
    def _factory() -> SeasonalNaiveForecaster:
        return SeasonalNaiveForecaster(SeasonalNaiveConfig(season_length=season_length))

    return _factory
