import pandas as pd

from fcp.backtest import BacktestConfig, build_seasonal_naive_factory, rolling_origin_backtest


def test_rolling_origin_backtest_runs_and_returns_results():
    # Create a simple daily series with enough points
    idx = pd.date_range("2024-01-01", periods=250, freq="D")
    y = pd.Series(range(250), index=idx)

    cfg = BacktestConfig(horizon=14, step=7, min_train_points=180)
    factory = build_seasonal_naive_factory(season_length=7)

    results = rolling_origin_backtest(
        y=y,
        freq="D",
        cfg=cfg,
        model_factory=factory,
        service_level=0.9,
        units_per_capacity=20.0,
        over_capacity_cost=1.0,
        under_capacity_cost=3.0,
    )


    assert len(results) > 0
    assert all(r.mae >= 0 for r in results)
    assert all(r.smape >= 0 for r in results)
