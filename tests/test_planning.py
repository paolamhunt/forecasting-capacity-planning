import pandas as pd

from fcp.planning import PlanningConfig, recommend_capacity


def test_recommend_capacity_increases_with_higher_service_level():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")
    forecast = pd.Series([100, 110, 120, 130, 140, 150, 160, 100, 110, 120, 130, 140, 150, 160], index=idx)

    cfg_low = PlanningConfig(
        service_level=0.7,
        units_per_capacity=20.0,
        over_capacity_cost=1.0,
        under_capacity_cost=3.0,
    )
    cfg_high = PlanningConfig(
        service_level=0.9,
        units_per_capacity=20.0,
        over_capacity_cost=1.0,
        under_capacity_cost=3.0,
    )

    cap_low = int(recommend_capacity(forecast, cfg_low)["recommended_capacity"].iloc[0])
    cap_high = int(recommend_capacity(forecast, cfg_high)["recommended_capacity"].iloc[0])

    assert cap_high >= cap_low
