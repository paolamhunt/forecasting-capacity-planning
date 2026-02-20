from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlanningConfig:
    """
    Convert demand forecast into capacity recommendations.

    Interpretation:
    - demand is "units of work per day" (tickets/orders/etc.)
    - one unit of capacity handles `units_per_capacity` work per day
    - we choose capacity based on a service level by taking a high quantile of demand
      (simple uncertainty proxy)
    """
    service_level: float  # e.g., 0.90 means plan for p90 demand
    units_per_capacity: float  # productivity per capacity unit per day
    over_capacity_cost: float
    under_capacity_cost: float


def recommend_capacity(
    demand_forecast: pd.Series,
    cfg: PlanningConfig,
) -> pd.DataFrame:
    """
    Recommend a single capacity value for the forecast horizon, plus an optional
    per-day capacity requirement series.

    Returns a DataFrame with:
    - demand_pXX
    - recommended_capacity
    """
    if not 0.5 <= cfg.service_level < 1.0:
        raise ValueError("service_level should be in [0.5, 1.0).")
    if cfg.units_per_capacity <= 0:
        raise ValueError("units_per_capacity must be > 0.")

    demand_forecast = demand_forecast.astype(float)

    q = float(cfg.service_level)
    demand_quantile = float(np.quantile(demand_forecast.values, q))

    recommended_capacity = float(np.ceil(demand_quantile / cfg.units_per_capacity))

    return pd.DataFrame(
        {
            f"demand_p{int(q * 100)}": [round(demand_quantile, 3)],
            "units_per_capacity": [cfg.units_per_capacity],
            "recommended_capacity": [int(recommended_capacity)],
            "over_capacity_cost": [cfg.over_capacity_cost],
            "under_capacity_cost": [cfg.under_capacity_cost],
        }
    )


def evaluate_capacity_cost(
    actual_demand: pd.Series,
    capacity: int,
    cfg: PlanningConfig,
) -> float:
    """
    Simple cost model:
    - If capacity is higher than demand (converted), pay over_capacity_cost per unused unit
    - If capacity is lower, pay under_capacity_cost per missing unit
    """
    if capacity < 0:
        raise ValueError("capacity must be >= 0")

    required_capacity = np.ceil(actual_demand.astype(float).values / cfg.units_per_capacity)

    over = np.maximum(capacity - required_capacity, 0.0)
    under = np.maximum(required_capacity - capacity, 0.0)

    cost = cfg.over_capacity_cost * over.sum() + cfg.under_capacity_cost * under.sum()
    return float(cost)
