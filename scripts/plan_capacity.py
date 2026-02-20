from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fcp.io import load_config, load_time_series, parse_data_config
from fcp.models import SeasonalNaiveConfig, SeasonalNaiveForecaster
from fcp.planning import PlanningConfig, recommend_capacity


def _ensure_docs_results_file() -> Path:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    results_path = docs_dir / "results.md"
    if not results_path.exists():
        results_path.write_text("# Results\n\n", encoding="utf-8")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a capacity recommendation from a forecast.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load series
    data_cfg = parse_data_config(cfg)
    y = load_time_series(data_cfg)

    # Forecast horizon from config
    bt_cfg = cfg.get("backtest", {})
    horizon = int(bt_cfg.get("horizon", 14))

    # Build baseline forecaster (seasonal naive)
    season_length = int(cfg.get("models", {}).get("seasonal_naive", {}).get("season_length", 7))
    model = SeasonalNaiveForecaster(SeasonalNaiveConfig(season_length=season_length)).fit(y)

    start = y.index.max() + pd.tseries.frequencies.to_offset(data_cfg.freq)
    forecast = model.predict(start=start, horizon=horizon, freq=data_cfg.freq)

    # Planning config
    p_cfg_raw = cfg.get("planning", {})
    p_cfg = PlanningConfig(
        service_level=float(p_cfg_raw["service_level"]),
        units_per_capacity=float(p_cfg_raw["units_per_capacity"]),
        over_capacity_cost=float(p_cfg_raw["over_capacity_cost"]),
        under_capacity_cost=float(p_cfg_raw["under_capacity_cost"]),
    )

    rec = recommend_capacity(forecast, p_cfg)

    print("\nForecast horizon demand (baseline):")
    print(forecast.to_string())
    print("\nCapacity recommendation:")
    print(rec.to_string(index=False))

    # Append to docs/results.md
    results_path = _ensure_docs_results_file()
    md = []
    md.append("## Capacity Recommendation (from latest horizon forecast)\n")
    md.append(f"- Model: **Seasonal Naive** (season_length={season_length})\n")
    md.append(f"- Horizon: **{horizon}** {data_cfg.freq}-steps\n")
    md.append(f"- Service level target: **p{int(p_cfg.service_level * 100)}**\n")
    md.append("\n")
    md.append(rec.to_markdown(index=False))
    md.append("\n")

    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nWrote recommendation to: {results_path}")


if __name__ == "__main__":
    main()
