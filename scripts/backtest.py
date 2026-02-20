from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import pandas as pd

from fcp.backtest import BacktestConfig, build_seasonal_naive_factory, rolling_origin_backtest
from fcp.io import load_config, load_time_series, parse_data_config


def _ensure_docs_results_file() -> Path:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    results_path = docs_dir / "results.md"
    if not results_path.exists():
        results_path.write_text("# Results\n\n", encoding="utf-8")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling-origin backtesting.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load series
    data_cfg = parse_data_config(cfg)
    y = load_time_series(data_cfg)

    # Backtest config
    bt_cfg_raw = cfg.get("backtest", {})
    bt_cfg = BacktestConfig(
        horizon=int(bt_cfg_raw["horizon"]),
        step=int(bt_cfg_raw["step"]),
        min_train_points=int(bt_cfg_raw["min_train_points"]),
    )

    # Model factory (seasonal naive baseline)
    season_length = int(cfg.get("models", {}).get("seasonal_naive", {}).get("season_length", 7))
    model_factory = build_seasonal_naive_factory(season_length=season_length)

    # Planning config (cost + capacity translation)
    p_cfg_raw = cfg.get("planning", {})
    main_sl = float(p_cfg_raw["service_level"])
    units_per_capacity = float(p_cfg_raw["units_per_capacity"])
    over_capacity_cost = float(p_cfg_raw["over_capacity_cost"])
    under_capacity_cost = float(p_cfg_raw["under_capacity_cost"])

    # Service-level sweep (tradeoff curve)
    service_levels = [0.70, 0.80, 0.90, 0.95]
    sweep_summary = []

    for sl in service_levels:
        sl_results = rolling_origin_backtest(
            y=y,
            freq=data_cfg.freq,
            cfg=bt_cfg,
            model_factory=model_factory,
            service_level=float(sl),
            units_per_capacity=units_per_capacity,
            over_capacity_cost=over_capacity_cost,
            under_capacity_cost=under_capacity_cost,
        )

        sweep_summary.append(
            {
                "service_level": sl,
                "avg_capacity": round(mean(r.recommended_capacity for r in sl_results), 2),
                "avg_mae": round(mean(r.mae for r in sl_results), 3),
                "avg_sMAPE(%)": round(mean(r.smape for r in sl_results), 3),
                "avg_planning_cost": round(mean(r.planning_cost for r in sl_results), 3),
            }
        )

    sweep_df = pd.DataFrame(sweep_summary)

    # Main run (fold-level table)
    results = rolling_origin_backtest(
        y=y,
        freq=data_cfg.freq,
        cfg=bt_cfg,
        model_factory=model_factory,
        service_level=main_sl,
        units_per_capacity=units_per_capacity,
        over_capacity_cost=over_capacity_cost,
        under_capacity_cost=under_capacity_cost,
    )

    avg_mae = mean(r.mae for r in results)
    avg_smape = mean(r.smape for r in results)
    avg_planning_cost = mean(r.planning_cost for r in results)

    rows = [
        {
            "fold": r.fold,
            "cutoff": r.cutoff.date().isoformat(),
            "mae": round(r.mae, 3),
            "sMAPE(%)": round(r.smape, 3),
            "recommended_capacity": r.recommended_capacity,
            "planning_cost": round(r.planning_cost, 3),
        }
        for r in results
    ]
    df = pd.DataFrame(rows)

    print("\nService-level sweep summary")
    print(sweep_df.to_string(index=False))

    print("\nBacktest summary (Seasonal Naive)")
    print(df.to_string(index=False))
    print(f"\nAverage MAE: {avg_mae:.3f}")
    print(f"Average sMAPE(%): {avg_smape:.3f}")
    print(f"Average planning cost: {avg_planning_cost:.3f}")

    # Write to docs/results.md
    results_path = _ensure_docs_results_file()

    md = []
    md.append("## Rolling-Origin Backtest – Seasonal Naive Baseline\n")
    md.append(f"- Horizon: **{bt_cfg.horizon}**\n")
    md.append(f"- Step: **{bt_cfg.step}**\n")
    md.append(f"- Season length: **{season_length}**\n")
    md.append(f"- Service level target: **p{int(main_sl * 100)}**\n")
    md.append(f"- Units per capacity: **{units_per_capacity}**\n")
    md.append(f"- Average MAE: **{avg_mae:.3f}**\n")
    md.append(f"- Average sMAPE(%): **{avg_smape:.3f}**\n")
    md.append(f"- Average planning cost: **{avg_planning_cost:.3f}**\n\n")
    md.append(df.to_markdown(index=False))
    md.append("\n")

    md2 = []
    md2.append("## Service-Level Sensitivity (Cost Tradeoff)\n")
    md2.append(
        "How planning cost changes as we increase the service level target "
        "(higher service level typically increases capacity and reduces under-capacity penalties).\n\n"
    )
    md2.append(sweep_df.to_markdown(index=False))
    md2.append("\n")

    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
        f.write("\n".join(md2) + "\n")

    print(f"\nWrote results to: {results_path}")


if __name__ == "__main__":
    main()
    