from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import pandas as pd

from fcp.backtest import BacktestConfig, build_seasonal_naive_factory, rolling_origin_backtest
from fcp.io import load_config, load_time_series, parse_data_config
from fcp.ridge_forecaster import RidgeForecaster, RidgeForecasterConfig


def _ensure_docs_results_file() -> Path:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    results_path = docs_dir / "results.md"
    if not results_path.exists():
        results_path.write_text("# Results\n\n", encoding="utf-8")
    return results_path


def _run_model(
    *,
    model_name: str,
    model_factory,
    y: pd.Series,
    freq: str,
    bt_cfg: BacktestConfig,
    service_level: float,
    units_per_capacity: float,
    over_capacity_cost: float,
    under_capacity_cost: float,
) -> tuple[pd.DataFrame, dict]:
    results = rolling_origin_backtest(
        y=y,
        freq=freq,
        cfg=bt_cfg,
        model_factory=model_factory,
        service_level=service_level,
        units_per_capacity=units_per_capacity,
        over_capacity_cost=over_capacity_cost,
        under_capacity_cost=under_capacity_cost,
    )

    fold_rows = [
        {
            "model": model_name,
            "fold": r.fold,
            "cutoff": r.cutoff.date().isoformat(),
            "mae": round(r.mae, 3),
            "sMAPE(%)": round(r.smape, 3),
            "recommended_capacity": r.recommended_capacity,
            "planning_cost": round(r.planning_cost, 3),
        }
        for r in results
    ]
    fold_df = pd.DataFrame(fold_rows)

    summary = {
        "model": model_name,
        "service_level": service_level,
        "avg_capacity": round(mean(r.recommended_capacity for r in results), 2),
        "avg_mae": round(mean(r.mae for r in results), 3),
        "avg_sMAPE(%)": round(mean(r.smape for r in results), 3),
        "avg_planning_cost": round(mean(r.planning_cost for r in results), 3),
    }
    return fold_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling-origin backtesting (multi-model).")
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

    # Planning config
    p_cfg_raw = cfg.get("planning", {})
    main_sl = float(p_cfg_raw["service_level"])
    units_per_capacity = float(p_cfg_raw["units_per_capacity"])
    over_capacity_cost = float(p_cfg_raw["over_capacity_cost"])
    under_capacity_cost = float(p_cfg_raw["under_capacity_cost"])

    # ---------- model factories ----------
    # Seasonal naive
    season_length = int(cfg.get("models", {}).get("seasonal_naive", {}).get("season_length", 7))
    seasonal_naive_factory = build_seasonal_naive_factory(season_length=season_length)

    # Ridge regression forecaster
    ridge_cfg = RidgeForecasterConfig(lags=(1, 7, 14), rolling_windows=(7,), alpha=1.0)

    def ridge_factory():
        return RidgeForecaster(ridge_cfg)

    models = [
        ("seasonal_naive", seasonal_naive_factory),
        ("ridge_lag_features", ridge_factory),
    ]

    # ---------- run ----------
    fold_dfs = []
    summaries = []

    for model_name, factory in models:
        fold_df, summary = _run_model(
            model_name=model_name,
            model_factory=factory,
            y=y,
            freq=data_cfg.freq,
            bt_cfg=bt_cfg,
            service_level=main_sl,
            units_per_capacity=units_per_capacity,
            over_capacity_cost=over_capacity_cost,
            under_capacity_cost=under_capacity_cost,
        )
        fold_dfs.append(fold_df)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values(["avg_planning_cost", "avg_mae"])

    print("\nModel comparison summary (lower planning cost is better)")
    print(summary_df.to_string(index=False))

    all_folds_df = pd.concat(fold_dfs, ignore_index=True)
    print("\nFold-level results")
    print(all_folds_df.to_string(index=False))

    # ---------- write to docs/results.md ----------
    results_path = _ensure_docs_results_file()

    md = []
    md.append("## Model Comparison (Forecast + Planning Cost)\n")
    md.append(f"- Horizon: **{bt_cfg.horizon}**\n")
    md.append(f"- Step: **{bt_cfg.step}**\n")
    md.append(f"- Service level target: **p{int(main_sl * 100)}**\n")
    md.append(f"- Units per capacity: **{units_per_capacity}**\n")
    md.append(f"- Costs: over={over_capacity_cost}, under={under_capacity_cost}\n\n")

    md.append("### Summary\n")
    md.append(summary_df.to_markdown(index=False))
    md.append("\n")

    md.append("### Fold-level details\n")
    md.append(all_folds_df.to_markdown(index=False))
    md.append("\n")

    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nWrote results to: {results_path}")


if __name__ == "__main__":
    main()