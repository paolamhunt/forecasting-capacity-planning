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

    data_cfg = parse_data_config(cfg)
    y = load_time_series(data_cfg)

    bt_cfg_raw = cfg.get("backtest", {})
    bt_cfg = BacktestConfig(
        horizon=int(bt_cfg_raw["horizon"]),
        step=int(bt_cfg_raw["step"]),
        min_train_points=int(bt_cfg_raw["min_train_points"]),
    )

    season_length = int(cfg.get("models", {}).get("seasonal_naive", {}).get("season_length", 7))
    model_factory = build_seasonal_naive_factory(season_length=season_length)

    results = rolling_origin_backtest(y=y, freq=data_cfg.freq, cfg=bt_cfg, model_factory=model_factory)

    # Summarize
    avg_mae = mean(r.mae for r in results)
    avg_smape = mean(r.smape for r in results)

    # Pretty table for markdown
    rows = [
        {"fold": r.fold, "cutoff": r.cutoff.date().isoformat(), "mae": round(r.mae, 3), "sMAPE(%)": round(r.smape, 3)}
        for r in results
    ]
    df = pd.DataFrame(rows)

    print("\nBacktest summary (Seasonal Naive)")
    print(df.to_string(index=False))
    print(f"\nAverage MAE: {avg_mae:.3f}")
    print(f"Average sMAPE(%): {avg_smape:.3f}")

    # Write to docs/results.md
    results_path = _ensure_docs_results_file()
    md = []
    md.append("## Rolling-Origin Backtest – Seasonal Naive Baseline\n")
    md.append(f"- Horizon: **{bt_cfg.horizon}**\n")
    md.append(f"- Step: **{bt_cfg.step}**\n")
    md.append(f"- Season length: **{season_length}**\n")
    md.append(f"- Average MAE: **{avg_mae:.3f}**\n")
    md.append(f"- Average sMAPE(%): **{avg_smape:.3f}**\n\n")
    md.append(df.to_markdown(index=False))
    md.append("\n")

    # Append results (keeps history)
    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nWrote results to: {results_path}")


if __name__ == "__main__":
    main()
