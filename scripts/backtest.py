import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling-origin backtesting (stub).")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    # Placeholder until we implement real backtesting logic.
    print(f"[stub] backtest would run using config: {args.config}")


if __name__ == "__main__":
    main()
