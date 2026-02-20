import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run capacity planning logic (stub).")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    # Placeholder until we implement real planning logic.
    print(f"[stub] capacity planning would run using config: {args.config}")


if __name__ == "__main__":
    main()
