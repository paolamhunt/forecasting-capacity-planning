# Forecasting for Capacity Planning

An applied time-series forecasting system that translates demand forecasts into capacity recommendations under service-level constraints.

This project is intentionally structured to demonstrate senior-level applied data science:
- Evaluation rigor
- Uncertainty handling
- Decision translation
- Production-quality project organization

---

## What This Project Demonstrates

- Rolling-origin backtesting
- Baseline vs. advanced forecasting models
- Uncertainty-aware planning (prediction intervals or scenarios)
- Demand → capacity translation logic
- SLA-driven cost tradeoff evaluation
- Modular, testable Python architecture

---

## Project Structure

- `src/fcp/` – core forecasting and planning logic
- `scripts/` – runnable entrypoints
- `configs/` – configuration files
- `docs/` – problem framing and approach documentation
- `tests/` – unit tests
- `.github/workflows/` – CI configuration

---

## Quickstart

After installation:

```bash
make install
make test
make backtest
make plan
