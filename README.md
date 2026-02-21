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
```

## Latest Results (Baseline)

This project evaluates not just forecast accuracy, but the downstream planning cost of capacity decisions.

Run:
```bash
make backtest
```
This command appends results to docs/results.md including:

- Rolling-origin backtest metrics (MAE, sMAPE)
- Recommended capacity per fold
- Planning cost per fold
- Service-level sensitivity sweep (p70/p80/p90/p95)

## Design Decisions

### Why evaluate planning cost instead of only forecast error?

In operational systems, forecast accuracy is not the end goal. 
Decisions made from forecasts drive staffing, cost, and service levels.

This project evaluates models not only on MAE and sMAPE, but on downstream planning cost under different service-level assumptions. This reflects real-world applied decision systems.

### Why include a seasonal naive baseline?

Seasonal naive is intentionally simple and difficult to outperform in strongly seasonal systems. It serves as a robustness baseline.

### Why add a ridge regression model?

Ridge regression with lag features represents a practical, explainable forecasting approach that:
- Captures autocorrelation
- Is computationally lightweight
- Is easy to deploy in production systems

The goal is not model complexity, but decision-quality improvement.

### Why perform service-level sensitivity analysis?

Service-level targets directly influence staffing cost tradeoffs. 
Evaluating p70, p80, p90, p95 reveals how decision cost scales with conservatism in planning.