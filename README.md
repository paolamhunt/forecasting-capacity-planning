# Forecasting for Capacity Planning

A forecasting and decision-support system that translates demand forecasts into capacity recommendations under service-level and cost constraints.

The project evaluates forecasting models not only on predictive accuracy, but also on the quality and cost of the downstream operational decisions they produce.

## What This Project Does

- Generates demand forecasts using baseline and regression-based approaches
- Evaluates models with rolling-origin backtesting
- Measures forecast quality using MAE and sMAPE
- Translates forecast distributions into capacity recommendations
- Evaluates under-capacity and over-capacity costs
- Compares decision quality across service-level targets
- Produces reproducible results through scripts, tests, and CI

## Why It Exists

Forecast accuracy alone is not enough in operational planning.

A model can produce a lower error while still leading to worse staffing, inventory, or infrastructure decisions. This project treats forecasting as one component of a larger decision system and evaluates models based on both prediction quality and downstream planning cost.
