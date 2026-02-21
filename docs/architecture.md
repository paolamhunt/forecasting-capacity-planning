# System Architecture

Data → Model → Forecast → Capacity Translation → Cost Evaluation

1. Time series data loaded via config
2. Model trained (seasonal naive or ridge)
3. Rolling-origin backtest generates forecasts
4. Forecast converted into capacity recommendation
5. Planning cost evaluated under service-level assumptions