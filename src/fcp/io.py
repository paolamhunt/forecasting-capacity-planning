from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class DataConfig:
    path: str
    date_col: str
    target_col: str
    freq: str


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config into a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")
    return cfg


def parse_data_config(cfg: dict[str, Any]) -> DataConfig:
    """Parse and validate the data section of config."""
    if "data" not in cfg or not isinstance(cfg["data"], dict):
        raise ValueError("Config missing required 'data' section.")

    data = cfg["data"]
    required = ["path", "date_col", "target_col", "freq"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Config data section missing keys: {missing}")

    return DataConfig(
        path=str(data["path"]),
        date_col=str(data["date_col"]),
        target_col=str(data["target_col"]),
        freq=str(data["freq"]),
    )


def load_time_series(data_cfg: DataConfig) -> pd.Series:
    """
    Load a time series from a CSV into a pandas Series with a DatetimeIndex.

    Expected columns:
    - date_col (e.g., 'ds'): parseable to datetime
    - target_col (e.g., 'y'): numeric
    """
    path = Path(data_cfg.path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    if data_cfg.date_col not in df.columns:
        raise ValueError(f"Missing date column '{data_cfg.date_col}' in {path}")
    if data_cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column '{data_cfg.target_col}' in {path}")

    df[data_cfg.date_col] = pd.to_datetime(df[data_cfg.date_col], errors="coerce")
    if df[data_cfg.date_col].isna().any():
        raise ValueError("Some dates could not be parsed. Check the CSV date column.")

    df = df.sort_values(data_cfg.date_col)
    s = pd.Series(df[data_cfg.target_col].astype(float).values, index=df[data_cfg.date_col])

    # Enforce frequency (simple approach): reindex to a complete date range.
    full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq=data_cfg.freq)
    s = s.reindex(full_idx)

    # For this portfolio project, we keep it simple: forward-fill missing values.
    # In real production, we'd be more careful and document the imputation approach.
    s = s.ffill()

    s.name = data_cfg.target_col
    return s
