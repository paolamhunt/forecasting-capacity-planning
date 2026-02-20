VENV := .venv
PY := $(VENV)/bin/python

.PHONY: venv install test lint backtest plan

venv:
	python3 -m venv $(VENV)

install: venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

backtest:
	$(PY) scripts/backtest.py --config configs/default.yaml

plan:
	$(PY) scripts/plan_capacity.py --config configs/default.yaml
