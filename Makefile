PY ?= .venv/bin/python
JOBS ?= -1

.PHONY: setup test validate smoke sweep sweep-r1 figures clean-smoke

setup:            ## create .venv with the locked dependencies
	python3 -m venv .venv
	$(PY) -m pip install -r requirements-lock.txt
	$(PY) -m pip install -e . --no-deps

test:             ## unit tests, under a minute
	$(PY) -m pytest -q

validate:         ## slow checks: port vs original simulator, exact DP vs 10^6 Monte Carlo episodes
	$(PY) -m pytest -q -m slow

smoke:            ## 2 cells x 3 replicates, then a small headline figure
	$(PY) scripts/sweep.py configs/smoke.yaml --out results/smoke --jobs $(JOBS)
	$(PY) scripts/figures.py --results results/smoke --out figures/smoke

sweep:            ## the full grid: configs/sweep_main.yaml, then configs/sweep_horizon.yaml
	$(PY) scripts/sweep.py configs/sweep_main.yaml --out results/main --jobs $(JOBS)
	$(PY) scripts/sweep.py configs/sweep_horizon.yaml --out results/horizon --jobs $(JOBS)

sweep-r1:         ## regime R1 only
	$(PY) scripts/sweep.py configs/sweep_main.yaml --out results/main --jobs $(JOBS) --only-regime R1

figures:          ## every figure and table from results/, plus the exact frontier and the oracle-WIS check
	$(PY) scripts/figures.py --results results/main --horizon results/horizon --out figures
	$(PY) scripts/design_frontier.py --out figures
	$(PY) scripts/oracle_wis.py --out figures

clean-smoke:
	rm -rf results/smoke figures/smoke
