# BrickAgain — the commands, in one place.
#
#     make help
#
# Targets that need the dataset say so and stop. They do not run part-way and
# leave you guessing: this repository deliberately ships no data, and a target
# that quietly did nothing would be worse than one that refuses.

PY      ?= ./.venv/bin/python
PIP     ?= ./.venv/bin/pip
PYTEST  ?= $(PY) -m pytest
DATA    ?= data/processed

.DEFAULT_GOAL := help
.PHONY: help setup lint test smoke figures eda features data train eval serve clean

help:  ## List the targets
	@grep -hE '^[a-z][a-zA-Z0-9_-]*:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Create .venv and install dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "done. vision extras: $(PIP) install -r requirements-vision.txt"

lint:  ## ruff over the source, scripts and tests
	$(PY) -m ruff check src scripts tests

test:  ## The whole suite
	$(PYTEST) tests/ -q

smoke:  ## The fast subset: config, splits, leakage, re-tiling
	$(PYTEST) tests/test_config.py tests/test_no_leakage.py tests/test_retile.py -q

figures:  ## Regenerate reports/figures/ from data/reports/
	$(PY) scripts/73_figures.py

eda:  ## Corpus statistics — needs the dataset
	@$(MAKE) --no-print-directory _needs-data WHAT="exploratory analysis"
	$(PY) scripts/01_eda.py

features:  ## Splits and the counterfactual set — needs the dataset
	@$(MAKE) --no-print-directory _needs-data WHAT="feature building"
	$(PY) scripts/03_build_splits.py
	$(PY) scripts/04_build_counterfactual.py

data:  ## How to obtain the dataset (it is not distributed here)
	@echo "This repository ships no data, by design."
	@echo
	@echo "  archived 8-brick track : AvaLovelace/StableText2Brick (Hugging Face)"
	@echo "  vision track           : photographs are not redistributable"
	@echo "  BrickNet real parts    : not part of this public snapshot"
	@echo
	@echo "See README.md for what each stage expects under $(DATA)/."

train:  ## Fit the vision classifier — needs the image set
	@$(MAKE) --no-print-directory _needs-data WHAT="training"
	$(PY) scripts/32_vision_train.py

eval:  ## Evaluate the vision classifier — needs the image set
	@$(MAKE) --no-print-directory _needs-data WHAT="evaluation"
	$(PY) scripts/33_vision_eval.py

serve:  ## Serve the interface on loopback
	$(PY) scripts/29_ui.py

clean:  ## Remove caches and generated figures
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache reports/figures/*.png

_needs-data:
	@test -d "$(DATA)" || { \
	  echo "refusing $(WHAT): $(DATA)/ is not here."; \
	  echo "run 'make data' to see where it comes from."; \
	  exit 1; }
