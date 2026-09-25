# Foresight — the commands you will actually type.
#
# A Makefile is the cheapest documentation there is: it is the only kind that stops
# working when it goes out of date. Everything below is what the README describes, and
# `make help` is what a new person runs first.
#
# Foresight is built across the book, so some of its commands arrive later than others.
# Those say which chapter brings them rather than failing with a missing file.

.DEFAULT_GOAL := help
.PHONY: help setup preflight data listings verify test lint clean \
        test-all contracts train score serve monitor demo docker

# The project's own Python once `make setup` has made it, so no command depends on the
# environment being switched on; the system's python3 only until then.
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export PYTHONPATH := $(CURDIR)/code

# `make train` trains the model named here; its settings are in
# code/foresight/pipeline/configs/$(MODEL).toml. ARGS passes anything
# else, for example ARGS="--set data.as_of=2024-12-01".
# `make score` and `make serve` take the day from FORESIGHT_ON if it is
# set, and Meridian's records stop at the end of 2025, so the book runs
# FORESIGHT_ON=2025-12-31 make score (or ARGS="--on 2025-12-31").
MODEL ?= renewal
ARGS ?=

help:  ## show this list
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## create the virtualenv and install everything
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	@# On Linux the default PyTorch wheel carries gigabytes of GPU libraries this book
	@# never uses; take the CPU build first. macOS and Windows wheels are CPU-only anyway.
	@if [ "$$(uname -s)" = "Linux" ]; then \
	  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu; fi
	.venv/bin/pip install -r requirements.txt
	@# One line in site-packages puts code/ on Python's path, so `from foresight.config
	@# import ...` works from any folder without setting PYTHONPATH.
	.venv/bin/python -c "import site, pathlib; pathlib.Path(site.getsitepackages()[0], 'zero-to-one-code.pth').write_text(str(pathlib.Path('code').resolve()) + chr(10))"
	@echo "Now: make data, then make preflight."

preflight:  ## check Python, the libraries and the dataset before you start
	$(PY) code/_preflight.py

data:  ## generate the Meridian dataset from its seed, and verify it
	$(PY) code/meridian/generate.py
	$(PY) code/meridian/verify.py
	$(PY) code/meridian/generate_ml.py
	$(PY) code/meridian/verify_ml.py

listings:  ## run every listing whose source changed, and capture its output
	$(PY) code/_runner.py

verify:  ## the code guarantee: nothing stale, and every listing prints the same thing twice
	$(PY) code/_runner.py --check --twice

# The suite has four layers (unit, data, model, service) and a second
# mark, slow, for every test that needs the generated dataset. `make
# test` is what to run after every change and what CI runs on every
# push; `make test-all` is what CI runs nightly.
test:  ## the fast tests, every layer, no dataset needed
	$(PY) -m pytest tests/ -q -m "not slow"

test-all:  ## every test, slow ones too: needs make data and the table
	$(PY) -m pytest tests/ -q

contracts:  ## check the warehouse's feeds against their data contracts
	$(PY) -m foresight.contracts $(ARGS)

lint:  ## the checks CI runs
	$(PY) -m ruff check code/ tests/
	$(PY) code/_runner.py --check

train:  ## train, check, evaluate and register a model: make train MODEL=renewal
	$(PY) -m foresight.pipeline.run $(MODEL) $(ARGS)

score:  ## the monthly list, if one is due, into data/foresight/scores.db
	$(PY) -m foresight.serve.batch $(ARGS)

serve:  ## run the prediction API on :8000
	$(PY) -m uvicorn foresight.serve.api:app --host 127.0.0.1 --port 8000

monitor:  ## check the live model for drift
	@echo "Drift monitoring arrives in Chapter 24."

demo:  ## the ten-minute demonstration: list, forecast, API, monitor, MCP server
	$(PY) -m foresight.demo

docker:  ## build and run the book's environment in a container
	docker compose up --build

clean:  ## remove build output and caches
	rm -rf .pytest_cache/ .ruff_cache/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
