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
        train score serve monitor demo docker

PY := python3
export PYTHONPATH := $(CURDIR)/code

help:  ## show this list
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## create the virtualenv and install everything
	$(PY) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@# One line in site-packages puts code/ on Python's path, so `from foresight.config
	@# import ...` works from any folder without setting PYTHONPATH.
	.venv/bin/python -c "import site, pathlib; pathlib.Path(site.getsitepackages()[0], 'zero-to-one-code.pth').write_text(str(pathlib.Path('code').resolve()) + chr(10))"
	@echo "Now: make preflight, then make data."

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

test:  ## the test suite
	$(PY) -m pytest tests/ -q

lint:  ## the checks CI runs
	$(PY) -m ruff check code/ tests/
	$(PY) code/_runner.py --check

train:  ## train Foresight's models with one command
	@echo "Foresight's one-command training arrives in Chapter 21."

score:  ## write tonight's scores back to the warehouse
	@echo "Batch scoring arrives in Chapter 22."

serve:  ## run the prediction API on :8000
	@echo "The online API arrives in Chapter 22."

monitor:  ## check the live model for drift
	@echo "Drift monitoring arrives in Chapter 24."

demo:  ## the ten-minute demonstration
	@echo "The ten-minute demonstration arrives in Chapter 27."

docker:  ## build and run the book's environment in a container
	docker compose up --build

clean:  ## remove build output and caches
	rm -rf .pytest_cache/ .ruff_cache/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
