# One-command operations. `make weekly` is the Tuesday-night command.
PY := .venv/bin/python

.PHONY: setup test backtest weekly site deploy

setup:            ## create the venv and install pinned deps
	uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]"

test:             ## run the test suite (feature pipeline, leakage, grading, pick log)
	$(PY) -m pytest

backtest:         ## walk-forward backtest on 2023-2025; writes reports/ and picks the shipping stage
	$(PY) -m gridiron backtest

weekly:           ## fetch, validate, grade last week, train, log picks, recap, site, commit
	$(PY) -m gridiron weekly

site:             ## regenerate site/ from the pick log
	$(PY) -m gridiron site

deploy:           ## push site/ to Vercel production (needs `npm i -g vercel` and `vercel login`)
	vercel deploy --prod --yes site
