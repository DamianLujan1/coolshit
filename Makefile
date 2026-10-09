# One-command operations. `make weekly` is the Tuesday-night command.
PY ?= .venv/bin/python

.PHONY: setup test backtest weekly dry-run publish site

setup:            ## create the venv and install pinned deps
	uv sync --locked --extra dev

test:             ## run the test suite (feature pipeline, leakage, grading, pick log)
	$(PY) -m pytest

backtest:         ## 2022 selection, frozen 2023-2025 audit; writes reports/
	$(PY) -m gridiron backtest

weekly:           ## fetch, validate, train, log picks, grade, record, recap, site (local only)
	$(PY) -m gridiron weekly

dry-run:          ## exercise full workflow in new OUTPUT_DIR; optional NOW=ISO_DATETIME
	@test -n "$(OUTPUT_DIR)" || (echo 'Set OUTPUT_DIR to a new isolated directory'; exit 1)
	$(PY) -m gridiron $(if $(NOW),--now "$(NOW)") weekly --offline --dry-run --output-dir "$(OUTPUT_DIR)"

publish:          ## explicitly publish from clean reviewed branch after approved Vercel setup
	@test -n "$(PUBLISH_BRANCH)" || (echo 'Set PUBLISH_BRANCH to the reviewed checked-out upstream branch'; exit 1)
	$(PY) -m gridiron weekly --publish --publish-branch "$(PUBLISH_BRANCH)"

site:             ## regenerate site/ from the pick log
	$(PY) -m gridiron site
