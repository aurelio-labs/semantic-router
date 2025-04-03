format:
	uv run ruff check . --fix

PYTHON_FILES=.
lint: PYTHON_FILES=.
lint_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$')

lint lint_diff:
	uv run ruff check .
	uv run ruff format .
	uv run mypy $(PYTHON_FILES)

test:
	uv run pytest -vv --cov=semantic_router --cov-report=term-missing --cov-report=xml --exitfirst --maxfail=1

test_functional:
	uv run pytest -vv  -s --exitfirst --maxfail=1 tests/functional
test_unit:
	uv run pytest -vv --exitfirst --maxfail=1 tests/unit

test_integration:
	uv run pytest -vv --exitfirst --maxfail=1 tests/integration
