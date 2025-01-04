format:
	poetry run black --target-version py313 -l 88 .
	poetry run ruff --select I --fix .

PYTHON_FILES=.
lint: PYTHON_FILES=.
lint_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$')

lint lint_diff:
	poetry run black --target-version py313 -l 88 $(PYTHON_FILES) --check
	poetry run ruff check .
	poetry run mypy $(PYTHON_FILES)

test:
	poetry run pytest -vv --cov=semantic_router --cov-report=term-missing --cov-report=xml --exitfirst --maxfail=1 

test_functional:
	poetry run pytest -vv --exitfirst --maxfail=1 tests/functional
test_unit:
	poetry run pytest -vv --exitfirst --maxfail=1 tests/unit
test_integration:
	poetry run pytest -vv --exitfirst --maxfail=1 tests/integration
