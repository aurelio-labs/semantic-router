format:
	poetry run black .
	poetry run ruff --select I --fix .

PYTHON_FILES=.
lint: PYTHON_FILES=.
lint_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$')

lint lint_diff:
	poetry run black $(PYTHON_FILES) --check
	poetry run ruff .

test:
	poetry run pytest -vv --cov=semantic_router --cov-report=term-missing --cov-fail-under=100
