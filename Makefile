format:
	poetry run black --target-version py39 .
	poetry run ruff --select I --fix .

PYTHON_DIFF_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$')

PYTHON_FILES=.
lint: PYTHON_FILES=.
lint_diff: PYTHON_FILES=$(PYTHON_DIFF_FILES)

lint lint_diff:
	@if [ -n "$(PYTHON_FILES)" ]; then \
		poetry run black --target-version py39 $(PYTHON_FILES) --check; \
		poetry run ruff .; \
		poetry run mypy $(PYTHON_FILES); \
	else \
		echo "No Python files to lint."; \
	fi

test:
	poetry run pytest -vv -n 20 --cov=semantic_router --cov-report=term-missing --cov-report=xml --cov-fail-under=80
