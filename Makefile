.PHONY: format lint lint_diff test test_unit test_integration test_live test_cov services services_down

format:
	uv run ruff check . --fix
	uv run ruff format .

PYTHON_FILES=.
lint: PYTHON_FILES=.
lint_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$')

lint lint_diff:
	uv run ruff check .
	uv run ruff format . --check
	uv run mypy $(PYTHON_FILES)

# Pytest defaults: parallel workers (every test uses uniquely named indexes and
# cleans up after itself, see tests/conftest.py) and a per-test timeout so a
# hung service can't stall the run.
PYTEST=uv run pytest -n auto --timeout=120

# Default suite: everything that runs against the local services in
# compose.yaml. No API keys needed. This is what CI runs on every PR.
test:
	$(PYTEST) -m "not live" tests

# Tests that need no services at all.
test_unit:
	$(PYTEST) -m "not live" tests/unit tests/functional

# Tests that hit real index backends (pinecone-local, pgvector, qdrant).
test_integration:
	$(PYTEST) -m "not live" tests/integration

# Tests that call paid third-party APIs (OpenAI, Cohere). Need keys in .env.
test_live:
	$(PYTEST) -m live tests

test_cov:
	$(PYTEST) -m "not live" --cov=semantic_router --cov-report=term-missing tests

# Start / stop the local service containers used by the integration tests.
services:
	docker compose up -d --wait

services_down:
	docker compose down -v
