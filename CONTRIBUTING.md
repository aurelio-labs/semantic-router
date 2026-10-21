# Contributing to the Semantic Router

The Aurelio Team welcome and encourage any contributions to the Semantic Router, big or small. Please feel free to contribute to new features, bug fixes, or documentation. We're always eager to hear your suggestions.

There are two release lines, on two branches. `v0` carries the 0.x releases (finishing at 0.2.0) and takes bug fixes only. `main` carries the 1.x rewrite and takes everything else. Open your pull request against the branch that owns the code you are changing.

Please follow these guidelines when making a contribution:
1. **Check for Existing Issues:** Before making any changes, [check here for related issues](https://github.com/aurelio-labs/semantic-router/issues).
2. **Run Your Changes by Us!** If no related issue exists yet, please create one and suggest your changes. Checking in with the team first will allow us to determine if the changes are in scope.
3. **Set Up Development Environment** If the changes are agreed, then you can go ahead and set up a development environment (see [Setting Up Your Development Environment](#setting-up-your-development-environment) below).
4. **Create an Early Draft Pull Request** Once you have commits ready to be shared, initiate a draft Pull Request with an initial version of your implementation and request feedback. It's advisable not to wait until the feature is fully completed.
5. **Ensure that All Pull Request Checks Pass** The checks appear towards the bottom of the Pull Request page on GitHub. They are:
    - **PR title** must be a [conventional commit](https://www.conventionalcommits.org/) subject of at most 60 characters, e.g. `feat: add Foo encoder` or `fix: handle empty utterances`.
    - **Lint**: run `make lint` locally (`make format` fixes most issues automatically).
    - **Tests**: run `make test` locally (see [Running the tests](#running-the-tests)). The same suite runs in CI for every pull request, including ones from forks, and needs no API keys.
    - Add tests for new code in the appropriate directory (see [Test tiers](#test-tiers)). `make test_cov` shows what your change leaves uncovered.

> **Feedback and Discussion:**
While we encourage you to initiate a draft Pull Request early to get feedback on your implementation, we also highly value discussions and questions. If you're unsure about any aspect of your contribution or need clarification on the project's direction, please don't hesitate to use the [Issues section](https://github.com/aurelio-labs/semantic-router/issues) of our repository. Engaging in discussions or asking questions before starting your work can help ensure that your efforts align well with the project's goals and existing work.

# Setting Up Your Development Environment

1. Fork the [repository](https://github.com/aurelio-labs/semantic-router) on GitHub and clone your fork:
    ```
    git clone https://github.com/<your-gh-username>/semantic-router.git
    cd semantic-router/
    ```

2. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

3. Create the virtual environment and install the project with the extras the tests need:
    ```
    uv sync --extra pinecone --extra qdrant --extra postgres --extra fastembed
    ```
    The test and lint tooling (pytest, ruff, mypy) is in the default `dev` dependency group, so every `uv sync` and `uv run` includes it. `uv sync --extra all` also works if you want every optional dependency (this pulls in torch and other large packages).

4. Install [Docker](https://docs.docker.com/get-docker/). The integration tests run against local containers.

# Running the tests

```
make services   # start pinecone-local, pgvector and qdrant (once per session)
make test       # the default suite: ~20 seconds, parallel, no API keys needed
```

`make services_down` stops the containers. Other targets:

| Target | What it runs |
|---|---|
| `make test` | Everything except `live` tests. This is what CI runs on every pull request. |
| `make test_unit` | `tests/unit` and `tests/functional` only. Needs no services. |
| `make test_integration` | `tests/integration` against the local containers. |
| `make test_live` | Tests marked `live`. Needs `OPENAI_API_KEY` / `COHERE_API_KEY` in `.env` (copy `.env.example`). |
| `make test_cov` | Same as `make test`, with a coverage report. |

You can also call pytest directly, e.g. `uv run pytest tests/unit/test_route.py -vv`. If the Pinecone or Postgres tests are selected and the containers aren't running, pytest stops immediately with a message telling you to run `make services`.

## Test tiers

- **Unit** (`tests/unit`, `tests/functional`): no network, external clients mocked.
- **Integration** (`tests/integration`, plus the index-backed tests in `tests/unit/test_router.py` and `tests/unit/test_sync.py`): real index backends via the local containers in `compose.yaml`, embedding with a small real model that runs in-process (`all-MiniLM-L6-v2` via fastembed, ~90MB, downloaded on first run and cached) so these need no API keys. This is where most bugs are caught, so prefer adding tests here.
- **Live** (`@pytest.mark.live`): the same tests parametrised with real encoders (OpenAI, Cohere). Excluded from `make test` and from pull-request CI. They run on every push to `main` or `v0`, nightly on `main`, and on a pull request when a maintainer adds the `run-live-tests` label.

Two rules keep the suite fast and parallel-safe:

1. Every test creates its own uniquely named index / table / collection. Use the `init_index()` helpers in the existing test modules, which do this for you.
2. Don't clean up by hand: `tests/conftest.py` deletes every Pinecone index, Postgres table and Qdrant collection a test creates.

# Continuous integration

- `.github/workflows/ci.yml` runs lint and `make test_cov` on Python 3.10 and 3.13 for every pull request and push to `main` or `v0`. The service containers are declared as GitHub Actions `services`, so the job needs no secrets and works for forks.
- `.github/workflows/live.yml` runs `make test_live` with the repo's API keys. Maintainers can run it against a pull request by adding the `run-live-tests` label (removed automatically when new commits are pushed) or from the *Run workflow* button with a PR number.
- `.github/workflows/docs.yml` publishes the docs on pushes to `main` (as the v1 docs) or `v0` (as the v0 docs) that touch `docs/` or `semantic_router/`.
