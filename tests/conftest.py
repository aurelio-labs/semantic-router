"""Shared pytest configuration.

Test tiers
----------
* default (``make test``): unit tests plus integration tests that run against
  the local service containers in ``compose.yaml`` (pinecone-local, pgvector,
  qdrant). No API keys are needed, so this is what CI runs on every pull
  request, including ones from forks.
* ``live`` (``make test_live``): tests marked ``@pytest.mark.live`` call paid
  third-party APIs (OpenAI, Cohere). They are excluded from the default run.
  Mark a test with the env var it needs, e.g. ``@pytest.mark.live("OPENAI_API_KEY")``,
  and it is skipped automatically when that key is missing. A bare ``live``
  mark requires every key in ``LIVE_KEYS``.
"""

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

# Env var -> tests that need it are skipped when it is unset.
LIVE_KEYS = ("OPENAI_API_KEY", "COHERE_API_KEY")


def _reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pytest_configure(config):
    # Load .env from the project root for local runs (CI sets env explicitly).
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    # Sensible local defaults so `make services && make test` works with no .env.
    os.environ.setdefault("PINECONE_API_KEY", "pclocal")
    os.environ.setdefault("PINECONE_API_BASE_URL", "http://localhost:5080")
    os.environ.setdefault("POSTGRES_HOST", "localhost")
    os.environ.setdefault("POSTGRES_PORT", "5432")
    os.environ.setdefault("POSTGRES_DB", "postgres")
    os.environ.setdefault("POSTGRES_USER", "postgres")
    os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
    # Qdrant falls back to the in-memory client when no server is listening.
    if "QDRANT_URL" not in os.environ and _reachable("localhost", 6333):
        os.environ["QDRANT_URL"] = "http://localhost:6333"


def pytest_collection_modifyitems(config, items):
    ids = [item.nodeid for item in items]

    # Fail fast, with instructions, if a needed local service is down.
    if any("PineconeIndex" in i for i in ids):
        url = urlparse(os.environ["PINECONE_API_BASE_URL"])
        if url.hostname and not _reachable(url.hostname, url.port or 80):
            pytest.exit(
                f"pinecone-local is not reachable at {url.geturl()}. "
                "Start the local services with `make services`, or deselect "
                "these tests with `-k 'not PineconeIndex'`.",
                returncode=2,
            )
    if any("PostgresIndex" in i for i in ids):
        host, port = os.environ["POSTGRES_HOST"], int(os.environ["POSTGRES_PORT"])
        if not _reachable(host, port):
            pytest.exit(
                f"Postgres is not reachable at {host}:{port}. "
                "Start the local services with `make services`, or deselect "
                "these tests with `-k 'not PostgresIndex'`.",
                returncode=2,
            )

    # Skip each live test when a key it needs is missing.
    for item in items:
        live_marks = list(item.iter_markers("live"))
        if not live_marks:
            continue
        needed = {key for mark in live_marks for key in mark.args} or set(LIVE_KEYS)
        missing = sorted(k for k in needed if not os.environ.get(k, "").strip())
        if missing:
            item.add_marker(
                pytest.mark.skip(reason=f"live test: set {', '.join(missing)} to run")
            )


@pytest.fixture(autouse=True)
def _cleanup_pinecone_indexes(monkeypatch):
    """Delete every Pinecone index a test creates once the test finishes.

    pinecone-local binds one data-plane port per index and only frees it when
    the index is deleted, so leaking indexes exhausts the published port range.
    Deleting them also keeps the emulator fast and the cloud account tidy when
    the live tests run against real Pinecone.
    """
    try:
        from semantic_router.index.pinecone import PineconeIndex
    except ImportError:
        yield
        return

    created: list = []
    original_init = PineconeIndex.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(PineconeIndex, "__init__", tracking_init)
    yield
    for index in created:
        try:
            index.client.delete_index(index.index_name)
        except Exception:
            pass  # already deleted by the test, or never created


@pytest.fixture(autouse=True)
def _cleanup_postgres_tables(monkeypatch):
    """Drop every Postgres table a test creates once the test finishes."""
    try:
        from semantic_router.index.postgres import PostgresIndex
    except ImportError:
        yield
        return

    created: list = []
    original_init = PostgresIndex.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(PostgresIndex, "__init__", tracking_init)
    yield
    for index in created:
        try:
            index.delete_index()
        except Exception:
            pass  # already deleted by the test, or never connected


@pytest.fixture(autouse=True)
def _cleanup_qdrant_collections(monkeypatch):
    """Drop every Qdrant collection a test creates once the test finishes."""
    try:
        from semantic_router.index.qdrant import QdrantIndex
    except ImportError:
        yield
        return

    created: list = []
    original_init = QdrantIndex.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(QdrantIndex, "__init__", tracking_init)
    yield
    for index in created:
        try:
            index.client.delete_collection(index.index_name)
        except Exception:
            pass  # already deleted by the test, or never created
