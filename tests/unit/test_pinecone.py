import types
from typing import Any, Dict, List

import numpy as np
import pytest

from semantic_router.index.pinecone import PineconeIndex


class _DummyServerlessSpec:
    def __init__(self, cloud: str, region: str):
        self.cloud = cloud
        self.region = region


def test_grpc_toggle_initializes_grpc_client(monkeypatch):
    fake_pinecone_grpc = types.ModuleType("pinecone.grpc")

    class DummyGRPCClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_pinecone_grpc.PineconeGRPC = DummyGRPCClient  # type: ignore[attr-defined]

    fake_pinecone = types.ModuleType("pinecone")
    fake_pinecone.ServerlessSpec = _DummyServerlessSpec  # type: ignore[attr-defined]

    monkeypatch.setitem(__import__("sys").modules, "pinecone.grpc", fake_pinecone_grpc)
    monkeypatch.setitem(__import__("sys").modules, "pinecone", fake_pinecone)

    idx = PineconeIndex(
        api_key="test-key",
        index_name="idx",
        base_url="https://api.pinecone.io",
        init_async_index=True,
        transport="grpc",
    )

    assert idx.client.__class__.__name__ == "DummyGRPCClient"
    assert idx.client.kwargs["api_key"] == "test-key"


def test_create_index_passes_options(monkeypatch):
    class DummyIndex:
        def describe_index_stats(self) -> Dict[str, Any]:
            return {"dimension": 1536}

    class DummyClient:
        def __init__(self):
            self.created: Dict[str, Any] = {}

        def has_index(self, name: str) -> bool:
            return False

        def create_index(self, **kwargs):
            self.created = kwargs

        def Index(self, name: str):  # noqa: N802
            return DummyIndex()

    monkeypatch.setattr(
        PineconeIndex,
        "ServerlessSpec",
        _DummyServerlessSpec,
        raising=False,
    )

    def _init_client(self, api_key: str):  # noqa: ANN001
        return DummyClient()

    monkeypatch.setattr(PineconeIndex, "_initialize_client", _init_client, raising=False)

    idx = PineconeIndex(
        api_key="test-key",
        index_name="idx",
        dimensions=1536,
        base_url="https://api.pinecone.io",
        deletion_protection="enabled",
        tags={"env": "test"},
    )

    created = idx.client.created  # type: ignore[attr-defined]
    assert created["deletion_protection"] == "enabled"
    assert created["tags"] == {"env": "test"}


def test_query_across_namespaces_merge(monkeypatch):
    idx = PineconeIndex(
        api_key="test-key",
        index_name="idx",
        base_url="https://api.pinecone.io",
        init_async_index=True,
    )

    class DummyIndex:
        def __init__(self, responses: Dict[str, List[Dict[str, Any]]]):
            self._responses = responses

        def query(self, *, vector, top_k, include_metadata, filter, namespace):  # noqa: ANN001
            return {"matches": list(self._responses.get(namespace, []))[:top_k]}

    responses = {
        "ns1": [
            {"id": "a", "score": 0.5, "metadata": {"sr_route": "r1"}},
            {"id": "b", "score": 0.3, "metadata": {"sr_route": "r2"}},
        ],
        "ns2": [
            {"id": "c", "score": 0.9, "metadata": {"sr_route": "r3"}},
        ],
    }
    idx.index = DummyIndex(responses)  # type: ignore[assignment]

    vec = np.array([0.1, 0.2, 0.3])
    merged = idx.query_across_namespaces(vec, namespaces=["ns1", "ns2"], top_k=2)
    assert "matches" in merged
    assert [m["id"] for m in merged["matches"]] == ["c", "a"]

