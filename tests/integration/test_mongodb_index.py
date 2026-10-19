"""What MongoDbIndex does beyond the shared index contract, against mongodb-atlas-local."""

import uuid

import numpy as np
import pytest

from semantic_router.config import _build_index, _index_config
from semantic_router.index import MongoDbIndex
from semantic_router.schema import Metric


def name() -> str:
    return f"t{uuid.uuid4().hex[:12]}"


def add(index: MongoDbIndex, rows: list[tuple[str, str, list[float]]]) -> None:
    index.add(
        embeddings=[v for _, _, v in rows],
        routes=[r for r, _, _ in rows],
        utterances=[u for _, u, _ in rows],
    )


ROWS = [
    ("a", "first", [1.0, 0.0, 0.0]),
    ("b", "second", [0.0, 1.0, 0.0]),
]


def test_namespaces_keep_records_apart():
    shared = name()
    one = MongoDbIndex(index_name=shared, namespace="one")
    two = MongoDbIndex(index_name=shared, namespace="two")
    add(one, ROWS[:1])
    add(two, ROWS[1:])

    assert [r.name for r in one.get_routes()] == ["a"]
    assert [r.name for r in two.get_routes()] == ["b"]
    # the nearest record overall is in the other namespace, so it must not come back
    _, names = two.query(vector=np.array([1.0, 0.0, 0.0]), top_k=2)
    assert names == ["b"]

    two.delete("a")
    assert [r.name for r in one.get_routes()] == ["a"]


def test_route_filter_limits_the_search():
    index = MongoDbIndex(index_name=name())
    add(index, ROWS)

    _, names = index.query(vector=np.array([1.0, 0.0, 0.0]), route_filter=["b"])

    assert names == ["b"]


def test_cosine_scores_are_on_the_same_scale_as_the_other_indexes():
    index = MongoDbIndex(index_name=name())
    add(index, ROWS)

    scores, names = index.query(vector=np.array([1.0, 0.0, 0.0]), top_k=2)

    # atlas reports (1 + cos) / 2: 1.0 and 0.5 here, mapped back to cos
    assert names == ["a", "b"]
    assert scores == pytest.approx([1.0, 0.0], abs=1e-6)


def test_a_second_handle_with_other_dimensions_is_refused():
    shared = name()
    add(MongoDbIndex(index_name=shared), ROWS)

    with pytest.raises(ValueError, match="3-dimension"):
        MongoDbIndex(index_name=shared).add(
            embeddings=[[1.0, 0.0, 0.0, 0.0]], routes=["c"], utterances=["third"]
        )


def test_manhattan_is_refused():
    with pytest.raises(ValueError, match="no manhattan similarity"):
        MongoDbIndex(index_name=name(), metric=Metric.MANHATTAN)


def test_a_file_holds_the_index_settings_but_not_the_connection_uri():
    index = MongoDbIndex(index_name=name(), namespace="tenant", sync_timeout=5.0)

    config = _index_config(index)

    dumped = config.model_dump()
    assert dumped["type"] == "mongodb"
    assert dumped["index_name"] == index.index_name
    assert dumped["namespace"] == "tenant"
    assert "connection_uri" not in dumped
    rebuilt = _build_index(config)
    assert isinstance(rebuilt, MongoDbIndex)
    assert (rebuilt.index_name, rebuilt.namespace) == (index.index_name, "tenant")
