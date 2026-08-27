import numpy as np
import pytest

from semantic_router.index.local import LocalIndex


@pytest.fixture
def index():
    index = LocalIndex()
    index.add(
        embeddings=[[1.0, 0.0]],
        routes=["known"],
        utterances=["hello"],
    )
    return index


def test_query_with_missing_route_filter_returns_no_results(index):
    scores, routes = index.query(
        vector=np.array([1.0, 0.0]),
        route_filter=["missing"],
    )

    assert scores.size == 0
    assert routes == []


@pytest.mark.asyncio
async def test_aquery_with_missing_route_filter_returns_no_results(index):
    scores, routes = await index.aquery(
        vector=np.array([1.0, 0.0]),
        route_filter=["missing"],
    )

    assert scores.size == 0
    assert routes == []
