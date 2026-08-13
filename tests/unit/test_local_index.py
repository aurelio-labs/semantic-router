import numpy as np
import pytest

from semantic_router.index import LocalIndex


@pytest.fixture
def local_index() -> LocalIndex:
    return LocalIndex(
        index=np.array([[1.0, 0.0], [0.0, 1.0]]),
        routes=np.array(["billing", "support"]),
    )


def test_query_returns_no_results_when_route_filter_has_no_matches(
    local_index: LocalIndex,
) -> None:
    scores, routes = local_index.query(
        vector=np.array([1.0, 0.0]),
        route_filter=["missing"],
    )

    assert scores.size == 0
    assert routes == []


@pytest.mark.asyncio
async def test_aquery_returns_no_results_when_route_filter_has_no_matches(
    local_index: LocalIndex,
) -> None:
    scores, routes = await local_index.aquery(
        vector=np.array([1.0, 0.0]),
        route_filter=["missing"],
    )

    assert scores.size == 0
    assert routes == []
