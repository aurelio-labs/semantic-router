import importlib
import os

import pytest

from semantic_router.encoders.openai import OpenAIEncoder
from semantic_router.index.local import LocalIndex
from semantic_router.index.pinecone import PineconeIndex
from semantic_router.index.qdrant import QdrantIndex
from semantic_router.layer import RouteLayer
from semantic_router.route import Route


def mock_encoder_call(utterances):
    # Define a mapping of utterances to return values
    mock_responses = {
        "Hello": [0.1, 0.2, 0.3],
        "Hi": [0.4, 0.5, 0.6],
        "Goodbye": [0.7, 0.8, 0.9],
        "Bye": [1.0, 1.1, 1.2],
        "Au revoir": [1.3, 1.4, 1.5],
    }
    return [mock_responses.get(u, [0.0, 0.0, 0.0]) for u in utterances]


@pytest.fixture
def openai_encoder(mocker):
    mocker.patch.object(OpenAIEncoder, "__call__", side_effect=mock_encoder_call)
    return OpenAIEncoder(name="test-openai-encoder", openai_api_key="test_api_key")


@pytest.fixture
def routes():
    return [
        Route(name="Route 1", utterances=["Hello", "Hi"]),
        Route(name="Route 2", utterances=["Goodbye", "Bye", "Au revoir"]),
    ]


def get_test_indexes():
    indexes = [LocalIndex]

    if importlib.util.find_spec("qdrant_client") is not None:
        indexes.append(QdrantIndex)
    return indexes


@pytest.mark.parametrize("index_cls", get_test_indexes())
class TestRouteLayerIntegration:
    def test_query_filter_pinecone(self, openai_encoder, routes, index_cls):
        pinecone_api_key = os.environ["PINECONE_API_KEY"]
        pineconeindex = PineconeIndex(api_key=pinecone_api_key)
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=pineconeindex
        )
        query_result = route_layer(text="Hello", route_filter=["Route 1"]).name

        try:
            route_layer(text="Hello", route_filter=["Route 8"]).name
        except ValueError:
            assert True

        assert query_result in ["Route 1"]
