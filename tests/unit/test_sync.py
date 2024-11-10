import importlib
import os
from datetime import datetime
import pytest
import time
from typing import Optional
from semantic_router.encoders import BaseEncoder, CohereEncoder, OpenAIEncoder
from semantic_router.index.pinecone import PineconeIndex
from semantic_router.index.qdrant import QdrantIndex
from semantic_router.layer import RouteLayer
from semantic_router.route import Route
from platform import python_version


PINECONE_SLEEP = 12


def mock_encoder_call(utterances):
    # Define a mapping of utterances to return values
    mock_responses = {
        "Hello": [0.1, 0.2, 0.3],
        "Hi": [0.4, 0.5, 0.6],
        "Goodbye": [0.7, 0.8, 0.9],
        "Bye": [1.0, 1.1, 1.2],
        "Au revoir": [1.3, 1.4, 1.5],
        "Asparagus": [-2.0, 1.0, 0.0],
    }
    return [mock_responses.get(u, [0.0, 0.0, 0.0]) for u in utterances]


TEST_ID = (
    f"{python_version().replace('.', '')}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
)


def init_index(
    index_cls,
    dimensions: Optional[int] = None,
    namespace: Optional[str] = "",
    sync: Optional[str] = None,
):
    """We use this function to initialize indexes with different names to avoid
    issues during testing.
    """
    if index_cls is PineconeIndex:
        index = index_cls(
            index_name=TEST_ID, dimensions=dimensions, namespace=namespace, sync=sync
        )
    else:
        index = index_cls()
    return index


def layer_json():
    return """{
    "encoder_type": "cohere",
    "encoder_name": "embed-english-v3.0",
    "routes": [
        {
            "name": "politics",
            "utterances": [
                "isn't politics the best thing ever",
                "why don't you tell me about your political opinions"
            ],
            "description": null,
            "function_schemas": null
        },
        {
            "name": "chitchat",
            "utterances": [
                "how's the weather today?",
                "how are things going?"
            ],
            "description": null,
            "function_schemas": null
        }
    ]
}"""


def layer_yaml():
    return """encoder_name: embed-english-v3.0
encoder_type: cohere
routes:
- description: null
  function_schemas: null
  name: politics
  utterances:
  - isn't politics the best thing ever
  - why don't you tell me about your political opinions
- description: null
  function_schemas: null
  name: chitchat
  utterances:
  - how's the weather today?
  - how are things going?
    """


@pytest.fixture
def base_encoder():
    return BaseEncoder(name="test-encoder", score_threshold=0.5)


@pytest.fixture
def cohere_encoder(mocker):
    mocker.patch.object(CohereEncoder, "__call__", side_effect=mock_encoder_call)
    return CohereEncoder(name="test-cohere-encoder", cohere_api_key="test_api_key")


@pytest.fixture
def openai_encoder(mocker):
    mocker.patch.object(OpenAIEncoder, "__call__", side_effect=mock_encoder_call)
    return OpenAIEncoder(name="text-embedding-3-small", openai_api_key="test_api_key")


@pytest.fixture
def routes():
    return [
        Route(name="Route 1", utterances=["Hello", "Hi"], metadata={"type": "default"}),
        Route(name="Route 2", utterances=["Goodbye", "Bye", "Au revoir"]),
    ]


@pytest.fixture
def routes_2():
    return [
        Route(name="Route 1", utterances=["Hello"]),
        Route(name="Route 2", utterances=["Hi"]),
    ]


@pytest.fixture
def routes_3():
    return [
        Route(name="Route 1", utterances=["Hello"]),
        Route(name="Route 2", utterances=["Asparagus"]),
    ]


@pytest.fixture
def routes_4():
    return [
        Route(name="Route 1", utterances=["Goodbye"], metadata={"type": "default"}),
        Route(name="Route 2", utterances=["Asparagus"]),
    ]


@pytest.fixture
def dynamic_routes():
    return [
        Route(
            name="Route 1",
            utterances=["Hello", "Hi"],
            function_schemas=[{"name": "test"}],
        ),
        Route(
            name="Route 2",
            utterances=["Goodbye", "Bye", "Au revoir"],
            function_schemas=[{"name": "test"}],
        ),
    ]


@pytest.fixture
def test_data():
    return [
        ("What's your opinion on the current government?", "politics"),
        ("what's the weather like today?", "chitchat"),
        ("what is the Pythagorean theorem?", "mathematics"),
        ("what is photosynthesis?", "biology"),
        ("tell me an interesting fact", None),
    ]


def get_test_indexes():
    indexes = []

    if importlib.util.find_spec("qdrant_client") is not None:
        indexes.append(QdrantIndex)
    if importlib.util.find_spec("pinecone") is not None:
        indexes.append(PineconeIndex)

    return indexes


@pytest.mark.parametrize("index_cls", get_test_indexes())
class TestRouteLayer:
    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_initialization(self, openai_encoder, routes, index_cls):
        index = init_index(index_cls)
        _ = RouteLayer(encoder=openai_encoder, routes=routes, top_k=10, index=index)

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_second_initialization_sync(self, openai_encoder, routes, index_cls):
        index = init_index(index_cls)
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, top_k=10, index=index
        )
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)  # allow for index to be populated
        assert route_layer.is_synced()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_second_initialization_not_synced(
        self, openai_encoder, routes, routes_2, index_cls
    ):
        index = init_index(index_cls, sync=None)
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes, index=index)
        _ = RouteLayer(encoder=openai_encoder, routes=routes_2, index=index)
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)  # allow for index to be populated
        assert route_layer.is_synced() is False

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_utterance_diff(self, openai_encoder, routes, routes_2, index_cls):
        index = init_index(index_cls)
        _ = RouteLayer(encoder=openai_encoder, routes=routes, top_k=10, index=index)
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)  # allow for index to be populated
        route_layer_2 = RouteLayer(
            encoder=openai_encoder, routes=routes_2, top_k=10, index=index
        )
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)  # allow for index to be populated
        diff = route_layer_2.get_utterance_diff()
        assert "  Route 1: Hello" in diff
        assert "+ Route 1: Hi" in diff
        assert "+ Route 2: Au revoir" in diff
        assert "+ Route 2: Bye" in diff
        assert "+ Route 2: Goodbye" in diff
        assert "  Route 2: Hi" in diff
