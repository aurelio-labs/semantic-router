import importlib
import os
import uuid
from datetime import datetime
from platform import python_version
from typing import Any, List
from unittest.mock import mock_open, patch

import numpy as np
import pytest

from semantic_router.encoders import CohereEncoder, DenseEncoder, OpenAIEncoder
from semantic_router.encoders.base import (
    AsymmetricDenseMixin,
    AsymmetricSparseMixin,
    SparseEncoder,
)
from semantic_router.index import LocalIndex, PineconeIndex, PostgresIndex, QdrantIndex
from semantic_router.llms import BaseLLM, OpenAILLM
from semantic_router.route import Route
from semantic_router.routers import HybridRouter, RouterConfig, SemanticRouter
from semantic_router.schema import RouteChoice, SparseEmbedding

PINECONE_BASE_URL = os.getenv("PINECONE_API_BASE_URL", "http://localhost:5080")


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
    return DenseEncoder(name="test-encoder", score_threshold=0.5)


@pytest.fixture
def cohere_encoder(mocker):
    mocker.patch.object(CohereEncoder, "__call__", side_effect=mock_encoder_call)

    # Mock async call
    async def async_mock_encoder_call(docs=None, utterances=None):
        # Handle either docs or utterances parameter
        texts = docs if docs is not None else utterances
        return mock_encoder_call(texts)

    mocker.patch.object(CohereEncoder, "acall", side_effect=async_mock_encoder_call)
    return CohereEncoder(name="test-cohere-encoder", cohere_api_key="test_api_key")


@pytest.fixture
def openai_encoder(mocker):
    # Mock the OpenAI client creation and API calls
    mocker.patch("openai.OpenAI")
    # Mock the __call__ method
    mocker.patch.object(OpenAIEncoder, "__call__", side_effect=mock_encoder_call)

    # Mock async call
    async def async_mock_encoder_call(docs=None, utterances=None):
        # Handle either docs or utterances parameter
        texts = docs if docs is not None else utterances
        return mock_encoder_call(texts)

    mocker.patch.object(OpenAIEncoder, "acall", side_effect=async_mock_encoder_call)
    # Create and return the mocked encoder
    encoder = OpenAIEncoder(name="text-embedding-3-small")
    return encoder


@pytest.fixture
def mock_openai_llm(mocker):
    # Mock the OpenAI LLM
    mocker.patch.object(OpenAILLM, "__call__", return_value="mocked response")

    # also async
    async def async_mock_llm_call(messages=None, **kwargs):
        return "mocked response"

    mocker.patch.object(OpenAILLM, "acall", side_effect=async_mock_llm_call)

    return OpenAILLM(name="fake-model-v1")


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
def routes_5():
    return [
        Route(name="Route 1", utterances=["Hello", "Hi"], metadata={"type": "default"}),
        Route(name="Route 2", utterances=["Goodbye", "Bye", "Au revoir"]),
        Route(name="Route 3", utterances=["Hello", "Hi"]),
        Route(name="Route 4", utterances=["Goodbye", "Bye", "Au revoir"]),
    ]


@pytest.fixture
def route_single_utterance():
    return [
        Route(name="Route 3", utterances=["Hello"]),
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


def get_test_encoders():
    encoders = [OpenAIEncoder]
    if importlib.util.find_spec("cohere") is not None:
        encoders.append(CohereEncoder)
    return encoders


def get_test_routers():
    routers = [SemanticRouter, HybridRouter]
    return routers


def get_test_indexes():
    indexes = [LocalIndex]
    if importlib.util.find_spec("qdrant_client") is not None:
        indexes.append(QdrantIndex)
    if importlib.util.find_spec("pinecone") is not None:
        indexes.append(PineconeIndex)
    if importlib.util.find_spec("psycopg") is not None:
        indexes.append(PostgresIndex)
    return indexes


def get_test_async_indexes():
    indexes = [LocalIndex]
    if importlib.util.find_spec("qdrant_client") is not None:
        indexes.append(QdrantIndex)
    if importlib.util.find_spec("pinecone") is not None:
        indexes.append(PineconeIndex)
    # PostgresIndex async operations are not fully supported; exclude from async tests
    return indexes


def init_index(
    index_cls,
    dimensions: int = 3,  # Default to 3 for our mock encoder
    namespace: str = "",
    index_name: str | None = None,
    init_async_index: bool = False,
):
    """Initialize indexes for unit testing."""
    if index_cls is QdrantIndex:
        index_name = index_name or f"test_{uuid.uuid4().hex}"
        return QdrantIndex(index_name=index_name, init_async_index=init_async_index)
    if index_cls is PineconeIndex:
        # In CI cloud mode, require a shared index to avoid quota/timeouts
        cloud_mode = os.getenv("PINECONE_API_BASE_URL", "").startswith(
            "https://api.pinecone.io"
        )
        if cloud_mode and not os.getenv("PINECONE_INDEX_NAME"):
            pytest.skip(
                "Skipping Pinecone in cloud: set PINECONE_INDEX_NAME to an existing index to run."
            )
        # Use local Pinecone instance
        index_name = (
            f"test-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            if not index_name
            else index_name
        )
        index = index_cls(
            index_name=index_name,
            dimensions=dimensions,
            namespace=namespace,
            init_async_index=init_async_index,
            base_url=PINECONE_BASE_URL,
        )
    elif index_cls is PostgresIndex:
        index = index_cls(
            index_name=index_name or "test_index",
            index_prefix="",
            namespace=namespace,
            dimensions=dimensions,
            init_async_index=init_async_index,
        )
    elif index_cls is None:
        return None
    else:
        index = index_cls(init_async_index=init_async_index)
    return index


class TestRouterConfig:
    def test_from_file_json(self, tmp_path):
        # Create a temporary JSON file with layer configuration
        config_path = tmp_path / "config.json"
        config_path.write_text(
            layer_json()
        )  # Assuming layer_json() returns a valid JSON string

        # Load the RouterConfig from the temporary file
        layer_config = RouterConfig.from_file(str(config_path))

        # Assertions to verify the loaded configuration
        assert layer_config.encoder_type == "cohere"
        assert layer_config.encoder_name == "embed-english-v3.0"
        assert len(layer_config.routes) == 2
        assert layer_config.routes[0].name == "politics"

    def test_from_file_yaml(self, tmp_path):
        # Create a temporary YAML file with layer configuration
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            layer_yaml()
        )  # Assuming layer_yaml() returns a valid YAML string

        # Load the RouterConfig from the temporary file
        layer_config = RouterConfig.from_file(str(config_path))

        # Assertions to verify the loaded configuration
        assert layer_config.encoder_type == "cohere"
        assert layer_config.encoder_name == "embed-english-v3.0"
        assert len(layer_config.routes) == 2
        assert layer_config.routes[0].name == "politics"

    def test_from_file_invalid_path(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            RouterConfig.from_file("nonexistent_path.json")
        assert "[Errno 2] No such file or directory: 'nonexistent_path.json'" in str(
            excinfo.value
        )

    def test_from_file_unsupported_type(self, tmp_path):
        # Create a temporary unsupported file
        config_path = tmp_path / "config.unsupported"
        config_path.write_text(layer_json())

        with pytest.raises(ValueError) as excinfo:
            RouterConfig.from_file(str(config_path))
        assert "Unsupported file type" in str(excinfo.value)

    def test_from_file_invalid_config(self, tmp_path):
        # Define an invalid configuration JSON
        invalid_config_json = """
        {
            "encoder_type": "cohere",
            "encoder_name": "embed-english-v3.0",
            "routes": "This should be a list, not a string"
        }"""

        # Write the invalid configuration to a temporary JSON file
        config_path = tmp_path / "invalid_config.json"
        with open(config_path, "w") as file:
            file.write(invalid_config_json)

        # Patch the is_valid function to return False for this test
        with patch("semantic_router.routers.base.is_valid", return_value=False):
            # Attempt to load the RouterConfig from the temporary file
            # and assert that it raises an exception due to invalid configuration
            with pytest.raises(Exception) as excinfo:
                RouterConfig.from_file(str(config_path))
            assert "Invalid config JSON or YAML" in str(excinfo.value), (
                "Loading an invalid configuration should raise an exception."
            )

    def test_from_file_with_llm(self, tmp_path):
        llm_config_json = """
        {
            "encoder_type": "cohere",
            "encoder_name": "embed-english-v3.0",
            "routes": [
                {
                    "name": "llm_route",
                    "utterances": ["tell me a joke", "say something funny"],
                    "llm": {
                        "module": "semantic_router.llms.base",
                        "class": "BaseLLM",
                        "model": "fake-model-v1"
                    }
                }
            ]
        }"""

        config_path = tmp_path / "config_with_llm.json"
        with open(config_path, "w") as file:
            file.write(llm_config_json)

        # Load the RouterConfig from the temporary file
        layer_config = RouterConfig.from_file(str(config_path))

        # Using BaseLLM because trying to create a usable Mock LLM is a nightmare.
        assert isinstance(layer_config.routes[0].llm, BaseLLM), (
            "LLM should be instantiated and associated with the route based on the "
        )
        "config"
        assert layer_config.routes[0].llm.name == "fake-model-v1", (
            "LLM instance should have the 'name' attribute set correctly"
        )

    def test_init(self):
        layer_config = RouterConfig()
        assert layer_config.routes == []

    def test_to_file_json(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = RouterConfig(routes=[route])
        with patch("builtins.open", mock_open()) as mocked_open:
            layer_config.to_file("data/test_output.json")
            mocked_open.assert_called_once_with("data/test_output.json", "w")

    def test_to_file_yaml(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = RouterConfig(routes=[route])
        with patch("builtins.open", mock_open()) as mocked_open:
            layer_config.to_file("data/test_output.yaml")
            mocked_open.assert_called_once_with("data/test_output.yaml", "w")

    def test_to_file_invalid(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = RouterConfig(routes=[route])
        with pytest.raises(ValueError):
            layer_config.to_file("test_output.txt")

    def test_from_file_invalid(self):
        with open("test.txt", "w") as f:
            f.write("dummy content")
        with pytest.raises(ValueError):
            RouterConfig.from_file("test.txt")
        os.remove("test.txt")

    def test_to_dict(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = RouterConfig(routes=[route])
        assert layer_config.to_dict()["routes"] == [route.to_dict()]

    def test_add(self):
        route = Route(name="test", utterances=["utterance"])
        route2 = Route(name="test2", utterances=["utterance2"])
        layer_config = RouterConfig()
        layer_config.add(route)
        # confirm route added
        assert layer_config.routes == [route]
        # add second route and check updates
        layer_config.add(route2)
        assert layer_config.routes == [route, route2]

    def test_get(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = RouterConfig(routes=[route])
        assert layer_config.get("test") == route

    def test_get_not_found(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = RouterConfig(routes=[route])
        assert layer_config.get("not_found") is None

    def test_remove(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = RouterConfig(routes=[route])
        layer_config.remove("test")
        assert layer_config.routes == []

    def test_setting_aggregation_methods(self, openai_encoder, routes):
        for agg in ["sum", "mean", "max"]:
            route_layer = SemanticRouter(
                encoder=openai_encoder,
                routes=routes,
                aggregation=agg,
            )
            assert route_layer.aggregation == agg

    def test_semantic_classify_multiple_routes_with_different_aggregation(
        self, openai_encoder, routes
    ):
        route_scores = [
            {"route": "Route 1", "score": 0.5},
            {"route": "Route 1", "score": 0.5},
            {"route": "Route 1", "score": 0.5},
            {"route": "Route 1", "score": 0.5},
            {"route": "Route 2", "score": 0.4},
            {"route": "Route 2", "score": 0.6},
            {"route": "Route 2", "score": 0.8},
            {"route": "Route 3", "score": 0.1},
            {"route": "Route 3", "score": 1.0},
        ]
        for agg in ["sum", "mean", "max"]:
            route_layer = SemanticRouter(
                encoder=openai_encoder,
                routes=routes,
                aggregation=agg,
            )
            classification, score = route_layer._semantic_classify(route_scores)

            if agg == "sum":
                assert classification == "Route 1"
                assert score == [0.5, 0.5, 0.5, 0.5]
            elif agg == "mean":
                assert classification == "Route 2"
                assert score == [0.4, 0.6, 0.8]
            elif agg == "max":
                assert classification == "Route 3"
                assert score == [0.1, 1.0]


class MockSymmetricDenseEncoder(DenseEncoder):
    def __call__(self, docs: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in docs]

    async def acall(self, docs: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in docs]


class MockSymmetricSparseEncoder(SparseEncoder):
    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    async def acall(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]


class MockAsymmetricDenseEncoder(DenseEncoder, AsymmetricDenseMixin):
    def __call__(self, docs: List[Any]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in docs]

    async def acall(self, docs: List[Any]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in docs]

    def encode_queries(self, docs: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in docs]

    def encode_documents(self, docs: List[str]) -> List[List[float]]:
        return [[0.4, 0.5, 0.6] for _ in docs]

    async def aencode_queries(self, docs: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in docs]

    async def aencode_documents(self, docs: List[str]) -> List[List[float]]:
        return [[0.4, 0.5, 0.6] for _ in docs]


class MockAsymmetricSparseEncoder(SparseEncoder, AsymmetricSparseMixin):
    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    async def acall(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    def encode_queries(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    def encode_documents(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    async def aencode_queries(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]

    async def aencode_documents(self, docs: List[str]) -> List[SparseEmbedding]:
        return [SparseEmbedding(embedding=np.array([[0, 0.1], [1, 0.2]])) for _ in docs]


@pytest.mark.parametrize(
    "dense_encoder_cls,sparse_encoder_cls,input_type",
    [
        (encoder, sparse_encoder, input_type)
        for encoder in [MockSymmetricDenseEncoder, MockAsymmetricDenseEncoder]
        for sparse_encoder in [MockSymmetricSparseEncoder, MockAsymmetricSparseEncoder]
        for input_type in ["queries", "documents"]
    ],
)
class TestHybridRouter:
    def test_encode(
        self, dense_encoder_cls, sparse_encoder_cls, input_type, routes, mocker
    ):
        encoder = dense_encoder_cls(name="Dense Encoder")
        sparse_encoder = sparse_encoder_cls(name="Sparse Encoder")
        router = HybridRouter(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            routes=routes,
        )

        # Set up spies for symmetric methods
        dense_call_spy = mocker.spy(dense_encoder_cls, "__call__")
        sparse_call_spy = mocker.spy(sparse_encoder_cls, "__call__")

        # Set up spies for asymmetric methods if applicable
        if isinstance(encoder, AsymmetricDenseMixin):
            dense_encode_queries_spy = mocker.spy(dense_encoder_cls, "encode_queries")
            dense_encode_documents_spy = mocker.spy(
                dense_encoder_cls, "encode_documents"
            )

        if isinstance(sparse_encoder, AsymmetricSparseMixin):
            sparse_encode_queries_spy = mocker.spy(sparse_encoder_cls, "encode_queries")
            sparse_encode_documents_spy = mocker.spy(
                sparse_encoder_cls, "encode_documents"
            )

        test_query = ["test query"]

        # Test synchronous encoding
        router._encode(test_query, input_type=input_type)

        # Verify correct methods were called based on encoder type and input_type
        if isinstance(encoder, AsymmetricDenseMixin):
            if input_type == "documents":
                assert dense_encode_documents_spy.called
            else:  # queries
                assert dense_encode_queries_spy.called
        else:
            assert dense_call_spy.called

        if isinstance(sparse_encoder, AsymmetricSparseMixin):
            if input_type == "documents":
                assert sparse_encode_documents_spy.called
            else:  # queries
                assert sparse_encode_queries_spy.called
        else:
            assert sparse_call_spy.called

    @pytest.mark.asyncio
    async def test_async_encode(
        self, dense_encoder_cls, sparse_encoder_cls, input_type, routes, mocker
    ):
        encoder = dense_encoder_cls(name="Dense Encoder")
        sparse_encoder = sparse_encoder_cls(name="Sparse Encoder")
        router = HybridRouter(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            routes=routes,
        )

        # Set up spies for symmetric methods
        dense_call_spy = mocker.spy(dense_encoder_cls, "acall")
        sparse_call_spy = mocker.spy(sparse_encoder_cls, "acall")

        # Set up spies for asymmetric methods if applicable
        if isinstance(encoder, AsymmetricDenseMixin):
            dense_encode_queries_spy = mocker.spy(dense_encoder_cls, "aencode_queries")
            dense_encode_documents_spy = mocker.spy(
                dense_encoder_cls, "aencode_documents"
            )

        if isinstance(sparse_encoder, AsymmetricSparseMixin):
            sparse_encode_queries_spy = mocker.spy(
                sparse_encoder_cls, "aencode_queries"
            )
            sparse_encode_documents_spy = mocker.spy(
                sparse_encoder_cls, "aencode_documents"
            )

        test_query = ["test query"]

        # Test asynchronous encoding
        await router._async_encode(test_query, input_type=input_type)

        # Verify correct methods were called based on encoder type and input_type
        if isinstance(encoder, AsymmetricDenseMixin):
            if input_type == "documents":
                assert dense_encode_documents_spy.called
            else:  # queries
                assert dense_encode_queries_spy.called
        else:
            assert dense_call_spy.called

        if isinstance(sparse_encoder, AsymmetricSparseMixin):
            if input_type == "documents":
                assert sparse_encode_documents_spy.called
            else:  # queries
                assert sparse_encode_queries_spy.called
        else:
            assert sparse_call_spy.called


@pytest.mark.parametrize(
    "router_cls,index_cls",
    [
        (router, index)
        for router in [HybridRouter, SemanticRouter]
        # None for default LocalIndex behavior, and PostgresIndex is not supported for async tests
        for index in [None] + get_test_async_indexes()
    ],
)
class TestRouterAsync:
    @pytest.mark.asyncio
    async def test_async_query_parameter(self, router_cls, index_cls, routes_5, mocker):
        """Test that we return expected values in RouteChoice objects."""
        # Create router with mock encoders
        dense_encoder = MockSymmetricDenseEncoder(name="Dense Encoder")
        index = init_index(index_cls, init_async_index=True)
        # we don't test postgres and hybrid together
        if index_cls is PostgresIndex and router_cls == HybridRouter:
            pytest.skip("PostgresIndex does not support hybrid")
        if router_cls == HybridRouter:
            sparse_encoder = MockSymmetricSparseEncoder(name="Sparse Encoder")
            router = router_cls(
                encoder=dense_encoder,
                sparse_encoder=sparse_encoder,
                routes=routes_5,
                index=index,
                auto_sync="local",
                init_async_index=True,
            )
        else:
            router = router_cls(
                encoder=dense_encoder,
                routes=routes_5,
                index=index,
                auto_sync="local",
                init_async_index=True,
            )

        # Setup a mock for the similarity calculation method
        _ = mocker.patch.object(
            router,
            "_score_routes",
            return_value=[
                ("Route 1", 0.9, [0.1, 0.2, 0.3]),
                ("Route 2", 0.8, [0.4, 0.5, 0.6]),
                ("Route 3", 0.7, [0.7, 0.8, 0.9]),
                ("Route 4", 0.6, [1.0, 1.1, 1.2]),
            ],
        )
        # Test without limit (should return only the top match)
        result = await router.acall("test query")
        assert result is not None
        assert isinstance(result, RouteChoice)

        # Confirm we have Route 1 and sim score
        assert result.name == "Route 1"
        assert result.similarity_score == 0.9
        assert result.function_call is None

    @pytest.mark.asyncio
    async def test_async_limit_parameter(self, router_cls, index_cls, routes_5, mocker):
        """Test that the limit parameter works correctly for async router calls."""
        # we don't test postgres and hybrid together
        if index_cls is PostgresIndex and router_cls == HybridRouter:
            pytest.skip("PostgresIndex does not support hybrid")
        # Create router with mock encoders
        dense_encoder = MockSymmetricDenseEncoder(name="Dense Encoder")
        index = init_index(index_cls, init_async_index=True)
        if router_cls == HybridRouter:
            sparse_encoder = MockSymmetricSparseEncoder(name="Sparse Encoder")
            router = router_cls(
                encoder=dense_encoder,
                sparse_encoder=sparse_encoder,
                routes=routes_5,
                index=index,
                auto_sync="local",
                init_async_index=True,
            )
        else:
            router = router_cls(
                encoder=dense_encoder,
                routes=routes_5,
                index=index,
                auto_sync="local",
                init_async_index=True,
            )

        # Setup a mock for the async similarity calculation method
        _ = mocker.patch.object(
            router,
            "_score_routes",
            return_value=[
                ("Route 1", 0.9, [0.1, 0.2, 0.3]),
                ("Route 2", 0.8, [0.4, 0.5, 0.6]),
                ("Route 3", 0.7, [0.7, 0.8, 0.9]),
                ("Route 4", 0.6, [1.0, 1.1, 1.2]),
            ],
        )

        # Test without limit (should return only the top match)
        result = await router.acall("test query")
        assert result is not None
        assert isinstance(result, RouteChoice)

        # Test with limit=2 (should return top 2 matches)
        result = await router.acall("test query", limit=2)
        assert result is not None
        assert len(result) == 2

        # Test with limit=None (should return all matches)
        result = await router.acall("test query", limit=None)
        assert result is not None
        assert len(result) == 4  # Should return all matches

    @pytest.mark.asyncio
    async def test_async_index_operations(
        self, router_cls, index_cls, routes, openai_encoder
    ):
        # we don't test postgres and hybrid together
        if index_cls is PostgresIndex and router_cls == HybridRouter:
            pytest.skip("PostgresIndex does not support hybrid")
        if index_cls is None:
            pytest.skip("Test only for specific index implementations")

        index = init_index(index_cls, init_async_index=True)

        if router_cls == HybridRouter:
            sparse_encoder = MockSymmetricSparseEncoder(name="Sparse Encoder")
            router = router_cls(
                encoder=openai_encoder,
                sparse_encoder=sparse_encoder,
                routes=[],
                index=index,
                auto_sync="local",
                init_async_index=True,
            )
        else:
            router = router_cls(
                encoder=openai_encoder,
                routes=[],
                index=index,
                auto_sync="local",
                init_async_index=True,
            )

        # Test adding routes
        assert await router.index.alen() == 0
        await router.aadd(routes[0])
        assert await router.index.alen() == 2  # "Hello" and "Hi"

        await router.aadd(routes[1])
        assert await router.index.alen() == 5  # All utterances

        # Test deleting routes
        await router.adelete("Route 1")
        assert await router.index.alen() == 3  # Only Route 2 utterances

        # Test delete
        await router.index.adelete_index()
        assert await router.index.alen() == 0


@pytest.mark.parametrize(
    "router_cls,index_cls",
    [
        (router, index)
        for router in [HybridRouter, SemanticRouter]
        for index in [None] + get_test_indexes()  # None for default LocalIndex behavior
    ],
)
class TestRouter:
    def test_query_parameter(self, router_cls, index_cls, routes_5, mocker):
        """Test that we return expected values in RouteChoice objects."""
        # Create router with mock encoders
        dense_encoder = MockSymmetricDenseEncoder(name="Dense Encoder")
        index = init_index(index_cls) if index_cls else None

        if router_cls == HybridRouter:
            sparse_encoder = MockSymmetricSparseEncoder(name="Sparse Encoder")
            router = router_cls(
                encoder=dense_encoder,
                sparse_encoder=sparse_encoder,
                routes=routes_5,
                index=index,
                auto_sync="local",
            )
        else:
            router = router_cls(
                encoder=dense_encoder,
                routes=routes_5,
                index=index,
                auto_sync="local",
            )

        # Setup a mock for the similarity calculation method
        _ = mocker.patch.object(
            router,
            "_score_routes",
            return_value=[
                ("Route 1", 0.9, [0.1, 0.2, 0.3]),
                ("Route 2", 0.8, [0.4, 0.5, 0.6]),
                ("Route 3", 0.7, [0.7, 0.8, 0.9]),
                ("Route 4", 0.6, [1.0, 1.1, 1.2]),
            ],
        )

        # Test without limit (should return only the top match)
        result = router("test query")
        assert result is not None
        assert isinstance(result, RouteChoice)

        # Confirm we have Route 1 and sim score
        assert result.name == "Route 1"
        assert result.similarity_score == 0.9
        assert result.function_call is None

    def test_limit_parameter(self, router_cls, index_cls, routes_5, mocker):
        """Test that the limit parameter works correctly for sync router calls."""
        # Create router with mock encoders
        dense_encoder = MockSymmetricDenseEncoder(name="Dense Encoder")
        index = init_index(index_cls) if index_cls else None

        if router_cls == HybridRouter:
            sparse_encoder = MockSymmetricSparseEncoder(name="Sparse Encoder")
            router = router_cls(
                encoder=dense_encoder,
                sparse_encoder=sparse_encoder,
                routes=routes_5,
                index=index,
                auto_sync="local",
            )
        else:
            router = router_cls(
                encoder=dense_encoder,
                routes=routes_5,
                index=index,
                auto_sync="local",
            )

        # Setup a mock for the similarity calculation method
        _ = mocker.patch.object(
            router,
            "_score_routes",
            return_value=[
                ("Route 1", 0.9, [0.1, 0.2, 0.3]),
                ("Route 2", 0.8, [0.4, 0.5, 0.6]),
                ("Route 3", 0.7, [0.7, 0.8, 0.9]),
                ("Route 4", 0.6, [1.0, 1.1, 1.2]),
            ],
        )

        # Test without limit (should return only the top match)
        result = router("test query")
        assert result is not None
        assert isinstance(result, RouteChoice)

        # Test with limit=2 (should return top 2 matches)
        result = router("test query", limit=2)
        assert result is not None
        assert len(result) == 2

        # Test with limit=None (should return all matches)
        result = router("test query", limit=None)
        assert result is not None
        assert len(result) == 4  # Should return all matches

    def test_index_operations(self, router_cls, index_cls, routes, openai_encoder):
        """Test index-specific operations like add, delete, and sync."""
        if index_cls is None:
            pytest.skip("Test only for specific index implementations")

        index = init_index(index_cls)

        if router_cls == HybridRouter:
            sparse_encoder = MockSymmetricSparseEncoder(name="Sparse Encoder")
            router = router_cls(
                encoder=openai_encoder,
                sparse_encoder=sparse_encoder,
                routes=[],
                index=index,
                auto_sync="local",
            )
        else:
            router = router_cls(
                encoder=openai_encoder,
                routes=[],
                index=index,
                auto_sync="local",
            )

        # Test adding routes
        assert len(router.index) == 0
        router.add(routes[0])
        assert len(router.index) == 2  # "Hello" and "Hi"

        router.add(routes[1])
        assert len(router.index) == 5  # All utterances

        # Test deleting routes
        router.delete("Route 1")
        assert len(router.index) == 3  # Only Route 2 utterances

        # Test delete
        router.index.delete_index()
        assert len(router.index) == 0
