import importlib
import os
import tempfile
import time
from datetime import datetime
from functools import wraps
from platform import python_version
from typing import Optional
from unittest.mock import mock_open, patch

import pytest

from semantic_router.encoders import CohereEncoder, DenseEncoder, OpenAIEncoder
from semantic_router.index.local import LocalIndex
from semantic_router.index.pinecone import PineconeIndex
from semantic_router.index.qdrant import QdrantIndex
from semantic_router.llms import BaseLLM, OpenAILLM
from semantic_router.route import Route
from semantic_router.routers import HybridRouter, RouterConfig, SemanticRouter
from semantic_router.utils.logger import logger

PINECONE_SLEEP = 8
RETRY_COUNT = 10


# retry decorator for PineconeIndex cases (which need delay)
def retry(max_retries: int = 5, delay: int = 8):
    """Retry decorator, currently used for PineconeIndex which often needs some time
    to be populated and have all correct data. Once full Pinecone mock is built we
    should remove this decorator.

    :param max_retries: Maximum number of retries.
    :param delay: Delay between retries in seconds.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            count = 0
            last_exception = None
            while count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {count} | Error in {func.__name__}: {e}")
                    last_exception = e
                    count += 1
                    time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


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
    index_name: Optional[str] = None,
):
    """We use this function to initialize indexes with different names to avoid
    issues during testing.
    """
    if index_cls is PineconeIndex:
        # we specify different index names to avoid dimensionality issues between different encoders
        index_name = TEST_ID if not index_name else f"{TEST_ID}-{index_name.lower()}"
        index = index_cls(
            index_name=index_name, dimensions=dimensions, namespace=namespace
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


def get_test_indexes():
    indexes = [LocalIndex]

    if importlib.util.find_spec("qdrant_client") is not None:
        indexes.append(QdrantIndex)
    if importlib.util.find_spec("pinecone") is not None:
        indexes.append(PineconeIndex)

    return indexes


def get_test_encoders():
    encoders = [OpenAIEncoder]
    if importlib.util.find_spec("cohere") is not None:
        encoders.append(CohereEncoder)
    return encoders


def get_test_routers():
    routers = [SemanticRouter]
    if importlib.util.find_spec("pinecone_text") is not None:
        routers.append(HybridRouter)
    return routers


@pytest.mark.parametrize(
    "index_cls,encoder_cls,router_cls",
    [
        (index, encoder, router)
        for index in get_test_indexes()
        for encoder in get_test_encoders()
        for router in get_test_routers()
    ],
)
class TestIndexEncoders:
    def test_initialization(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
            top_k=10,
        )
        score_threshold = route_layer.score_threshold
        if isinstance(route_layer, HybridRouter):
            assert score_threshold == encoder.score_threshold * route_layer.alpha
        else:
            assert score_threshold == encoder.score_threshold
        assert route_layer.top_k == 10

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated():
            assert len(route_layer.index) == 5

        check_index_populated()

        assert (
            len(set(route_layer._get_route_names()))
            if route_layer._get_route_names() is not None
            else 0 == 2
        )

    def test_initialization_different_encoders(
        self, encoder_cls, index_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, index=index)
        score_threshold = route_layer.score_threshold
        if isinstance(route_layer, HybridRouter):
            assert score_threshold == encoder.score_threshold * route_layer.alpha
        else:
            assert score_threshold == encoder.score_threshold

    def test_initialization_no_encoder(self, index_cls, encoder_cls, router_cls):
        route_layer_none = router_cls(encoder=None)
        score_threshold = route_layer_none.score_threshold
        if isinstance(route_layer_none, HybridRouter):
            assert score_threshold == 0.3 * route_layer_none.alpha
        else:
            assert score_threshold == 0.3


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
            assert "Invalid config JSON or YAML" in str(
                excinfo.value
            ), "Loading an invalid configuration should raise an exception."

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
        assert isinstance(
            layer_config.routes[0].llm, BaseLLM
        ), "LLM should be instantiated and associated with the route based on the "
        "config"
        assert (
            layer_config.routes[0].llm.name == "fake-model-v1"
        ), "LLM instance should have the 'name' attribute set correctly"

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


@pytest.mark.parametrize(
    "index_cls,encoder_cls,router_cls",
    [
        (index, encoder, router)
        for index in get_test_indexes()
        for encoder in [OpenAIEncoder]
        for router in get_test_routers()
    ],
)
class TestSemanticRouter:
    def test_initialization_dynamic_route(
        self, dynamic_routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=dynamic_routes,
            index=index,
            auto_sync="local",
        )
        score_threshold = route_layer.score_threshold
        if isinstance(route_layer, HybridRouter):
            assert score_threshold == encoder.score_threshold * route_layer.alpha
        else:
            assert score_threshold == encoder.score_threshold

    def test_add_single_utterance(
        self, routes, route_single_utterance, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )
        route_layer.add(routes=route_single_utterance)
        score_threshold = route_layer.score_threshold
        if isinstance(route_layer, HybridRouter):
            assert score_threshold == encoder.score_threshold * route_layer.alpha
        else:
            assert score_threshold == encoder.score_threshold

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated():
            _ = route_layer("Hello")
            assert len(route_layer.index.get_utterances()) == 6

        check_index_populated()

    def test_init_and_add_single_utterance(
        self, route_single_utterance, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            index=index,
            auto_sync="local",
        )
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)  # allow for index to be updated
        route_layer.add(routes=route_single_utterance)
        score_threshold = route_layer.score_threshold
        if isinstance(route_layer, HybridRouter):
            assert score_threshold == encoder.score_threshold * route_layer.alpha
        else:
            assert score_threshold == encoder.score_threshold

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated():
            _ = route_layer("Hello")
            assert len(route_layer.index.get_utterances()) == 1

        check_index_populated()

    def test_delete_index(self, routes, index_cls, encoder_cls, router_cls):
        # TODO merge .delete_index() and .delete_all() and get working
        index = init_index(index_cls)
        encoder = encoder_cls()
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )

        # delete index
        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def delete_index():
            route_layer.index.delete_index()
            # assert index empty
            assert route_layer.index.get_utterances() == []

        delete_index()

    def test_add_route(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder, routes=[], index=index, auto_sync="local"
        )
        # Initially, the local routes list should be empty
        assert route_layer.routes == []

        # same for the remote index
        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_empty():
            assert route_layer.index.get_utterances() == []

        check_index_empty()
        # Add route1 and check
        route_layer.add(routes=routes[0])

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated1():
            assert route_layer.routes == [routes[0]]
            assert route_layer.index is not None
            assert len(route_layer.index.get_utterances()) == 2

        check_index_populated1()

        # Add route2 and check
        route_layer.add(routes=routes[1])

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated2():
            assert route_layer.routes == [routes[0], routes[1]]
            assert len(route_layer.index.get_utterances()) == 5

        check_index_populated2()

    def test_list_route_names(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_route_names():
            route_names = route_layer.list_route_names()
            assert set(route_names) == {
                route.name for route in routes
            }, "The list of route names should match the names of the routes added."

        check_route_names()

    def test_delete_route(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )

        # Delete a route by name
        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def delete_route_by_name():
            route_to_delete = routes[0].name
            route_layer.delete(route_to_delete)
            # Ensure the route is no longer in the route layer
            assert (
                route_to_delete not in route_layer.list_route_names()
            ), "The route should be deleted from the route layer."
            # Ensure the route's utterances are no longer in the index
            for utterance in routes[0].utterances:
                assert (
                    utterance not in route_layer.index
                ), "The route's utterances should be deleted from the index."

        delete_route_by_name()

    def test_remove_route_not_found(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder, routes=routes, index=index, auto_sync="local"
        )

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def delete_non_existent_route():
            # Attempt to remove a route that does not exist
            non_existent_route = "non-existent-route"
            route_layer.delete(non_existent_route)
            # we should see warning in logs only (ie no errors)

        delete_non_existent_route()

    def test_add_multiple_routes(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            index=index,
            auto_sync="local",
        )

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_index_populated():
            route_layer.add(routes=routes)
            assert route_layer.index is not None
            assert len(route_layer.index.get_utterances()) == 5

        check_index_populated()

        # # clear index if pinecone
        # if index_cls is PineconeIndex:
        #     @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        #     def clear_index():
        #         route_layer.index.index.delete(delete_all=True)
        #         assert route_layer.index.get_utterances() == []
        #     clear_index()

    def test_query_and_classification(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        encoder.score_threshold = 0.1
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
            aggregation="max",
        )

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_query_result():
            query_result = route_layer(text="Hello").name
            assert query_result in ["Route 1", "Route 2"]

        check_query_result()

    def test_query_filter(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        encoder.score_threshold = 0.1
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
            aggregation="max",
        )

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_raises_value_error():
            try:
                route_layer(text="Hello", route_filter=["Route 8"]).name
            except ValueError:
                assert True

        check_raises_value_error()

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_query_result():
            query_result = route_layer(text="Hello", route_filter=["Route 1"]).name
            assert query_result in ["Route 1"]

        check_query_result()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_namespace_pinecone_index(self, routes, index_cls, encoder_cls, router_cls):
        if index_cls is PineconeIndex:
            encoder = encoder_cls()
            pineconeindex = init_index(
                index_cls, namespace="test", index_name=encoder.__class__.__name__
            )
            route_layer = router_cls(
                encoder=encoder,
                routes=routes,
                index=pineconeindex,
                auto_sync="local",
            )
            time.sleep(PINECONE_SLEEP)  # allow for index to be updated

            @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            def check_query_result():
                query_result = route_layer(text="Hello", route_filter=["Route 1"]).name
                assert query_result in ["Route 1"]

            check_query_result()

            @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            def delete_namespace():
                route_layer.index.index.delete(namespace="test", delete_all=True)

            delete_namespace()

    def test_query_with_no_index(self, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        route_layer = router_cls(encoder=encoder)
        # TODO: probably should avoid running this with multiple encoders or find a way to set dims
        with pytest.raises(ValueError):
            assert route_layer(text="Anything").name is None

    def test_query_with_vector(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        encoder.score_threshold = 0.1
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
            aggregation="max",
        )
        # create vectors
        vector = encoder(["hello"])
        if router_cls is HybridRouter:
            sparse_vector = route_layer.sparse_encoder(["hello"])[0]

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_query_result():
            if router_cls is HybridRouter:
                query_result = route_layer(
                    vector=vector, sparse_vector=sparse_vector
                ).name
            else:
                query_result = route_layer(vector=vector).name
            assert query_result in ["Route 1", "Route 2"]

        check_query_result()

    def test_query_with_no_text_or_vector(
        self, routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        with pytest.raises(ValueError):
            route_layer()

    def test_is_ready(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_is_ready():
            assert route_layer.index.is_ready()

        check_is_ready()


@pytest.mark.parametrize(
    "index_cls,encoder_cls,router_cls",
    [
        (index, encoder, router)
        for index in [LocalIndex]  # no need to test with multiple indexes
        for encoder in [OpenAIEncoder]  # no need to test with multiple encoders
        for router in get_test_routers()
    ],
)
class TestRouterOnly:
    def test_semantic_classify(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )
        classification, score = route_layer._semantic_classify(
            [
                {"route": "Route 1", "score": 0.9},
                {"route": "Route 2", "score": 0.1},
            ]
        )
        assert classification == "Route 1"
        assert score == [0.9]

    def test_semantic_classify_multiple_routes(
        self, routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )
        classification, score = route_layer._semantic_classify(
            [
                {"route": "Route 1", "score": 0.9},
                {"route": "Route 2", "score": 0.1},
                {"route": "Route 1", "score": 0.8},
            ]
        )
        assert classification == "Route 1"
        assert score == [0.9, 0.8]

    def test_query_no_text_dynamic_route(
        self, dynamic_routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=dynamic_routes, index=index)
        # create vectors
        vector = encoder(["hello"])
        if router_cls is HybridRouter:
            sparse_vector = route_layer.sparse_encoder(["hello"])[0]
        with pytest.raises(ValueError):
            if router_cls is HybridRouter:
                route_layer(vector=vector, sparse_vector=sparse_vector)
            else:
                route_layer(vector=vector)

    def test_pass_threshold(self, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            index=index,
            auto_sync="local",
        )
        assert not route_layer._pass_threshold([], 0.3)
        assert route_layer._pass_threshold([0.6, 0.7], 0.3)

    def test_failover_score_threshold(self, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            index=index,
            auto_sync="local",
        )
        if router_cls is HybridRouter:
            assert route_layer.score_threshold == 0.3 * route_layer.alpha
        else:
            assert route_layer.score_threshold == 0.3

    def test_json(self, routes, index_cls, encoder_cls, router_cls):
        temp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        try:
            temp_path = temp.name  # Save the temporary file's path
            temp.close()  # Close the file to ensure it can be opened again on Windows
            encoder = encoder_cls()
            index = init_index(index_cls, index_name=encoder.__class__.__name__)
            route_layer = router_cls(
                encoder=encoder,
                routes=routes,
                index=index,
                auto_sync="local",
            )
            route_layer.to_json(temp_path)
            assert os.path.exists(temp_path)
            route_layer_from_file = SemanticRouter.from_json(temp_path)
            assert (
                route_layer_from_file.index is not None
                and route_layer_from_file._get_route_names() is not None
            )
        finally:
            os.remove(temp_path)  # Ensure the file is deleted even if the test fails

    def test_yaml(self, routes, index_cls, encoder_cls, router_cls):
        temp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        try:
            temp_path = temp.name  # Save the temporary file's path
            temp.close()  # Close the file to ensure it can be opened again on Windows
            encoder = encoder_cls()
            index = init_index(index_cls, index_name=encoder.__class__.__name__)
            route_layer = router_cls(
                encoder=encoder,
                routes=routes,
                index=index,
                auto_sync="local",
            )
            route_layer.to_yaml(temp_path)
            assert os.path.exists(temp_path)
            route_layer_from_file = SemanticRouter.from_yaml(temp_path)
            assert (
                route_layer_from_file.index is not None
                and route_layer_from_file._get_route_names() is not None
            )
        finally:
            os.remove(temp_path)  # Ensure the file is deleted even if the test fails

    def test_config(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        # confirm route creation functions as expected
        layer_config = route_layer.to_config()
        assert layer_config.routes == route_layer.routes
        # now load from config and confirm it's the same
        route_layer_from_config = SemanticRouter.from_config(layer_config, index)
        assert (
            route_layer_from_config._get_route_names() == route_layer._get_route_names()
        )
        if router_cls is HybridRouter:
            # TODO: need to fix HybridRouter from config
            # assert (
            #     route_layer_from_config.score_threshold
            #     == route_layer.score_threshold * route_layer.alpha
            # )
            pass
        else:
            assert (
                route_layer_from_config.score_threshold == route_layer.score_threshold
            )

    def test_get_thresholds(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        if router_cls is HybridRouter:
            # TODO: fix this
            target = encoder.score_threshold * route_layer.alpha
            assert route_layer.get_thresholds() == {
                "Route 1": target,
                "Route 2": target,
            }
        else:
            assert route_layer.get_thresholds() == {"Route 1": 0.3, "Route 2": 0.3}

    def test_with_multiple_routes_passing_threshold(
        self, routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        route_layer.score_threshold = 0.3  # Set the score_threshold if needed
        # Assuming route_layer is already set up with routes "Route 1" and "Route 2"
        query_results = [
            {"route": "Route 1", "score": 0.1},
            {"route": "Route 2", "score": 0.8},
            {"route": "Route 1", "score": 0.9},
        ]
        expected = [("Route 1", 0.9), ("Route 2", 0.8)]
        results = route_layer._semantic_classify_multiple_routes(query_results)
        assert sorted(results) == sorted(
            expected
        ), "Should classify and return routes above their thresholds"

    def test_with_no_routes_passing_threshold(
        self, routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        # set threshold to 1.0 so that no routes pass
        route_layer.score_threshold = 1.0
        query_results = [
            {"route": "Route 1", "score": 0.01},
            {"route": "Route 2", "score": 0.02},
        ]
        expected = []
        results = route_layer._semantic_classify_multiple_routes(query_results)
        assert (
            results == expected
        ), "Should return an empty list when no routes pass their thresholds"

    def test_with_no_query_results(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        route_layer.score_threshold = 0.5
        query_results = []
        expected = []
        results = route_layer._semantic_classify_multiple_routes(query_results)
        assert (
            results == expected
        ), "Should return an empty list when there are no query results"

    def test_with_unrecognized_route(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        route_layer.score_threshold = 0.5
        # Test with a route name that does not exist in the route_layer's routes
        query_results = [{"route": "UnrecognizedRoute", "score": 0.9}]
        expected = []
        results = route_layer._semantic_classify_multiple_routes(query_results)
        assert results == expected, "Should ignore and not return unrecognized routes"

    def test_set_aggregation_method_with_unsupported_value(
        self, routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        unsupported_aggregation = "unsupported_aggregation_method"
        with pytest.raises(
            ValueError,
            match=f"Unsupported aggregation method chosen: {unsupported_aggregation}. Choose either 'SUM', 'MEAN', or 'MAX'.",
        ):
            route_layer._set_aggregation_method(unsupported_aggregation)

    def test_refresh_routes_not_implemented(
        self, routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        with pytest.raises(
            NotImplementedError, match="This method has not yet been implemented."
        ):
            route_layer._refresh_routes()

    def test_update_threshold(self, routes, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        route_name = "Route 1"
        new_threshold = 0.8
        route_layer.update(name=route_name, threshold=new_threshold)
        updated_route = route_layer.get(route_name)
        assert (
            updated_route.score_threshold == new_threshold
        ), f"Expected threshold to be updated to {new_threshold}, but got {updated_route.score_threshold}"

    def test_update_non_existent_route(
        self, routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        non_existent_route = "Non-existent Route"
        with pytest.raises(
            ValueError,
            match=f"Route '{non_existent_route}' not found. Nothing updated.",
        ):
            route_layer.update(name=non_existent_route, threshold=0.7)

    def test_update_without_parameters(
        self, routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        with pytest.raises(
            ValueError,
            match="At least one of 'threshold' or 'utterances' must be provided.",
        ):
            route_layer.update(name="Route 1")

    def test_update_utterances_not_implemented(
        self, routes, index_cls, encoder_cls, router_cls
    ):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(encoder=encoder, routes=routes, index=index)
        with pytest.raises(
            NotImplementedError,
            match="The update method cannot be used for updating utterances yet.",
        ):
            route_layer.update(name="Route 1", utterances=["New utterance"])


@pytest.mark.parametrize(
    "index_cls,encoder_cls,router_cls",
    [
        (index, encoder, router)
        for index in get_test_indexes()
        for encoder in [OpenAIEncoder]
        for router in get_test_routers()
    ],
)
class TestLayerFit:
    def test_eval(self, routes, test_data, index_cls, encoder_cls, router_cls):
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_is_ready():
            assert route_layer.index.is_ready()

        check_is_ready()
        # unpack test data
        X, y = zip(*test_data)
        # evaluate
        route_layer.evaluate(X=list(X), y=list(y), batch_size=int(len(X) / 5))

    def test_fit(self, routes, test_data, index_cls, encoder_cls, router_cls):
        # TODO: this is super slow for PineconeIndex, need to fix
        if index_cls is PineconeIndex:
            return
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_is_ready():
            assert route_layer.index.is_ready()

        check_is_ready()
        # unpack test data
        X, y = zip(*test_data)
        route_layer.fit(X=list(X), y=list(y), batch_size=int(len(X) / 5))

    def test_fit_local(self, routes, test_data, index_cls, encoder_cls, router_cls):
        # TODO: this is super slow for PineconeIndex, need to fix
        if index_cls is PineconeIndex:
            return
        encoder = encoder_cls()
        index = init_index(index_cls, index_name=encoder.__class__.__name__)
        route_layer = router_cls(
            encoder=encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_is_ready():
            assert route_layer.index.is_ready()

        check_is_ready()
        # unpack test data
        X, y = zip(*test_data)
        route_layer.fit(
            X=list(X), y=list(y), batch_size=int(len(X) / 5), local_execution=True
        )
