import importlib
import os
import tempfile
from unittest.mock import mock_open, patch

import pytest
import time

from semantic_router.encoders import BaseEncoder, CohereEncoder, OpenAIEncoder
from semantic_router.index.local import LocalIndex
from semantic_router.index.pinecone import PineconeIndex
from semantic_router.index.qdrant import QdrantIndex
from semantic_router.layer import LayerConfig, RouteLayer
from semantic_router.llms.base import BaseLLM
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
    return OpenAIEncoder(name="text-embedding-ada-002", openai_api_key="test_api_key")


@pytest.fixture
def routes():
    return [
        Route(name="Route 1", utterances=["Hello", "Hi"]),
        Route(name="Route 2", utterances=["Goodbye", "Bye", "Au revoir"]),
    ]


@pytest.fixture
def routes_2():
    return [
        Route(name="Route 1", utterances=["Hello"]),
        Route(name="Route 2", utterances=["Hello"]),
    ]


@pytest.fixture
def routes_3():
    return [
        Route(name="Route 1", utterances=["Hello"]),
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
    indexes = [LocalIndex]

    if importlib.util.find_spec("qdrant_client") is not None:
        indexes.append(QdrantIndex)
    return indexes


@pytest.mark.parametrize("index_cls", get_test_indexes())
class TestRouteLayer:
    def test_initialization(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, top_k=10, index=index_cls()
        )
        assert openai_encoder.score_threshold == 0.82
        assert route_layer.score_threshold == 0.82
        assert route_layer.top_k == 10
        assert len(route_layer.index) if route_layer.index is not None else 0 == 5
        assert (
            len(set(route_layer._get_route_names()))
            if route_layer._get_route_names() is not None
            else 0 == 2
        )

    def test_initialization_different_encoders(
        self, cohere_encoder, openai_encoder, index_cls
    ):
        route_layer_cohere = RouteLayer(encoder=cohere_encoder, index=index_cls())
        assert cohere_encoder.score_threshold == 0.3
        assert route_layer_cohere.score_threshold == 0.3
        route_layer_openai = RouteLayer(encoder=openai_encoder, index=index_cls())
        assert route_layer_openai.score_threshold == 0.82

    def test_initialization_no_encoder(self, openai_encoder, index_cls):
        os.environ["OPENAI_API_KEY"] = "test_api_key"
        route_layer_none = RouteLayer(encoder=None)
        assert route_layer_none.score_threshold == openai_encoder.score_threshold

    def test_initialization_dynamic_route(
        self, cohere_encoder, openai_encoder, dynamic_routes, index_cls
    ):
        route_layer_cohere = RouteLayer(
            encoder=cohere_encoder, routes=dynamic_routes, index=index_cls()
        )
        assert route_layer_cohere.score_threshold == 0.3
        route_layer_openai = RouteLayer(
            encoder=openai_encoder, routes=dynamic_routes, index=index_cls()
        )
        assert openai_encoder.score_threshold == 0.82
        assert route_layer_openai.score_threshold == 0.82

    def test_add_route(self, openai_encoder, index_cls):
        route_layer = RouteLayer(encoder=openai_encoder, index=index_cls())
        route1 = Route(name="Route 1", utterances=["Yes", "No"])
        route2 = Route(name="Route 2", utterances=["Maybe", "Sure"])

        # Initially, the routes list should be empty
        assert route_layer.routes == []

        # Add route1 and check
        route_layer.add(route=route1)
        assert route_layer.routes == [route1]
        assert route_layer.index is not None
        # Use the describe method to get the number of vectors
        assert route_layer.index.describe()["vectors"] == 2

        # Add route2 and check
        route_layer.add(route=route2)
        assert route_layer.routes == [route1, route2]
        assert route_layer.index.describe()["vectors"] == 4

    def test_list_route_names(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        route_names = route_layer.list_route_names()
        assert set(route_names) == {
            route.name for route in routes
        }, "The list of route names should match the names of the routes added."

    def test_delete_route(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        # Delete a route by name
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

    def test_remove_route_not_found(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        # Attempt to remove a route that does not exist
        non_existent_route = "non-existent-route"
        route_layer.delete(non_existent_route)
        # we should see warning in logs only (ie no errors)

    def test_add_multiple_routes(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(encoder=openai_encoder, index=index_cls())
        route_layer._add_routes(routes=routes)
        assert route_layer.index is not None
        assert route_layer.index.describe()["vectors"] == 5

    def test_query_and_classification(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        query_result = route_layer(text="Hello").name
        assert query_result in ["Route 1", "Route 2"]

    def test_query_filter(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        query_result = route_layer(text="Hello", route_filter=["Route 1"]).name

        try:
            route_layer(text="Hello", route_filter=["Route 8"]).name
        except ValueError:
            assert True

        assert query_result in ["Route 1"]

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_query_filter_pinecone(self, openai_encoder, routes, index_cls):
        if type(index_cls) == PineconeIndex:
            pinecone_api_key = os.environ["PINECONE_API_KEY"]
            pineconeindex = PineconeIndex(api_key=pinecone_api_key)
            route_layer = RouteLayer(
                encoder=openai_encoder, routes=routes, index=pineconeindex
            )
            time.sleep(10)  # allow for index to be populated
            query_result = route_layer(text="Hello", route_filter=["Route 1"]).name

            try:
                route_layer(text="Hello", route_filter=["Route 8"]).name
            except ValueError:
                assert True

            assert query_result in ["Route 1"]

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_namespace_pinecone_index(self, openai_encoder, routes, index_cls):
        if type(index_cls) == PineconeIndex:
            pinecone_api_key = os.environ["PINECONE_API_KEY"]
            pineconeindex = PineconeIndex(api_key=pinecone_api_key, namespace="test")
            route_layer = RouteLayer(
                encoder=openai_encoder, routes=routes, index=pineconeindex
            )
            time.sleep(10)  # allow for index to be populated
            query_result = route_layer(text="Hello", route_filter=["Route 1"]).name

            try:
                route_layer(text="Hello", route_filter=["Route 8"]).name
            except ValueError:
                assert True

            assert query_result in ["Route 1"]

    def test_query_with_no_index(self, openai_encoder, index_cls):
        route_layer = RouteLayer(encoder=openai_encoder, index=index_cls())
        with pytest.raises(ValueError):
            assert route_layer(text="Anything").name is None

    def test_query_with_vector(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        vector = [0.1, 0.2, 0.3]
        query_result = route_layer(vector=vector).name
        assert query_result in ["Route 1", "Route 2"]

    def test_query_with_no_text_or_vector(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        with pytest.raises(ValueError):
            route_layer()

    def test_semantic_classify(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        classification, score = route_layer._semantic_classify(
            [
                {"route": "Route 1", "score": 0.9},
                {"route": "Route 2", "score": 0.1},
            ]
        )
        assert classification == "Route 1"
        assert score == [0.9]

    def test_semantic_classify_multiple_routes(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
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
        self, openai_encoder, dynamic_routes, index_cls
    ):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=dynamic_routes, index=index_cls()
        )
        vector = [0.1, 0.2, 0.3]
        with pytest.raises(ValueError):
            route_layer(vector=vector)

    def test_pass_threshold(self, openai_encoder, index_cls):
        route_layer = RouteLayer(encoder=openai_encoder, index=index_cls())
        assert not route_layer._pass_threshold([], 0.5)
        assert route_layer._pass_threshold([0.6, 0.7], 0.5)

    def test_failover_score_threshold(self, base_encoder, index_cls):
        route_layer = RouteLayer(encoder=base_encoder, index=index_cls())
        assert route_layer.score_threshold == 0.5

    def test_json(self, openai_encoder, routes, index_cls):
        temp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        try:
            temp_path = temp.name  # Save the temporary file's path
            temp.close()  # Close the file to ensure it can be opened again on Windows
            os.environ["OPENAI_API_KEY"] = "test_api_key"
            route_layer = RouteLayer(
                encoder=openai_encoder, routes=routes, index=index_cls()
            )
            route_layer.to_json(temp_path)
            assert os.path.exists(temp_path)
            route_layer_from_file = RouteLayer.from_json(temp_path)
            assert (
                route_layer_from_file.index is not None
                and route_layer_from_file._get_route_names() is not None
            )
        finally:
            os.remove(temp_path)  # Ensure the file is deleted even if the test fails

    def test_yaml(self, openai_encoder, routes, index_cls):
        temp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        try:
            temp_path = temp.name  # Save the temporary file's path
            temp.close()  # Close the file to ensure it can be opened again on Windows
            os.environ["OPENAI_API_KEY"] = "test_api_key"
            route_layer = RouteLayer(
                encoder=openai_encoder, routes=routes, index=index_cls()
            )
            route_layer.to_yaml(temp_path)
            assert os.path.exists(temp_path)
            route_layer_from_file = RouteLayer.from_yaml(temp_path)
            assert (
                route_layer_from_file.index is not None
                and route_layer_from_file._get_route_names() is not None
            )
        finally:
            os.remove(temp_path)  # Ensure the file is deleted even if the test fails

    def test_from_file_json(openai_encoder, tmp_path, index_cls):
        # Create a temporary JSON file with layer configuration
        config_path = tmp_path / "config.json"
        config_path.write_text(
            layer_json()
        )  # Assuming layer_json() returns a valid JSON string

        # Load the LayerConfig from the temporary file
        layer_config = LayerConfig.from_file(str(config_path))

        # Assertions to verify the loaded configuration
        assert layer_config.encoder_type == "cohere"
        assert layer_config.encoder_name == "embed-english-v3.0"
        assert len(layer_config.routes) == 2
        assert layer_config.routes[0].name == "politics"

    def test_from_file_yaml(openai_encoder, tmp_path, index_cls):
        # Create a temporary YAML file with layer configuration
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            layer_yaml()
        )  # Assuming layer_yaml() returns a valid YAML string

        # Load the LayerConfig from the temporary file
        layer_config = LayerConfig.from_file(str(config_path))

        # Assertions to verify the loaded configuration
        assert layer_config.encoder_type == "cohere"
        assert layer_config.encoder_name == "embed-english-v3.0"
        assert len(layer_config.routes) == 2
        assert layer_config.routes[0].name == "politics"

    def test_from_file_invalid_path(self, index_cls):
        with pytest.raises(FileNotFoundError) as excinfo:
            LayerConfig.from_file("nonexistent_path.json")
        assert "[Errno 2] No such file or directory: 'nonexistent_path.json'" in str(
            excinfo.value
        )

    def test_from_file_unsupported_type(self, tmp_path, index_cls):
        # Create a temporary unsupported file
        config_path = tmp_path / "config.unsupported"
        config_path.write_text(layer_json())

        with pytest.raises(ValueError) as excinfo:
            LayerConfig.from_file(str(config_path))
        assert "Unsupported file type" in str(excinfo.value)

    def test_from_file_invalid_config(self, tmp_path, index_cls):
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
        with patch("semantic_router.layer.is_valid", return_value=False):
            # Attempt to load the LayerConfig from the temporary file
            # and assert that it raises an exception due to invalid configuration
            with pytest.raises(Exception) as excinfo:
                LayerConfig.from_file(str(config_path))
            assert "Invalid config JSON or YAML" in str(
                excinfo.value
            ), "Loading an invalid configuration should raise an exception."

    def test_from_file_with_llm(self, tmp_path, index_cls):
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

        # Load the LayerConfig from the temporary file
        layer_config = LayerConfig.from_file(str(config_path))

        # Using BaseLLM because trying to create a usable Mock LLM is a nightmare.
        assert isinstance(
            layer_config.routes[0].llm, BaseLLM
        ), "LLM should be instantiated and associated with the route based on the "
        "config"
        assert (
            layer_config.routes[0].llm.name == "fake-model-v1"
        ), "LLM instance should have the 'name' attribute set correctly"

    def test_config(self, openai_encoder, routes, index_cls):
        os.environ["OPENAI_API_KEY"] = "test_api_key"
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        # confirm route creation functions as expected
        layer_config = route_layer.to_config()
        assert layer_config.routes == routes
        # now load from config and confirm it's the same
        route_layer_from_config = RouteLayer.from_config(layer_config, index_cls())
        assert (
            route_layer_from_config._get_route_names() == route_layer._get_route_names()
        )
        assert route_layer_from_config.score_threshold == route_layer.score_threshold

    def test_get_thresholds(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        assert route_layer.get_thresholds() == {"Route 1": 0.82, "Route 2": 0.82}

    def test_with_multiple_routes_passing_threshold(
        self, openai_encoder, routes, index_cls
    ):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        route_layer.score_threshold = 0.5  # Set the score_threshold if needed
        # Assuming route_layer is already set up with routes "Route 1" and "Route 2"
        query_results = [
            {"route": "Route 1", "score": 0.6},
            {"route": "Route 2", "score": 0.7},
            {"route": "Route 1", "score": 0.8},
        ]
        # Override _pass_threshold to always return True for this test
        route_layer._pass_threshold = lambda scores, threshold: True
        expected = [("Route 1", 0.8), ("Route 2", 0.7)]
        results = route_layer._semantic_classify_multiple_routes(query_results)
        assert sorted(results) == sorted(
            expected
        ), "Should classify and return routes above their thresholds"

    def test_with_no_routes_passing_threshold(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        route_layer.score_threshold = 0.5
        # Override _pass_threshold to always return False for this test
        route_layer._pass_threshold = lambda scores, threshold: False
        query_results = [
            {"route": "Route 1", "score": 0.3},
            {"route": "Route 2", "score": 0.2},
        ]
        expected = []
        results = route_layer._semantic_classify_multiple_routes(query_results)
        assert (
            results == expected
        ), "Should return an empty list when no routes pass their thresholds"

    def test_with_no_query_results(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        route_layer.score_threshold = 0.5
        query_results = []
        expected = []
        results = route_layer._semantic_classify_multiple_routes(query_results)
        assert (
            results == expected
        ), "Should return an empty list when there are no query results"

    def test_with_unrecognized_route(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        route_layer.score_threshold = 0.5
        # Test with a route name that does not exist in the route_layer's routes
        query_results = [{"route": "UnrecognizedRoute", "score": 0.9}]
        expected = []
        results = route_layer._semantic_classify_multiple_routes(query_results)
        assert results == expected, "Should ignore and not return unrecognized routes"

    def test_retrieve_with_text(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        text = "Hello"
        results = route_layer.retrieve_multiple_routes(text=text)
        assert len(results) >= 1, "Expected at least one result"
        assert any(
            result.name in ["Route 1", "Route 2"] for result in results
        ), "Expected the result to be either 'Route 1' or 'Route 2'"

    def test_retrieve_with_vector(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        vector = [0.1, 0.2, 0.3]
        results = route_layer.retrieve_multiple_routes(vector=vector)
        assert len(results) >= 1, "Expected at least one result"
        assert any(
            result.name in ["Route 1", "Route 2"] for result in results
        ), "Expected the result to be either 'Route 1' or 'Route 2'"

    def test_retrieve_without_text_or_vector(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        with pytest.raises(ValueError, match="Either text or vector must be provided"):
            route_layer.retrieve_multiple_routes()

    def test_retrieve_no_matches(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        text = "Asparagus"
        results = route_layer.retrieve_multiple_routes(text=text)
        assert len(results) == 0, f"Expected no results, but got {len(results)}"

    def test_retrieve_one_match(self, openai_encoder, routes_3, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes_3, index=index_cls()
        )
        text = "Hello"
        results = route_layer.retrieve_multiple_routes(text=text)
        assert len(results) == 1, f"Expected one result, and got {len(results)}"
        matched_routes = [result.name for result in results]
        assert "Route 1" in matched_routes, "Expected 'Route 1' to be a match"

    def test_retrieve_with_text_for_multiple_matches(
        self, openai_encoder, routes_2, index_cls
    ):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes_2, index=index_cls()
        )
        text = "Hello"
        results = route_layer.retrieve_multiple_routes(text=text)
        assert len(results) == 2, "Expected two results"
        matched_routes = [result.name for result in results]
        assert "Route 1" in matched_routes, "Expected 'Route 1' to be a match"
        assert "Route 2" in matched_routes, "Expected 'Route 2' to be a match"

    def test_set_aggregation_method_with_unsupported_value(
        self, openai_encoder, routes, index_cls
    ):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        unsupported_aggregation = "unsupported_aggregation_method"
        with pytest.raises(
            ValueError,
            match=f"Unsupported aggregation method chosen: {unsupported_aggregation}. Choose either 'SUM', 'MEAN', or 'MAX'.",
        ):
            route_layer._set_aggregation_method(unsupported_aggregation)

    def test_refresh_routes_not_implemented(self, openai_encoder, routes, index_cls):
        route_layer = RouteLayer(
            encoder=openai_encoder, routes=routes, index=index_cls()
        )
        with pytest.raises(
            NotImplementedError, match="This method has not yet been implemented."
        ):
            route_layer._refresh_routes()


class TestLayerFit:
    def test_eval(self, openai_encoder, routes, test_data):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        # unpack test data
        X, y = zip(*test_data)
        # evaluate
        route_layer.evaluate(X=X, y=y, batch_size=int(len(test_data) / 5))

    def test_fit(self, openai_encoder, routes, test_data):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        # unpack test data
        X, y = zip(*test_data)
        route_layer.fit(X=X, y=y, batch_size=int(len(test_data) / 5))


# Add more tests for edge cases and error handling as needed.


class TestLayerConfig:
    def test_init(self):
        layer_config = LayerConfig()
        assert layer_config.routes == []

    def test_to_file_json(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = LayerConfig(routes=[route])
        with patch("builtins.open", mock_open()) as mocked_open:
            layer_config.to_file("data/test_output.json")
            mocked_open.assert_called_once_with("data/test_output.json", "w")

    def test_to_file_yaml(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = LayerConfig(routes=[route])
        with patch("builtins.open", mock_open()) as mocked_open:
            layer_config.to_file("data/test_output.yaml")
            mocked_open.assert_called_once_with("data/test_output.yaml", "w")

    def test_to_file_invalid(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = LayerConfig(routes=[route])
        with pytest.raises(ValueError):
            layer_config.to_file("test_output.txt")

    def test_from_file_json(self):
        mock_json_data = layer_json()
        with patch("builtins.open", mock_open(read_data=mock_json_data)) as mocked_open:
            layer_config = LayerConfig.from_file("data/test.json")
            mocked_open.assert_called_once_with("data/test.json", "r")
            assert isinstance(layer_config, LayerConfig)

    def test_from_file_yaml(self):
        mock_yaml_data = layer_yaml()
        with patch("builtins.open", mock_open(read_data=mock_yaml_data)) as mocked_open:
            layer_config = LayerConfig.from_file("data/test.yaml")
            mocked_open.assert_called_once_with("data/test.yaml", "r")
            assert isinstance(layer_config, LayerConfig)

    def test_from_file_invalid(self):
        with open("test.txt", "w") as f:
            f.write("dummy content")
        with pytest.raises(ValueError):
            LayerConfig.from_file("test.txt")
        os.remove("test.txt")

    def test_to_dict(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = LayerConfig(routes=[route])
        assert layer_config.to_dict()["routes"] == [route.to_dict()]

    def test_add(self):
        route = Route(name="test", utterances=["utterance"])
        route2 = Route(name="test2", utterances=["utterance2"])
        layer_config = LayerConfig()
        layer_config.add(route)
        # confirm route added
        assert layer_config.routes == [route]
        # add second route and check updates
        layer_config.add(route2)
        assert layer_config.routes == [route, route2]

    def test_get(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = LayerConfig(routes=[route])
        assert layer_config.get("test") == route

    def test_get_not_found(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = LayerConfig(routes=[route])
        assert layer_config.get("not_found") is None

    def test_remove(self):
        route = Route(name="test", utterances=["utterance"])
        layer_config = LayerConfig(routes=[route])
        layer_config.remove("test")
        assert layer_config.routes == []

    def test_setting_aggregation_methods(self, openai_encoder, routes):
        for agg in ["sum", "mean", "max"]:
            route_layer = RouteLayer(
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
            route_layer = RouteLayer(
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
