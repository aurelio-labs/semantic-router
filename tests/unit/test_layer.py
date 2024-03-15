import os
import tempfile
from unittest.mock import mock_open, patch

import pytest

from semantic_router.encoders import BaseEncoder, CohereEncoder, OpenAIEncoder
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
    return [mock_responses.get(u, [0, 0, 0]) for u in utterances]


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
            "function_schema": null
        },
        {
            "name": "chitchat",
            "utterances": [
                "how's the weather today?",
                "how are things going?"
            ],
            "description": null,
            "function_schema": null
        }
    ]
}"""


def layer_yaml():
    return """encoder_name: embed-english-v3.0
encoder_type: cohere
routes:
- description: null
  function_schema: null
  name: politics
  utterances:
  - isn't politics the best thing ever
  - why don't you tell me about your political opinions
- description: null
  function_schema: null
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
    return OpenAIEncoder(name="test-openai-encoder", openai_api_key="test_api_key")


@pytest.fixture
def routes():
    return [
        Route(name="Route 1", utterances=["Hello", "Hi"]),
        Route(name="Route 2", utterances=["Goodbye", "Bye", "Au revoir"]),
    ]


@pytest.fixture
def dynamic_routes():
    return [
        Route(
            name="Route 1", utterances=["Hello", "Hi"], function_schema={"name": "test"}
        ),
        Route(
            name="Route 2",
            utterances=["Goodbye", "Bye", "Au revoir"],
            function_schema={"name": "test"},
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


class TestRouteLayer:
    def test_initialization(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes, top_k=10)
        assert openai_encoder.score_threshold == 0.82
        assert route_layer.score_threshold == 0.82
        assert route_layer.top_k == 10
        assert len(route_layer.index) if route_layer.index is not None else 0 == 5
        assert (
            len(set(route_layer._get_route_names()))
            if route_layer._get_route_names() is not None
            else 0 == 2
        )

    def test_initialization_different_encoders(self, cohere_encoder, openai_encoder):
        route_layer_cohere = RouteLayer(encoder=cohere_encoder)
        assert cohere_encoder.score_threshold == 0.3
        assert route_layer_cohere.score_threshold == 0.3
        route_layer_openai = RouteLayer(encoder=openai_encoder)
        assert route_layer_openai.score_threshold == 0.82

    def test_initialization_no_encoder(self, openai_encoder):
        os.environ["OPENAI_API_KEY"] = "test_api_key"
        route_layer_none = RouteLayer(encoder=None)
        assert route_layer_none.score_threshold == openai_encoder.score_threshold

    def test_initialization_dynamic_route(
        self, cohere_encoder, openai_encoder, dynamic_routes
    ):
        route_layer_cohere = RouteLayer(encoder=cohere_encoder, routes=dynamic_routes)
        assert route_layer_cohere.score_threshold == 0.3
        route_layer_openai = RouteLayer(encoder=openai_encoder, routes=dynamic_routes)
        assert openai_encoder.score_threshold == 0.82
        assert route_layer_openai.score_threshold == 0.82

    def test_add_route(self, openai_encoder):
        route_layer = RouteLayer(encoder=openai_encoder)
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

    def test_list_route_names(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        route_names = route_layer.list_route_names()
        assert set(route_names) == {
            route.name for route in routes
        }, "The list of route names should match the names of the routes added."

    def test_delete_route(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
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

    def test_remove_route_not_found(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        # Attempt to remove a route that does not exist
        non_existent_route = "non-existent-route"
        with pytest.raises(ValueError) as excinfo:
            route_layer.delete(non_existent_route)
        assert (
            str(excinfo.value) == f"Route `{non_existent_route}` not found"
        ), "Attempting to remove a non-existent route should raise a ValueError."

    def test_add_multiple_routes(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder)
        route_layer._add_routes(routes=routes)
        assert route_layer.index is not None
        assert route_layer.index.describe()["vectors"] == 5

    def test_query_and_classification(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        query_result = route_layer(text="Hello").name
        assert query_result in ["Route 1", "Route 2"]

    def test_query_with_no_index(self, openai_encoder):
        route_layer = RouteLayer(encoder=openai_encoder)
        with pytest.raises(ValueError):
            assert route_layer(text="Anything").name is None

    def test_query_with_vector(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        vector = [0.1, 0.2, 0.3]
        query_result = route_layer(vector=vector).name
        assert query_result in ["Route 1", "Route 2"]

    def test_query_with_no_text_or_vector(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        with pytest.raises(ValueError):
            route_layer()

    def test_semantic_classify(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        classification, score = route_layer._semantic_classify(
            [
                {"route": "Route 1", "score": 0.9},
                {"route": "Route 2", "score": 0.1},
            ]
        )
        assert classification == "Route 1"
        assert score == [0.9]

    def test_semantic_classify_multiple_routes(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        classification, score = route_layer._semantic_classify(
            [
                {"route": "Route 1", "score": 0.9},
                {"route": "Route 2", "score": 0.1},
                {"route": "Route 1", "score": 0.8},
            ]
        )
        assert classification == "Route 1"
        assert score == [0.9, 0.8]

    def test_query_no_text_dynamic_route(self, openai_encoder, dynamic_routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=dynamic_routes)
        vector = [0.1, 0.2, 0.3]
        with pytest.raises(ValueError):
            route_layer(vector=vector)

    def test_pass_threshold(self, openai_encoder):
        route_layer = RouteLayer(encoder=openai_encoder)
        assert not route_layer._pass_threshold([], 0.5)
        assert route_layer._pass_threshold([0.6, 0.7], 0.5)

    def test_failover_score_threshold(self, base_encoder):
        route_layer = RouteLayer(encoder=base_encoder)
        assert route_layer.score_threshold == 0.5

    def test_json(self, openai_encoder, routes):
        temp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        try:
            temp_path = temp.name  # Save the temporary file's path
            temp.close()  # Close the file to ensure it can be opened again on Windows
            os.environ["OPENAI_API_KEY"] = "test_api_key"
            route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
            route_layer.to_json(temp_path)
            assert os.path.exists(temp_path)
            route_layer_from_file = RouteLayer.from_json(temp_path)
            assert (
                route_layer_from_file.index is not None
                and route_layer_from_file._get_route_names() is not None
            )
        finally:
            os.remove(temp_path)  # Ensure the file is deleted even if the test fails

    def test_yaml(self, openai_encoder, routes):
        temp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        try:
            temp_path = temp.name  # Save the temporary file's path
            temp.close()  # Close the file to ensure it can be opened again on Windows
            os.environ["OPENAI_API_KEY"] = "test_api_key"
            route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
            route_layer.to_yaml(temp_path)
            assert os.path.exists(temp_path)
            route_layer_from_file = RouteLayer.from_yaml(temp_path)
            assert (
                route_layer_from_file.index is not None
                and route_layer_from_file._get_route_names() is not None
            )
        finally:
            os.remove(temp_path)  # Ensure the file is deleted even if the test fails

    def test_from_file_json(openai_encoder, tmp_path):
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

    def test_from_file_yaml(openai_encoder, tmp_path):
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

    def test_from_file_invalid_path(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            LayerConfig.from_file("nonexistent_path.json")
        assert "[Errno 2] No such file or directory: 'nonexistent_path.json'" in str(
            excinfo.value
        )

    def test_from_file_unsupported_type(self, tmp_path):
        # Create a temporary unsupported file
        config_path = tmp_path / "config.unsupported"
        config_path.write_text(layer_json())

        with pytest.raises(ValueError) as excinfo:
            LayerConfig.from_file(str(config_path))
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
        with patch("semantic_router.layer.is_valid", return_value=False):
            # Attempt to load the LayerConfig from the temporary file
            # and assert that it raises an exception due to invalid configuration
            with pytest.raises(Exception) as excinfo:
                LayerConfig.from_file(str(config_path))
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

    def test_config(self, openai_encoder, routes):
        os.environ["OPENAI_API_KEY"] = "test_api_key"
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        # confirm route creation functions as expected
        layer_config = route_layer.to_config()
        assert layer_config.routes == routes
        # now load from config and confirm it's the same
        route_layer_from_config = RouteLayer.from_config(layer_config)
        assert (route_layer_from_config.index.index == route_layer.index.index).all()
        assert (
            route_layer_from_config._get_route_names() == route_layer._get_route_names()
        )
        assert route_layer_from_config.score_threshold == route_layer.score_threshold

    def test_get_thresholds(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        assert route_layer.get_thresholds() == {"Route 1": 0.82, "Route 2": 0.82}


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
