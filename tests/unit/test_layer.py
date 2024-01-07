import os
import tempfile
from unittest.mock import mock_open, patch

import pytest

from semantic_router.encoders import BaseEncoder, CohereEncoder, OpenAIEncoder
from semantic_router.layer import LayerConfig, RouteLayer
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
        Route(name="Route 1", utterances=["Hello", "Hi"], function_schema="test"),
        Route(name="Route 2", utterances=["Goodbye", "Bye", "Au revoir"]),
    ]


class TestRouteLayer:
    def test_initialization(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        assert openai_encoder.score_threshold == 0.82
        assert route_layer.score_threshold == 0.82
        assert len(route_layer.index) if route_layer.index is not None else 0 == 5
        assert (
            len(set(route_layer.categories))
            if route_layer.categories is not None
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

    def test_initialization_dynamic_route(self, cohere_encoder, openai_encoder):
        route_layer_cohere = RouteLayer(encoder=cohere_encoder)
        assert route_layer_cohere.score_threshold == 0.3
        route_layer_openai = RouteLayer(encoder=openai_encoder)
        assert openai_encoder.score_threshold == 0.82
        assert route_layer_openai.score_threshold == 0.82

    def test_add_route(self, openai_encoder):
        route_layer = RouteLayer(encoder=openai_encoder)
        route1 = Route(name="Route 1", utterances=["Yes", "No"])
        route2 = Route(name="Route 2", utterances=["Maybe", "Sure"])

        route_layer.add(route=route1)
        assert route_layer.index is not None and route_layer.categories is not None
        assert route_layer.index.shape[0] == 2
        assert len(set(route_layer.categories)) == 1
        assert set(route_layer.categories) == {"Route 1"}

        route_layer.add(route=route2)
        assert route_layer.index.shape[0] == 4
        assert len(set(route_layer.categories)) == 2
        assert set(route_layer.categories) == {"Route 1", "Route 2"}
        del route_layer

    def test_add_multiple_routes(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder)
        route_layer._add_routes(routes=routes)
        assert route_layer.index is not None and route_layer.categories is not None
        assert route_layer.index.shape[0] == 5
        assert len(set(route_layer.categories)) == 2

    def test_query_and_classification(self, openai_encoder, routes):
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        query_result = route_layer("Hello").name
        assert query_result in ["Route 1", "Route 2"]

    def test_query_with_no_index(self, openai_encoder):
        route_layer = RouteLayer(encoder=openai_encoder)
        assert route_layer("Anything").name is None

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

    def test_pass_threshold(self, openai_encoder):
        route_layer = RouteLayer(encoder=openai_encoder)
        assert not route_layer._pass_threshold([], 0.5)
        assert route_layer._pass_threshold([0.6, 0.7], 0.5)

    def test_failover_score_threshold(self, base_encoder):
        route_layer = RouteLayer(encoder=base_encoder)
        assert route_layer.score_threshold == 0.5

    def test_json(self, openai_encoder, routes):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
            os.environ["OPENAI_API_KEY"] = "test_api_key"
            route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
            route_layer.to_json(temp.name)
            assert os.path.exists(temp.name)
            route_layer_from_file = RouteLayer.from_json(temp.name)
            assert (
                route_layer_from_file.index is not None
                and route_layer_from_file.categories is not None
            )
            os.remove(temp.name)

    def test_yaml(self, openai_encoder, routes):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
            os.environ["OPENAI_API_KEY"] = "test_api_key"
            route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
            route_layer.to_yaml(temp.name)
            assert os.path.exists(temp.name)
            route_layer_from_file = RouteLayer.from_yaml(temp.name)
            assert (
                route_layer_from_file.index is not None
                and route_layer_from_file.categories is not None
            )
            os.remove(temp.name)

    def test_config(self, openai_encoder, routes):
        os.environ["OPENAI_API_KEY"] = "test_api_key"
        route_layer = RouteLayer(encoder=openai_encoder, routes=routes)
        # confirm route creation functions as expected
        layer_config = route_layer.to_config()
        assert layer_config.routes == routes
        # now load from config and confirm it's the same
        route_layer_from_config = RouteLayer.from_config(layer_config)
        assert (route_layer_from_config.index == route_layer.index).all()
        assert (route_layer_from_config.categories == route_layer.categories).all()
        assert route_layer_from_config.score_threshold == route_layer.score_threshold


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
