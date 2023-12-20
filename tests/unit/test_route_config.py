import os
from unittest.mock import mock_open, patch

import pytest

from semantic_router.route import Route, RouteConfig


class TestRouteConfig:
    def test_init(self):
        route_config = RouteConfig()
        assert route_config.routes == []

    def test_to_file_json(self):
        route = Route(name="test", utterances=["utterance"])
        route_config = RouteConfig(routes=[route])
        with patch("builtins.open", mock_open()) as mocked_open:
            route_config.to_file("data/test_output.json")
            mocked_open.assert_called_once_with("data/test_output.json", "w")

    def test_to_file_yaml(self):
        route = Route(name="test", utterances=["utterance"])
        route_config = RouteConfig(routes=[route])
        with patch("builtins.open", mock_open()) as mocked_open:
            route_config.to_file("data/test_output.yaml")
            mocked_open.assert_called_once_with("data/test_output.yaml", "w")

    def test_to_file_invalid(self):
        route = Route(name="test", utterances=["utterance"])
        route_config = RouteConfig(routes=[route])
        with pytest.raises(ValueError):
            route_config.to_file("test_output.txt")

    def test_from_file_json(self):
        mock_json_data = '[{"name": "test", "utterances": ["utterance"]}]'
        with patch("builtins.open", mock_open(read_data=mock_json_data)) as mocked_open:
            route_config = RouteConfig.from_file("data/test.json")
            mocked_open.assert_called_once_with("data/test.json", "r")
            assert isinstance(route_config, RouteConfig)

    def test_from_file_yaml(self):
        mock_yaml_data = "- name: test\n  utterances:\n  - utterance"
        with patch("builtins.open", mock_open(read_data=mock_yaml_data)) as mocked_open:
            route_config = RouteConfig.from_file("data/test.yaml")
            mocked_open.assert_called_once_with("data/test.yaml", "r")
            assert isinstance(route_config, RouteConfig)

    def test_from_file_invalid(self):
        with open("test.txt", "w") as f:
            f.write("dummy content")
        with pytest.raises(ValueError):
            RouteConfig.from_file("test.txt")
        os.remove("test.txt")

    def test_to_dict(self):
        route = Route(name="test", utterances=["utterance"])
        route_config = RouteConfig(routes=[route])
        assert route_config.to_dict() == [route.to_dict()]

    def test_add(self):
        route = Route(name="test", utterances=["utterance"])
        route_config = RouteConfig()
        route_config.add(route)
        assert route_config.routes == [route]

    def test_get(self):
        route = Route(name="test", utterances=["utterance"])
        route_config = RouteConfig(routes=[route])
        assert route_config.get("test") == route

    def test_get_not_found(self):
        route = Route(name="test", utterances=["utterance"])
        route_config = RouteConfig(routes=[route])
        assert route_config.get("not_found") is None

    def test_remove(self):
        route = Route(name="test", utterances=["utterance"])
        route_config = RouteConfig(routes=[route])
        route_config.remove("test")
        assert route_config.routes == []
