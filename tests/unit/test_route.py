import os
from unittest.mock import AsyncMock, mock_open, patch

import pytest

from semantic_router.route import Route, RouteConfig, is_valid


# Is valid test:
def test_is_valid_with_valid_json():
    valid_json = '{"name": "test_route", "utterances": ["hello", "hi"]}'
    assert is_valid(valid_json) is True


def test_is_valid_with_missing_keys():
    invalid_json = '{"name": "test_route"}'  # Missing 'utterances'
    with patch("semantic_router.route.logger") as mock_logger:
        assert is_valid(invalid_json) is False
        mock_logger.warning.assert_called_once()


def test_is_valid_with_valid_json_list():
    valid_json_list = (
        '[{"name": "test_route1", "utterances": ["hello"]}, '
        '{"name": "test_route2", "utterances": ["hi"]}]'
    )
    assert is_valid(valid_json_list) is True


def test_is_valid_with_invalid_json_list():
    invalid_json_list = (
        '[{"name": "test_route1"}, {"name": "test_route2", "utterances": ["hi"]}]'
    )
    with patch("semantic_router.route.logger") as mock_logger:
        assert is_valid(invalid_json_list) is False
        mock_logger.warning.assert_called_once()


def test_is_valid_with_invalid_json():
    invalid_json = '{"name": "test_route", "utterances": ["hello", "hi" invalid json}'
    with patch("semantic_router.route.logger") as mock_logger:
        assert is_valid(invalid_json) is False
        mock_logger.error.assert_called_once()


class TestRoute:
    @pytest.mark.asyncio
    @patch("semantic_router.route.llm", new_callable=AsyncMock)
    async def test_generate_dynamic_route(self, mock_llm):
        print(f"mock_llm: {mock_llm}")
        mock_llm.return_value = """
        <config>
        {
            "name": "test_function",
            "utterances": [
                "example_utterance_1",
                "example_utterance_2",
                "example_utterance_3",
                "example_utterance_4",
                "example_utterance_5"]
        }
        </config>
        """
        function_schema = {"name": "test_function", "type": "function"}
        route = await Route._generate_dynamic_route(function_schema)
        assert route.name == "test_function"
        assert route.utterances == [
            "example_utterance_1",
            "example_utterance_2",
            "example_utterance_3",
            "example_utterance_4",
            "example_utterance_5",
        ]

    def test_to_dict(self):
        route = Route(name="test", utterances=["utterance"])
        expected_dict = {
            "name": "test",
            "utterances": ["utterance"],
            "description": None,
        }
        assert route.to_dict() == expected_dict

    def test_from_dict(self):
        route_dict = {"name": "test", "utterances": ["utterance"]}
        route = Route.from_dict(route_dict)
        assert route.name == "test"
        assert route.utterances == ["utterance"]

    @pytest.mark.asyncio
    @patch("semantic_router.route.llm", new_callable=AsyncMock)
    async def test_from_dynamic_route(self, mock_llm):
        # Mock the llm function
        mock_llm.return_value = """
        <config>
        {
            "name": "test_function",
            "utterances": [
                "example_utterance_1",
                "example_utterance_2",
                "example_utterance_3",
                "example_utterance_4",
                "example_utterance_5"]
        }
        </config>
        """

        def test_function(input: str):
            """Test function docstring"""
            pass

        dynamic_route = await Route.from_dynamic_route(test_function)

        assert dynamic_route.name == "test_function"
        assert dynamic_route.utterances == [
            "example_utterance_1",
            "example_utterance_2",
            "example_utterance_3",
            "example_utterance_4",
            "example_utterance_5",
        ]

    def test_parse_route_config(self):
        config = """
        <config>
        {
            "name": "test_function",
            "utterances": [
                "example_utterance_1",
                "example_utterance_2",
                "example_utterance_3",
                "example_utterance_4",
                "example_utterance_5"]
        }
        </config>
        """
        expected_config = """
        {
            "name": "test_function",
            "utterances": [
                "example_utterance_1",
                "example_utterance_2",
                "example_utterance_3",
                "example_utterance_4",
                "example_utterance_5"]
        }
        """
        assert Route._parse_route_config(config).strip() == expected_config.strip()


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
