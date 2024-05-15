import pytest
import json
from io import BytesIO
from semantic_router.encoders import BedrockEncoder


@pytest.fixture
def bedrock_encoder(mocker):
    mocker.patch("semantic_router.encoders.bedrock.BedrockEncoder._initialize_client")
    return BedrockEncoder(
        access_key_id="fake_id",
        secret_access_key="fake_secret",
        session_token="fake_token",
        region="us-west-2",
    )


class TestBedrockEncoder:
    def test_initialisation_with_default_values(self, bedrock_encoder):
        assert (
            bedrock_encoder.input_type == "search_query"
        ), "Default input type not set correctly"
        assert bedrock_encoder.region == "us-west-2", "Region should be initialised"

    def test_initialisation_with_custom_values(self, mocker):
        # mocker.patch(
        #     "semantic_router.encoders.bedrock.BedrockEncoder._initialize_client"
        # )
        name = "custom_model"
        score_threshold = 0.5
        input_type = "custom_input"
        bedrock_encoder = BedrockEncoder(
            name=name,
            score_threshold=score_threshold,
            input_type=input_type,
            access_key_id="fake_id",
            secret_access_key="fake_secret",
            session_token="fake_token",
            region="us-west-2",
        )
        assert bedrock_encoder.name == name, "Custom name not set correctly"
        assert bedrock_encoder.region == "us-west-2", "Custom region not set correctly"
        assert (
            bedrock_encoder.score_threshold == score_threshold
        ), "Custom score threshold not set correctly"
        assert (
            bedrock_encoder.input_type == input_type
        ), "Custom input type not set correctly"

    def test_call_method(self, bedrock_encoder):
        response_content = json.dumps({"embedding": [0.1, 0.2, 0.3]})
        response_body = BytesIO(response_content.encode("utf-8"))
        mock_response = {"body": response_body}
        bedrock_encoder.client.invoke_model.return_value = mock_response
        result = bedrock_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(item, list) for item in result
        ), "Each item in result should be a list"
        assert result == [[0.1, 0.2, 0.3]], "Embedding should be [0.1, 0.2, 0.3]"

    def test_raises_value_error_if_client_is_not_initialised(self, mocker):
        mocker.patch(
            "semantic_router.encoders.bedrock.BedrockEncoder._initialize_client",
            side_effect=Exception("Client initialisation failed"),
        )
        with pytest.raises(ValueError):
            BedrockEncoder(
                access_key_id="fake_id",
                secret_access_key="fake_secret",
                session_token="fake_token",
                region="us-west-2",
            )

    def test_raises_value_error_if_call_to_bedrock_fails(self, bedrock_encoder):
        bedrock_encoder.client.invoke_model.side_effect = Exception(
            "Bedrock call failed."
        )
        with pytest.raises(ValueError):
            bedrock_encoder(["test"])


@pytest.fixture
def bedrock_encoder_with_cohere(mocker):
    mocker.patch("semantic_router.encoders.bedrock.BedrockEncoder._initialize_client")
    return BedrockEncoder(
        name="cohere_model",
        access_key_id="fake_id",
        secret_access_key="fake_secret",
        session_token="fake_token",
        region="us-west-2",
    )


class TestBedrockEncoderWithCohere:
    def test_cohere_embedding_single_chunk(self, bedrock_encoder_with_cohere):
        response_content = json.dumps({"embeddings": [[0.1, 0.2, 0.3]]})
        response_body = BytesIO(response_content.encode("utf-8"))
        mock_response = {"body": response_body}
        bedrock_encoder_with_cohere.client.invoke_model.return_value = mock_response
        result = bedrock_encoder_with_cohere(["short test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(item, list) for item in result
        ), "Each item should be a list"
        assert result == [[0.1, 0.2, 0.3]], "Expected embedding [0.1, 0.2, 0.3]"

    def test_cohere_input_type(self, bedrock_encoder_with_cohere):
        bedrock_encoder_with_cohere.input_type = "different_type"
        response_content = json.dumps({"embeddings": [[0.1, 0.2, 0.3]]})
        response_body = BytesIO(response_content.encode("utf-8"))
        mock_response = {"body": response_body}
        bedrock_encoder_with_cohere.client.invoke_model.return_value = mock_response
        result = bedrock_encoder_with_cohere(["test with different input type"])
        assert isinstance(result, list), "Result should be a list"
        assert result == [[0.1, 0.2, 0.3]], "Expected specific embeddings"
