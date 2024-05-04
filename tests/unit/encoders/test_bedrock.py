import pytest
import json
from io import BytesIO
from semantic_router.encoders import BedrockEncoder


@pytest.fixture
def bedrock_encoder(mocker):
    mocker.patch("boto3.Session")
    mocker.patch("boto3.Session.client")
    return BedrockEncoder()


class TestBedrockEncoder:
    def test_initialisation_with_default_values(self, bedrock_encoder):
        assert bedrock_encoder.client is not None, "Client should be initialised"
        assert bedrock_encoder.type == "bedrock", "Default type not set correctly"
        assert (
            bedrock_encoder.input_type == "search_query"
        ), "Default input type not set correctly"
        assert bedrock_encoder.session is not None, "Session should be initialised"
        assert bedrock_encoder.region is not None, "Region should be initialised"

    def test_initialisation_with_custom_values(self, mocker):
        mocker.patch("boto3.Session")
        mocker.patch("boto3.Session.client")
        name = "custom_model"
        session = mocker.Mock()
        region = "us-west-2"
        score_threshold = 0.5
        input_type = "custom_input"
        bedrock_encoder = BedrockEncoder(
            name=name,
            session=session,
            region=region,
            score_threshold=score_threshold,
            input_type=input_type,
        )
        assert bedrock_encoder.name == name, "Custom name not set correctly"
        assert bedrock_encoder.session == session, "Custom session not set correctly"
        assert bedrock_encoder.region == region, "Custom region not set correctly"
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

    def test_returns_list_of_embeddings_for_valid_input(self, bedrock_encoder):
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
        mocker.patch("boto3.Session.client", return_value=None)
        with pytest.raises(ValueError):
            BedrockEncoder()

    def test_raises_value_error_if_call_to_bedrock_fails(self, bedrock_encoder):
        bedrock_encoder.client.invoke_model.side_effect = Exception(
            "Bedrock call failed."
        )
        with pytest.raises(ValueError):
            bedrock_encoder(["test"])

    def test_raises_value_error_if_no_aws_session_credentials(self, mocker):
        mocker.patch("boto3.Session")
        mock_session = mocker.Mock()
        mock_session.get_credentials.return_value = None
        with pytest.raises(ValueError, match="Could not get AWS session"):
            BedrockEncoder(session=mock_session)

    def test_raises_value_error_if_no_aws_region(self, mocker):
        mocker.patch("boto3.Session")
        mock_session = mocker.Mock()
        mock_session.region_name = None
        with pytest.raises(ValueError, match="No AWS region provided"):
            BedrockEncoder(session=mock_session)

    def test_raises_value_error_if_client_initialisation_fails(self, mocker):
        mocker.patch("boto3.Session")
        mock_session = mocker.Mock()
        mock_session.client.side_effect = Exception("Client initialisation failed")
        with pytest.raises(ValueError, match="Bedrock client failed to initialise"):
            BedrockEncoder(session=mock_session)

    def test_raises_value_error_for_unknown_model_name(self, mocker):
        mocker.patch("boto3.Session")
        mock_session = mocker.Mock()
        mock_session.get_credentials.return_value = True
        mocker.patch("boto3.Session.client")

        unknown_model_name = "unknown_model"
        bedrock_encoder = BedrockEncoder(
            name=unknown_model_name,
            session=mock_session,
            region="us-west-2",
        )

        with pytest.raises(ValueError, match="Unknown model name"):
            bedrock_encoder(["test"])


@pytest.fixture
def bedrock_encoder_with_cohere(mocker):
    mocker.patch("boto3.Session")
    mock_session = mocker.Mock()
    mock_session.get_credentials.return_value = True
    mocker.patch("boto3.Session.client")
    return BedrockEncoder(name="cohere_model", session=mock_session, region="us-west-2")


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
