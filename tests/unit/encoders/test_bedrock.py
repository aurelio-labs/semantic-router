import json
import os
from io import BytesIO

import pytest

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


class TestBedrockEncoder:
    def test_initialisation_with_default_values(self, bedrock_encoder):
        assert bedrock_encoder.input_type == "search_query", (
            "Default input type not set correctly"
        )
        assert bedrock_encoder.region == "us-west-2", "Region should be initialised"

    def test_initialisation_with_custom_values(self, mocker):
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
        assert bedrock_encoder.score_threshold == score_threshold, (
            "Custom score threshold not set correctly"
        )
        assert bedrock_encoder.input_type == input_type, (
            "Custom input type not set correctly"
        )

    def test_initialisation_with_session_token(self, mocker):
        mocker.patch(
            "semantic_router.encoders.bedrock.BedrockEncoder._initialize_client"
        )
        bedrock_encoder = BedrockEncoder(
            access_key_id="fake_id",
            secret_access_key="fake_secret",
            session_token="fake_token",
            region="us-west-2",
        )
        assert bedrock_encoder.session_token == "fake_token", (
            "Session token not set correctly"
        )

    def test_initialisation_with_missing_access_key(self, mocker):
        mocker.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "env_id"})
        mocker.patch(
            "semantic_router.encoders.bedrock.BedrockEncoder._initialize_client"
        )
        bedrock_encoder = BedrockEncoder(
            access_key_id=None,
            secret_access_key="fake_secret",
            session_token="fake_token",
            region="us-west-2",
        )
        assert bedrock_encoder.access_key_id == "env_id", (
            "Access key ID not set correctly from environment variable"
        )

    def test_missing_access_key_id(self, mocker):
        mocker.patch(
            "semantic_router.encoders.bedrock.BedrockEncoder._initialize_client"
        )

        with pytest.raises(ValueError):
            BedrockEncoder(access_key_id=None, secret_access_key="fake_secret")

    def test_missing_secret_access_key(self, mocker):
        mocker.patch(
            "semantic_router.encoders.bedrock.BedrockEncoder._initialize_client"
        )

        with pytest.raises(ValueError):
            BedrockEncoder(access_key_id="fake_id", secret_access_key=None)

    def test_initialisation_missing_env_variables(self, mocker):
        mocker.patch.dict(os.environ, {}, clear=True)
        with pytest.raises(ValueError):
            BedrockEncoder(
                access_key_id=None,
                secret_access_key=None,
                session_token=None,
                region=None,
            )

    def test_failed_client_initialisation(self, mocker):
        mocker.patch.dict(os.environ, clear=True)

        mocker.patch(
            "semantic_router.encoders.bedrock.BedrockEncoder._initialize_client",
            side_effect=Exception("Initialization failed"),
        )

        with pytest.raises(ValueError):
            BedrockEncoder(access_key_id="fake_id", secret_access_key="fake_secret")

    def test_call_method(self, bedrock_encoder):
        response_content = json.dumps({"embedding": [0.1, 0.2, 0.3]})
        response_body = BytesIO(response_content.encode("utf-8"))
        mock_response = {"body": response_body}
        bedrock_encoder.client.invoke_model.return_value = mock_response
        result = bedrock_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(item, list) for item in result), (
            "Each item in result should be a list"
        )
        assert result == [[0.1, 0.2, 0.3]], "Embedding should be [0.1, 0.2, 0.3]"

    def test_call_with_expired_token(self, mocker, bedrock_encoder):
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "ExpiredTokenException"}}
        mocker.patch(
            "semantic_router.encoders.bedrock.BedrockEncoder._initialize_client",
            return_value=None,
        )

        def invoke_model_side_effect(*args, **kwargs):
            if not invoke_model_side_effect.expired_token_raised:
                invoke_model_side_effect.expired_token_raised = True
                raise ClientError(error_response, "invoke_model")
            else:
                return {
                    "body": BytesIO(
                        json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode("utf-8")
                    )
                }

        invoke_model_side_effect.expired_token_raised = False
        bedrock_encoder.client.invoke_model.side_effect = invoke_model_side_effect

        with pytest.raises(ValueError):
            bedrock_encoder(["test"])

        bedrock_encoder._initialize_client.assert_called_once_with(
            bedrock_encoder.access_key_id,
            bedrock_encoder.secret_access_key,
            None,
            bedrock_encoder.region,
        )

    def test_raises_value_error_if_call_to_bedrock_fails(self, bedrock_encoder):
        bedrock_encoder.client.invoke_model.side_effect = Exception(
            "Bedrock call failed."
        )
        with pytest.raises(ValueError):
            bedrock_encoder(["test"])

    def test_call_with_unknown_model_name(self, bedrock_encoder):
        bedrock_encoder.name = "unknown_model"
        with pytest.raises(ValueError):
            bedrock_encoder(["test"])

    def test_chunking_functionality(self, bedrock_encoder):
        docs = ["This is a long text that needs to be chunked properly."]
        chunked_docs = bedrock_encoder.chunk_strings(docs, MAX_WORDS=5)
        assert isinstance(chunked_docs, list), "Chunked result should be a list"
        assert len(chunked_docs[0]) > 1, (
            "Document should be chunked into multiple parts"
        )
        assert all(isinstance(chunk, str) for chunk in chunked_docs[0]), (
            "Chunks should be strings"
        )

    def test_get_env_variable(self):
        var_name = "TEST_ENV_VAR"
        default_value = "default"
        os.environ[var_name] = "env_value"
        assert BedrockEncoder.get_env_variable(var_name, None) == "env_value"
        assert (
            BedrockEncoder.get_env_variable(var_name, None, default_value)
            == "env_value"
        )
        assert (
            BedrockEncoder.get_env_variable("NON_EXISTENT_VAR", None, default_value)
            == default_value
        )

    def test_get_env_variable_missing(self):
        with pytest.raises(ValueError):
            BedrockEncoder.get_env_variable("MISSING_VAR", None)

    def test_uninitialised_client(self, bedrock_encoder):
        bedrock_encoder.client = None

        with pytest.raises(ValueError):
            bedrock_encoder(["test"])

    def test_missing_env_variables(self, mocker):
        mocker.patch.dict(os.environ, clear=True)

        with pytest.raises(ValueError):
            BedrockEncoder()


class TestBedrockEncoderWithCohere:
    def test_cohere_embedding_single_chunk(self, bedrock_encoder_with_cohere):
        response_content = json.dumps({"embeddings": [[0.1, 0.2, 0.3]]})
        response_body = BytesIO(response_content.encode("utf-8"))
        mock_response = {"body": response_body}
        bedrock_encoder_with_cohere.client.invoke_model.return_value = mock_response
        result = bedrock_encoder_with_cohere(["short test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(item, list) for item in result), (
            "Each item should be a list"
        )
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
