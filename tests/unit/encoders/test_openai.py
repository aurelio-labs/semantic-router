from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai import OpenAIError
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from semantic_router.encoders import OpenAIEncoder


@pytest.fixture
def mock_openai_client():
    with patch("openai.Client") as mock_client:
        yield mock_client


@pytest.fixture
def mock_openai_async_client():
    with patch("openai.AsyncClient") as mock_async_client:
        yield mock_async_client


@pytest.fixture
def openai_encoder(mock_openai_client, mock_openai_async_client):
    return OpenAIEncoder(openai_api_key="fake_key", max_retries=2)


class TestOpenAIEncoder:
    def test_openai_encoder_init_success(self, mocker):
        # -- Mock the return value of os.getenv 3 times: model name, api key and org ID
        side_effect = ["fake-model-name", "fake-api-key", "fake-org-id"]
        mocker.patch("os.getenv", side_effect=side_effect)
        encoder = OpenAIEncoder()
        assert encoder._client is not None

    def test_openai_encoder_init_no_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            OpenAIEncoder()

    def test_openai_encoder_call_uninitialized_client(self, openai_encoder):
        # Set the client to None to simulate an uninitialized client
        openai_encoder._client = None
        with pytest.raises(ValueError) as e:
            openai_encoder(["test document"])
        assert "OpenAI client is not initialized." in str(e.value)

    def test_openai_encoder_init_exception(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("openai.Client", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            OpenAIEncoder()
        assert (
            "OpenAI API client failed to initialize. Error: Initialization error"
            in str(e.value)
        )

    def test_openai_encoder_call_success(self, openai_encoder, mocker):
        mock_embeddings = mocker.Mock()
        mock_embeddings.data = [
            Embedding(embedding=[0.1, 0.2], index=0, object="embedding")
        ]

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)  # To speed up the test

        mock_embedding = Embedding(index=0, object="embedding", embedding=[0.1, 0.2])
        # Mock the CreateEmbeddingResponse object
        mock_response = CreateEmbeddingResponse(
            model="text-embedding-ada-002",
            object="list",
            usage=Usage(prompt_tokens=0, total_tokens=20),
            data=[mock_embedding],
        )

        responses = [OpenAIError("OpenAI error"), mock_response]
        mocker.patch.object(
            openai_encoder._client.embeddings, "create", side_effect=responses
        )
        with patch("semantic_router.encoders.openai.sleep", return_value=None):
            embeddings = openai_encoder(["test document"])
        assert embeddings == [[0.1, 0.2]]

    def test_openai_encoder_call_failure_non_openai_error(self, openai_encoder, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)  # To speed up the test
        mocker.patch.object(
            openai_encoder._client.embeddings,
            "create",
            side_effect=Exception("Non-OpenAIError"),
        )
        with patch("semantic_router.encoders.openai.sleep", return_value=None):
            with pytest.raises(ValueError) as e:
                openai_encoder(["test document"])

        assert "OpenAI API call failed. Error: Non-OpenAIError" in str(e.value)

    def test_openai_encoder_call_successful_retry(self, openai_encoder, mocker):
        mock_embeddings = mocker.Mock()
        mock_embeddings.data = [
            Embedding(embedding=[0.1, 0.2], index=0, object="embedding")
        ]

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)  # To speed up the test

        mock_embedding = Embedding(index=0, object="embedding", embedding=[0.1, 0.2])
        # Mock the CreateEmbeddingResponse object
        mock_response = CreateEmbeddingResponse(
            model="text-embedding-ada-002",
            object="list",
            usage=Usage(prompt_tokens=0, total_tokens=20),
            data=[mock_embedding],
        )

        responses = [OpenAIError("OpenAI error"), mock_response]
        mocker.patch.object(
            openai_encoder._client.embeddings, "create", side_effect=responses
        )
        with patch("semantic_router.encoders.openai.sleep", return_value=None):
            embeddings = openai_encoder(["test document"])
        assert embeddings == [[0.1, 0.2]]

    def test_retry_logic_sync(self, openai_encoder, mock_openai_client, mocker):
        # Mock the embeddings.create method to raise an error twice, then succeed
        mock_create = Mock(
            side_effect=[
                OpenAIError("API error"),
                OpenAIError("API error"),
                CreateEmbeddingResponse(
                    data=[
                        Embedding(
                            embedding=[0.1, 0.2, 0.3], index=0, object="embedding"
                        )
                    ],
                    model="text-embedding-3-small",
                    object="list",
                    usage={"prompt_tokens": 5, "total_tokens": 5},
                ),
            ]
        )
        mock_openai_client.return_value.embeddings.create = mock_create
        mocker.patch("time.sleep", return_value=None)  # To speed up the test

        # Patch the sleep function in the encoder module to avoid actual sleep
        with patch("semantic_router.encoders.openai.sleep", return_value=None):
            result = openai_encoder(["test document"])

        assert result == [[0.1, 0.2, 0.3]]
        assert mock_create.call_count == 3

    def test_no_retry_on_max_retries_zero(self, openai_encoder, mock_openai_client):
        openai_encoder.max_retries = 0
        # Mock the embeddings.create method to always raise an error
        mock_create = Mock(side_effect=OpenAIError("API error"))
        mock_openai_client.return_value.embeddings.create = mock_create

        with pytest.raises(OpenAIError):
            openai_encoder(["test document"])

        assert mock_create.call_count == 1  # Only the initial attempt, no retries

    def test_retry_logic_sync_max_retries_exceeded(
        self, openai_encoder, mock_openai_client, mocker
    ):
        # Mock the embeddings.create method to always raise an error
        mock_create = Mock(side_effect=OpenAIError("API error"))
        mock_openai_client.return_value.embeddings.create = mock_create
        mocker.patch("time.sleep", return_value=None)  # To speed up the test

        # Patch the sleep function in the encoder module to avoid actual sleep
        with patch("semantic_router.encoders.openai.sleep", return_value=None):
            with pytest.raises(OpenAIError):
                openai_encoder(["test document"])

        assert mock_create.call_count == 3  # Initial attempt + 2 retries

    @pytest.mark.asyncio
    async def test_retry_logic_async(
        self, openai_encoder, mock_openai_async_client, mocker
    ):
        # Set up the mock to fail twice, then succeed
        mock_create = AsyncMock(
            side_effect=[
                OpenAIError("API error"),
                OpenAIError("API error"),
                CreateEmbeddingResponse(
                    data=[
                        Embedding(
                            embedding=[0.1, 0.2, 0.3], index=0, object="embedding"
                        )
                    ],
                    model="text-embedding-3-small",
                    object="list",
                    usage={"prompt_tokens": 5, "total_tokens": 5},
                ),
            ]
        )
        mock_openai_async_client.return_value.embeddings.create = mock_create
        mocker.patch("asyncio.sleep", return_value=None)  # To speed up the test

        # Patch the asleep function in the encoder module to avoid actual sleep
        with patch("semantic_router.encoders.openai.asleep", return_value=None):
            result = await openai_encoder.acall(["test document"])

        assert result == [[0.1, 0.2, 0.3]]
        assert mock_create.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_logic_async_max_retries_exceeded(
        self, openai_encoder, mock_openai_async_client, mocker
    ):
        # Mock the embeddings.create method to always raise an error
        async def raise_error(*args, **kwargs):
            raise OpenAIError("API error")

        mock_create = Mock(side_effect=raise_error)
        mock_openai_async_client.return_value.embeddings.create = mock_create
        mocker.patch("asyncio.sleep", return_value=None)  # To speed up the test

        # Patch the asleep function in the encoder module to avoid actual sleep
        with patch("semantic_router.encoders.openai.asleep", return_value=None):
            with pytest.raises(OpenAIError):
                await openai_encoder.acall(["test document"])

        assert mock_create.call_count == 3  # Initial attempt + 2 retries

    @pytest.mark.asyncio
    async def test_no_retry_on_max_retries_zero_async(
        self, openai_encoder, mock_openai_async_client
    ):
        openai_encoder.max_retries = 0

        # Mock the embeddings.create method to always raise an error
        async def raise_error(*args, **kwargs):
            raise OpenAIError("API error")

        mock_create = AsyncMock(side_effect=raise_error)
        mock_openai_async_client.return_value.embeddings.create = mock_create

        with pytest.raises(OpenAIError):
            await openai_encoder.acall(["test document"])

        assert mock_create.call_count == 1  # Only the initial attempt, no retries
