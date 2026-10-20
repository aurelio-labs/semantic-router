from unittest.mock import AsyncMock, Mock

import pytest
from mistralai.models import EmbeddingResponse, EmbeddingResponseData, UsageInfo

from semantic_router.encoders import MistralEncoder
from semantic_router.encoders.mistral import mistral_to_list


def embed_response(embeddings: list[list[float] | None]) -> EmbeddingResponse:
    """Build a real Mistral embedding response so tests fail if the SDK shape changes.

    :param embeddings: One embedding per entry; None models an entry returned
        without an embedding.
    :type embeddings: list[list[float] | None]
    :return: A Mistral embedding response.
    :rtype: EmbeddingResponse
    """
    return EmbeddingResponse(
        id="test-id",
        object="list",
        model="mistral-embed",
        usage=UsageInfo(prompt_tokens=1, total_tokens=1),
        data=[
            EmbeddingResponseData(object="embedding", embedding=e, index=i)
            for i, e in enumerate(embeddings)
        ],
    )


@pytest.fixture
def mock_mistral(mocker):
    """Patch the Mistral client and expose its mocked sync/async embed methods."""
    client = Mock()
    client.embeddings.create = Mock(return_value=embed_response([[0.1, 0.2, 0.3]]))
    client.embeddings.create_async = AsyncMock(
        return_value=embed_response([[0.1, 0.2, 0.3]])
    )
    mocker.patch("semantic_router.encoders.mistral.Mistral", return_value=client)
    return client


@pytest.fixture
def encoder(mock_mistral):
    return MistralEncoder("mistral-embed", mistralai_api_key="test_api_key")


class TestMistralToList:
    def test_extracts_one_embedding_per_entry(self):
        response = embed_response([[0.1, 0.2], [0.3, 0.4]])
        assert mistral_to_list(response) == [[0.1, 0.2], [0.3, 0.4]]

    def test_raises_on_empty_data(self):
        response = embed_response([])
        with pytest.raises(ValueError, match="No embeddings found"):
            mistral_to_list(response)

    def test_raises_on_entry_without_embedding(self):
        response = embed_response([[0.1, 0.2], None])
        with pytest.raises(ValueError, match="empty entry"):
            mistral_to_list(response)


class TestMistralEncoderInit:
    def test_initialization_with_api_key(self, mock_mistral):
        enc = MistralEncoder("mistral-embed", mistralai_api_key="test_api_key")
        assert enc.name == "mistral-embed"
        assert enc.type == "mistral"
        assert enc.score_threshold == 0.4

    def test_initialization_with_mistralai_env_key(self, mock_mistral, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.setenv("MISTRALAI_API_KEY", "test_api_key")
        assert MistralEncoder("mistral-embed").name == "mistral-embed"

    def test_initialization_with_mistral_env_key(self, mock_mistral, monkeypatch):
        monkeypatch.delenv("MISTRALAI_API_KEY", raising=False)
        monkeypatch.setenv("MISTRAL_API_KEY", "test_api_key")
        assert MistralEncoder("mistral-embed").name == "mistral-embed"

    def test_initialization_without_api_key(self, mock_mistral, monkeypatch):
        monkeypatch.delenv("MISTRALAI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Expected API key"):
            MistralEncoder()

    def test_default_model_name(self, mock_mistral, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test_api_key")
        # name must not carry the litellm "mistral/" prefix any more
        assert MistralEncoder().name == "mistral-embed"

    def test_api_key_passed_to_client(self, mocker, monkeypatch):
        monkeypatch.delenv("MISTRALAI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        client_cls = mocker.patch("semantic_router.encoders.mistral.Mistral")
        MistralEncoder("mistral-embed", mistralai_api_key="test_api_key")
        client_cls.assert_called_once_with(api_key="test_api_key")


class TestMistralEncoderSync:
    def test_encode_queries(self, encoder, mock_mistral):
        assert encoder.encode_queries(["test"]) == [[0.1, 0.2, 0.3]]
        mock_mistral.embeddings.create.assert_called_once_with(
            inputs=["test"],
            model="mistral-embed",
            encoding_format="float",
        )

    def test_encode_documents(self, encoder, mock_mistral):
        # Mistral embeddings are symmetric, so documents reuse the query call
        assert encoder.encode_documents(["test"]) == [[0.1, 0.2, 0.3]]
        assert mock_mistral.embeddings.create.call_args.kwargs["inputs"] == ["test"]

    def test_call_delegates_to_encode_queries(self, encoder, mock_mistral):
        assert encoder(["test"]) == [[0.1, 0.2, 0.3]]
        mock_mistral.embeddings.create.assert_called_once()

    def test_handles_multiple_inputs_correctly(self, encoder, mock_mistral):
        mock_mistral.embeddings.create.return_value = embed_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        assert encoder.encode_documents(["test1", "test2"]) == [[0.1, 0.2], [0.3, 0.4]]
        assert mock_mistral.embeddings.create.call_args.kwargs["inputs"] == [
            "test1",
            "test2",
        ]

    def test_kwargs_forwarded(self, encoder, mock_mistral):
        encoder.encode_queries(["test"], output_dimension=256)
        assert (
            mock_mistral.embeddings.create.call_args.kwargs["output_dimension"] == 256
        )

    def test_raises_error_on_api_failure(self, encoder, mock_mistral):
        mock_mistral.embeddings.create.side_effect = Exception("API call failed")
        with pytest.raises(ValueError, match="Mistral API call failed"):
            encoder.encode_queries(["test"])

    def test_raises_error_on_empty_response(self, encoder, mock_mistral):
        mock_mistral.embeddings.create.return_value = embed_response([])
        with pytest.raises(ValueError, match="Mistral API call failed"):
            encoder.encode_queries(["test"])


class TestMistralEncoderAsync:
    @pytest.mark.asyncio
    async def test_aencode_queries(self, encoder, mock_mistral):
        assert await encoder.aencode_queries(["test"]) == [[0.1, 0.2, 0.3]]
        mock_mistral.embeddings.create_async.assert_awaited_once_with(
            inputs=["test"],
            model="mistral-embed",
            encoding_format="float",
        )

    @pytest.mark.asyncio
    async def test_aencode_documents(self, encoder, mock_mistral):
        assert await encoder.aencode_documents(["test"]) == [[0.1, 0.2, 0.3]]
        mock_mistral.embeddings.create_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_acall_delegates_to_aencode_queries(self, encoder, mock_mistral):
        assert await encoder.acall(["test"]) == [[0.1, 0.2, 0.3]]
        mock_mistral.embeddings.create_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_multiple_inputs_correctly(self, encoder, mock_mistral):
        mock_mistral.embeddings.create_async.return_value = embed_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        result = await encoder.aencode_documents(["test1", "test2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_uses_async_method(self, encoder, mock_mistral):
        """The async path must not fall back to the blocking call."""
        await encoder.aencode_queries(["test"])
        mock_mistral.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_error_on_api_failure(self, encoder, mock_mistral):
        mock_mistral.embeddings.create_async.side_effect = Exception("API call failed")
        with pytest.raises(ValueError, match="Mistral API call failed"):
            await encoder.aencode_queries(["test"])

    @pytest.mark.asyncio
    async def test_raises_error_on_empty_response(self, encoder, mock_mistral):
        mock_mistral.embeddings.create_async.return_value = embed_response([])
        with pytest.raises(ValueError, match="Mistral API call failed"):
            await encoder.aencode_documents(["test"])
