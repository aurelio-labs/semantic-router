from unittest.mock import AsyncMock, Mock

import pytest
from cohere.types.embed_by_type_response import EmbedByTypeResponse
from cohere.types.embed_by_type_response_embeddings import (
    EmbedByTypeResponseEmbeddings,
)

from semantic_router.encoders import CohereEncoder
from semantic_router.encoders.cohere import docs2cohere_embed_input


def embed_response(embeddings: list[list[float]] | None) -> EmbedByTypeResponse:
    """Build a real Cohere embed response so tests fail if the SDK shape changes.

    :param embeddings: The float embeddings to return, or None to simulate a
        response carrying no float embeddings.
    :type embeddings: list[list[float]] | None
    :return: A Cohere embed response.
    :rtype: EmbedByTypeResponse
    """
    return EmbedByTypeResponse(
        id="test-id",
        embeddings=EmbedByTypeResponseEmbeddings(float_=embeddings),
    )


@pytest.fixture
def mock_cohere(mocker):
    """Patch both Cohere clients and expose their mocked ``embed`` methods."""
    sync_client = Mock()
    sync_client.embed = Mock(return_value=embed_response([[0.1, 0.2, 0.3]]))
    async_client = Mock()
    async_client.embed = AsyncMock(return_value=embed_response([[0.1, 0.2, 0.3]]))
    mocker.patch("cohere.ClientV2", return_value=sync_client)
    mocker.patch("cohere.AsyncClientV2", return_value=async_client)
    return sync_client, async_client


@pytest.fixture
def encoder(mock_cohere):
    return CohereEncoder("embed-english-v3.0", cohere_api_key="test_api_key")


class TestDocs2CohereEmbedInput:
    def test_one_input_per_doc(self):
        assert docs2cohere_embed_input(["a", "b"]) == [
            {"content": [{"type": "text", "text": "a"}]},
            {"content": [{"type": "text", "text": "b"}]},
        ]

    def test_empty_docs(self):
        assert docs2cohere_embed_input([]) == []


class TestCohereEncoderInit:
    def test_initialization_with_api_key(self, mock_cohere):
        enc = CohereEncoder("embed-english-v3.0", cohere_api_key="test_api_key")
        assert enc.name == "embed-english-v3.0"
        assert enc.type == "cohere"
        assert enc.score_threshold == 0.3

    def test_initialization_with_env_api_key(self, mock_cohere, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test_api_key")
        assert CohereEncoder("embed-english-v3.0").name == "embed-english-v3.0"

    def test_initialization_without_api_key(self, mock_cohere, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Expected API key"):
            CohereEncoder()

    def test_default_model_name(self, mock_cohere, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test_api_key")
        assert CohereEncoder().name == "embed-english-v3.0"

    def test_api_key_passed_to_both_clients(self, mocker, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        sync_cls = mocker.patch("cohere.ClientV2")
        async_cls = mocker.patch("cohere.AsyncClientV2")
        CohereEncoder("embed-english-v3.0", cohere_api_key="test_api_key")
        sync_cls.assert_called_once_with(api_key="test_api_key")
        async_cls.assert_called_once_with(api_key="test_api_key")


class TestCohereEncoderSync:
    def test_encode_queries(self, encoder, mock_cohere):
        sync_client, _ = mock_cohere
        assert encoder.encode_queries(["test"]) == [[0.1, 0.2, 0.3]]
        sync_client.embed.assert_called_once_with(
            inputs=[{"content": [{"type": "text", "text": "test"}]}],
            input_type="search_query",
            model="embed-english-v3.0",
            embedding_types=["float"],
        )

    def test_encode_documents(self, encoder, mock_cohere):
        sync_client, _ = mock_cohere
        assert encoder.encode_documents(["test"]) == [[0.1, 0.2, 0.3]]
        assert sync_client.embed.call_args.kwargs["input_type"] == "search_document"

    def test_call_delegates_to_encode_queries(self, encoder, mock_cohere):
        sync_client, _ = mock_cohere
        assert encoder(["test"]) == [[0.1, 0.2, 0.3]]
        assert sync_client.embed.call_args.kwargs["input_type"] == "search_query"

    def test_handles_multiple_inputs_correctly(self, encoder, mock_cohere):
        sync_client, _ = mock_cohere
        sync_client.embed.return_value = embed_response([[0.1, 0.2], [0.3, 0.4]])
        assert encoder.encode_documents(["test1", "test2"]) == [[0.1, 0.2], [0.3, 0.4]]
        # one embed input per doc, otherwise Cohere returns a single embedding
        assert sync_client.embed.call_args.kwargs["inputs"] == [
            {"content": [{"type": "text", "text": "test1"}]},
            {"content": [{"type": "text", "text": "test2"}]},
        ]

    def test_kwargs_forwarded(self, encoder, mock_cohere):
        sync_client, _ = mock_cohere
        encoder.encode_queries(["test"], output_dimension=256)
        assert sync_client.embed.call_args.kwargs["output_dimension"] == 256

    def test_raises_error_on_api_failure(self, encoder, mock_cohere):
        sync_client, _ = mock_cohere
        sync_client.embed.side_effect = Exception("API call failed")
        with pytest.raises(ValueError, match="Cohere API call failed"):
            encoder.encode_queries(["test"])

    def test_raises_error_on_none_embeddings(self, encoder, mock_cohere):
        sync_client, _ = mock_cohere
        sync_client.embed.return_value = embed_response(None)
        with pytest.raises(ValueError, match="Cohere API call returned None"):
            encoder.encode_queries(["test"])


class TestCohereEncoderAsync:
    @pytest.mark.asyncio
    async def test_aencode_queries(self, encoder, mock_cohere):
        _, async_client = mock_cohere
        assert await encoder.aencode_queries(["test"]) == [[0.1, 0.2, 0.3]]
        async_client.embed.assert_awaited_once_with(
            inputs=[{"content": [{"type": "text", "text": "test"}]}],
            input_type="search_query",
            model="embed-english-v3.0",
            embedding_types=["float"],
        )

    @pytest.mark.asyncio
    async def test_aencode_documents(self, encoder, mock_cohere):
        _, async_client = mock_cohere
        assert await encoder.aencode_documents(["test"]) == [[0.1, 0.2, 0.3]]
        assert async_client.embed.call_args.kwargs["input_type"] == "search_document"

    @pytest.mark.asyncio
    async def test_acall_delegates_to_aencode_queries(self, encoder, mock_cohere):
        _, async_client = mock_cohere
        assert await encoder.acall(["test"]) == [[0.1, 0.2, 0.3]]
        assert async_client.embed.call_args.kwargs["input_type"] == "search_query"

    @pytest.mark.asyncio
    async def test_handles_multiple_inputs_correctly(self, encoder, mock_cohere):
        _, async_client = mock_cohere
        async_client.embed.return_value = embed_response([[0.1, 0.2], [0.3, 0.4]])
        result = await encoder.aencode_documents(["test1", "test2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert len(async_client.embed.call_args.kwargs["inputs"]) == 2

    @pytest.mark.asyncio
    async def test_uses_async_client(self, encoder, mock_cohere):
        """The async path must not fall back to the blocking client."""
        sync_client, _ = mock_cohere
        await encoder.aencode_queries(["test"])
        sync_client.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_error_on_api_failure(self, encoder, mock_cohere):
        _, async_client = mock_cohere
        async_client.embed.side_effect = Exception("API call failed")
        with pytest.raises(ValueError, match="Cohere API call failed"):
            await encoder.aencode_queries(["test"])

    @pytest.mark.asyncio
    async def test_raises_error_on_none_embeddings(self, encoder, mock_cohere):
        _, async_client = mock_cohere
        async_client.embed.return_value = embed_response(None)
        with pytest.raises(ValueError, match="Cohere API call returned None"):
            await encoder.aencode_documents(["test"])
