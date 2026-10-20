from unittest.mock import AsyncMock, Mock

import pytest
from voyageai.api_resources import VoyageResponse
from voyageai.object.embeddings import EmbeddingsObject

from semantic_router.encoders import VoyageEncoder


def embed_response(embeddings: list[list[float]]) -> EmbeddingsObject:
    """Build a real Voyage embeddings object so tests fail if the SDK shape changes.

    :param embeddings: One embedding per input document.
    :type embeddings: list[list[float]]
    :return: A Voyage embeddings object.
    :rtype: EmbeddingsObject
    """
    response = VoyageResponse.construct_from(
        {
            "data": [
                {"embedding": e, "index": i, "object": "embedding"}
                for i, e in enumerate(embeddings)
            ],
            "usage": {"total_tokens": 3},
        }
    )
    return EmbeddingsObject(response)


@pytest.fixture
def mock_voyage(mocker):
    """Patch both Voyage clients and expose their mocked ``embed`` methods."""
    sync_client = Mock()
    sync_client.embed = Mock(return_value=embed_response([[0.1, 0.2, 0.3]]))
    async_client = Mock()
    async_client.embed = AsyncMock(return_value=embed_response([[0.1, 0.2, 0.3]]))
    mocker.patch("semantic_router.encoders.voyage.Client", return_value=sync_client)
    mocker.patch(
        "semantic_router.encoders.voyage.AsyncClient", return_value=async_client
    )
    return sync_client, async_client


@pytest.fixture
def encoder(mock_voyage):
    return VoyageEncoder("voyage-3", api_key="test_api_key")


class TestVoyageEncoderInit:
    def test_initialization_with_api_key(self, mock_voyage):
        enc = VoyageEncoder("voyage-3", api_key="test_api_key")
        assert enc.name == "voyage-3"
        assert enc.type == "voyage"
        assert enc.score_threshold == 0.4

    def test_initialization_with_env_api_key(self, mock_voyage, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test_api_key")
        assert VoyageEncoder("voyage-3").name == "voyage-3"

    def test_initialization_without_api_key(self, mock_voyage, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Expected API key"):
            VoyageEncoder()

    def test_default_model_name(self, mock_voyage, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test_api_key")
        # name must not carry the litellm "voyage/" prefix any more
        assert VoyageEncoder().name == "voyage-3-lite"

    def test_api_key_passed_to_both_clients(self, mocker, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        sync_cls = mocker.patch("semantic_router.encoders.voyage.Client")
        async_cls = mocker.patch("semantic_router.encoders.voyage.AsyncClient")
        VoyageEncoder("voyage-3", api_key="test_api_key")
        sync_cls.assert_called_once_with(api_key="test_api_key")
        async_cls.assert_called_once_with(api_key="test_api_key")


class TestVoyageEncoderSync:
    def test_encode_queries(self, encoder, mock_voyage):
        sync_client, _ = mock_voyage
        assert encoder.encode_queries(["test"]) == [[0.1, 0.2, 0.3]]
        sync_client.embed.assert_called_once_with(
            texts=["test"],
            model="voyage-3",
            input_type="query",
        )

    def test_encode_documents(self, encoder, mock_voyage):
        sync_client, _ = mock_voyage
        assert encoder.encode_documents(["test"]) == [[0.1, 0.2, 0.3]]
        assert sync_client.embed.call_args.kwargs["input_type"] == "document"

    def test_call_delegates_to_encode_queries(self, encoder, mock_voyage):
        sync_client, _ = mock_voyage
        assert encoder(["test"]) == [[0.1, 0.2, 0.3]]
        assert sync_client.embed.call_args.kwargs["input_type"] == "query"

    def test_handles_multiple_inputs_correctly(self, encoder, mock_voyage):
        sync_client, _ = mock_voyage
        sync_client.embed.return_value = embed_response([[0.1, 0.2], [0.3, 0.4]])
        assert encoder.encode_documents(["test1", "test2"]) == [[0.1, 0.2], [0.3, 0.4]]
        assert sync_client.embed.call_args.kwargs["texts"] == ["test1", "test2"]

    def test_kwargs_forwarded(self, encoder, mock_voyage):
        sync_client, _ = mock_voyage
        encoder.encode_queries(["test"], output_dimension=256, truncation=False)
        assert sync_client.embed.call_args.kwargs["output_dimension"] == 256
        assert sync_client.embed.call_args.kwargs["truncation"] is False

    def test_raises_error_on_api_failure(self, encoder, mock_voyage):
        sync_client, _ = mock_voyage
        sync_client.embed.side_effect = Exception("API call failed")
        with pytest.raises(ValueError, match="Voyage API call failed"):
            encoder.encode_queries(["test"])


class TestVoyageEncoderAsync:
    @pytest.mark.asyncio
    async def test_aencode_queries(self, encoder, mock_voyage):
        _, async_client = mock_voyage
        assert await encoder.aencode_queries(["test"]) == [[0.1, 0.2, 0.3]]
        async_client.embed.assert_awaited_once_with(
            texts=["test"],
            model="voyage-3",
            input_type="query",
        )

    @pytest.mark.asyncio
    async def test_aencode_documents(self, encoder, mock_voyage):
        _, async_client = mock_voyage
        assert await encoder.aencode_documents(["test"]) == [[0.1, 0.2, 0.3]]
        assert async_client.embed.call_args.kwargs["input_type"] == "document"

    @pytest.mark.asyncio
    async def test_acall_delegates_to_aencode_queries(self, encoder, mock_voyage):
        _, async_client = mock_voyage
        assert await encoder.acall(["test"]) == [[0.1, 0.2, 0.3]]
        assert async_client.embed.call_args.kwargs["input_type"] == "query"

    @pytest.mark.asyncio
    async def test_handles_multiple_inputs_correctly(self, encoder, mock_voyage):
        _, async_client = mock_voyage
        async_client.embed.return_value = embed_response([[0.1, 0.2], [0.3, 0.4]])
        result = await encoder.aencode_documents(["test1", "test2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert async_client.embed.call_args.kwargs["texts"] == ["test1", "test2"]

    @pytest.mark.asyncio
    async def test_uses_async_client(self, encoder, mock_voyage):
        """The async path must not fall back to the blocking client."""
        sync_client, _ = mock_voyage
        await encoder.aencode_queries(["test"])
        sync_client.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_error_on_api_failure(self, encoder, mock_voyage):
        _, async_client = mock_voyage
        async_client.embed.side_effect = Exception("API call failed")
        with pytest.raises(ValueError, match="Voyage API call failed"):
            await encoder.aencode_documents(["test"])
