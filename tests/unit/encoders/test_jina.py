from unittest.mock import AsyncMock, Mock

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage
from openai.types.embedding import Embedding

from semantic_router.encoders import JinaEncoder
from semantic_router.encoders.jina import JINA_BASE_URL, jina_to_list


def embed_response(embeddings: list[list[float]]) -> CreateEmbeddingResponse:
    """Build a real OpenAI-shaped embedding response, as Jina's endpoint returns.

    Using the genuine SDK model means these tests fail if the response shape drifts.

    :param embeddings: One embedding per input document.
    :type embeddings: list[list[float]]
    :return: An embedding response.
    :rtype: CreateEmbeddingResponse
    """
    return CreateEmbeddingResponse(
        object="list",
        model="jina-embeddings-v3",
        usage=Usage(prompt_tokens=1, total_tokens=1),
        data=[
            Embedding(object="embedding", embedding=e, index=i)
            for i, e in enumerate(embeddings)
        ],
    )


@pytest.fixture
def mock_jina(mocker):
    """Patch both OpenAI clients and expose their mocked ``create`` methods."""
    sync_client = Mock()
    sync_client.embeddings.create = Mock(return_value=embed_response([[0.1, 0.2, 0.3]]))
    async_client = Mock()
    async_client.embeddings.create = AsyncMock(
        return_value=embed_response([[0.1, 0.2, 0.3]])
    )
    sync_cls = mocker.patch("openai.OpenAI", return_value=sync_client)
    async_cls = mocker.patch("openai.AsyncOpenAI", return_value=async_client)
    return sync_client, async_client, sync_cls, async_cls


@pytest.fixture
def encoder(mock_jina):
    return JinaEncoder("jina-embeddings-v3", api_key="test_api_key")


class TestJinaToList:
    def test_extracts_one_embedding_per_entry(self):
        assert jina_to_list(embed_response([[0.1, 0.2], [0.3, 0.4]])) == [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

    def test_raises_on_empty_response(self):
        with pytest.raises(ValueError, match="No embeddings found"):
            jina_to_list(embed_response([]))


class TestJinaEncoderInit:
    def test_initialization_with_api_key(self, mock_jina):
        enc = JinaEncoder("jina-embeddings-v3", api_key="test_api_key")
        assert enc.name == "jina-embeddings-v3"
        assert enc.type == "jina"
        assert enc.score_threshold == 0.4

    def test_initialization_with_env_api_key(self, mock_jina, monkeypatch):
        monkeypatch.delenv("JINA_AI_API_KEY", raising=False)
        monkeypatch.setenv("JINA_API_KEY", "test_api_key")
        assert JinaEncoder("jina-embeddings-v3").name == "jina-embeddings-v3"

    def test_initialization_with_legacy_env_api_key(self, mock_jina, monkeypatch):
        """JINA_AI_API_KEY is what the old litellm-based encoder read."""
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        monkeypatch.setenv("JINA_AI_API_KEY", "legacy_key")
        _, _, sync_cls, _ = mock_jina
        JinaEncoder("jina-embeddings-v3")
        assert sync_cls.call_args.kwargs["api_key"] == "legacy_key"

    def test_jina_api_key_takes_precedence(self, mock_jina, monkeypatch):
        monkeypatch.setenv("JINA_AI_API_KEY", "legacy_key")
        monkeypatch.setenv("JINA_API_KEY", "new_key")
        _, _, sync_cls, _ = mock_jina
        JinaEncoder("jina-embeddings-v3")
        assert sync_cls.call_args.kwargs["api_key"] == "new_key"

    def test_initialization_without_api_key(self, mock_jina, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        monkeypatch.delenv("JINA_AI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Expected API key"):
            JinaEncoder()

    def test_default_model_name(self, mock_jina, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "test_api_key")
        # name must not carry the litellm "jina_ai/" prefix any more
        assert JinaEncoder().name == "jina-embeddings-v3"

    def test_clients_point_at_jina_endpoint(self, mock_jina, monkeypatch):
        """Both clients must target Jina, not the default OpenAI endpoint."""
        monkeypatch.delenv("JINA_BASE_URL", raising=False)
        _, _, sync_cls, async_cls = mock_jina
        JinaEncoder("jina-embeddings-v3", api_key="test_api_key")
        assert JINA_BASE_URL == "https://api.jina.ai/v1"
        sync_cls.assert_called_once_with(api_key="test_api_key", base_url=JINA_BASE_URL)
        async_cls.assert_called_once_with(
            api_key="test_api_key", base_url=JINA_BASE_URL
        )

    def test_base_url_override_parameter(self, mock_jina):
        _, _, sync_cls, _ = mock_jina
        JinaEncoder(
            "jina-embeddings-v3",
            api_key="test_api_key",
            base_url="http://localhost:8080/v1",
        )
        assert sync_cls.call_args.kwargs["base_url"] == "http://localhost:8080/v1"

    def test_base_url_override_env_var(self, mock_jina, monkeypatch):
        monkeypatch.delenv("JINA_AI_API_BASE", raising=False)
        monkeypatch.setenv("JINA_BASE_URL", "http://env-host/v1")
        _, _, sync_cls, _ = mock_jina
        JinaEncoder("jina-embeddings-v3", api_key="test_api_key")
        assert sync_cls.call_args.kwargs["base_url"] == "http://env-host/v1"

    def test_base_url_legacy_env_var(self, mock_jina, monkeypatch):
        """JINA_AI_API_BASE is what the old litellm-based encoder read."""
        monkeypatch.delenv("JINA_BASE_URL", raising=False)
        monkeypatch.setenv("JINA_AI_API_BASE", "http://legacy-host/v1")
        _, _, sync_cls, _ = mock_jina
        JinaEncoder("jina-embeddings-v3", api_key="test_api_key")
        assert sync_cls.call_args.kwargs["base_url"] == "http://legacy-host/v1"


class TestJinaEncoderSync:
    def test_encode_queries(self, encoder, mock_jina):
        sync_client, _, _, _ = mock_jina
        assert encoder.encode_queries(["test"]) == [[0.1, 0.2, 0.3]]
        sync_client.embeddings.create.assert_called_once_with(
            input=["test"],
            model="jina-embeddings-v3",
        )

    def test_encode_documents(self, encoder, mock_jina):
        sync_client, _, _, _ = mock_jina
        # Jina embeddings are symmetric here, so documents reuse the query call
        assert encoder.encode_documents(["test"]) == [[0.1, 0.2, 0.3]]
        assert sync_client.embeddings.create.call_args.kwargs["input"] == ["test"]

    def test_call_delegates_to_encode_queries(self, encoder, mock_jina):
        sync_client, _, _, _ = mock_jina
        assert encoder(["test"]) == [[0.1, 0.2, 0.3]]
        sync_client.embeddings.create.assert_called_once()

    def test_handles_multiple_inputs_correctly(self, encoder, mock_jina):
        sync_client, _, _, _ = mock_jina
        sync_client.embeddings.create.return_value = embed_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        assert encoder.encode_documents(["test1", "test2"]) == [[0.1, 0.2], [0.3, 0.4]]
        assert sync_client.embeddings.create.call_args.kwargs["input"] == [
            "test1",
            "test2",
        ]

    def test_kwargs_forwarded(self, encoder, mock_jina):
        """Jina-specific options ride along via extra_body."""
        sync_client, _, _, _ = mock_jina
        encoder.encode_queries(
            ["test"], dimensions=256, extra_body={"task": "retrieval.query"}
        )
        kwargs = sync_client.embeddings.create.call_args.kwargs
        assert kwargs["dimensions"] == 256
        assert kwargs["extra_body"] == {"task": "retrieval.query"}

    def test_raises_error_on_api_failure(self, encoder, mock_jina):
        sync_client, _, _, _ = mock_jina
        sync_client.embeddings.create.side_effect = Exception("API call failed")
        with pytest.raises(ValueError, match="Jina API call failed"):
            encoder.encode_queries(["test"])

    def test_raises_error_on_empty_response(self, encoder, mock_jina):
        sync_client, _, _, _ = mock_jina
        sync_client.embeddings.create.return_value = embed_response([])
        with pytest.raises(ValueError, match="Jina API call failed"):
            encoder.encode_queries(["test"])


class TestJinaEncoderAsync:
    @pytest.mark.asyncio
    async def test_aencode_queries(self, encoder, mock_jina):
        _, async_client, _, _ = mock_jina
        assert await encoder.aencode_queries(["test"]) == [[0.1, 0.2, 0.3]]
        async_client.embeddings.create.assert_awaited_once_with(
            input=["test"],
            model="jina-embeddings-v3",
        )

    @pytest.mark.asyncio
    async def test_aencode_documents(self, encoder, mock_jina):
        _, async_client, _, _ = mock_jina
        assert await encoder.aencode_documents(["test"]) == [[0.1, 0.2, 0.3]]
        assert async_client.embeddings.create.call_args.kwargs["input"] == ["test"]

    @pytest.mark.asyncio
    async def test_acall_delegates_to_aencode_queries(self, encoder, mock_jina):
        _, async_client, _, _ = mock_jina
        assert await encoder.acall(["test"]) == [[0.1, 0.2, 0.3]]
        async_client.embeddings.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_multiple_inputs_correctly(self, encoder, mock_jina):
        _, async_client, _, _ = mock_jina
        async_client.embeddings.create.return_value = embed_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        result = await encoder.aencode_documents(["test1", "test2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_uses_async_client(self, encoder, mock_jina):
        """The async path must not fall back to the blocking client."""
        sync_client, _, _, _ = mock_jina
        await encoder.aencode_queries(["test"])
        sync_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_error_on_api_failure(self, encoder, mock_jina):
        _, async_client, _, _ = mock_jina
        async_client.embeddings.create.side_effect = Exception("API call failed")
        with pytest.raises(ValueError, match="Jina API call failed"):
            await encoder.aencode_documents(["test"])
