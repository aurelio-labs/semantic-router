from unittest.mock import AsyncMock, Mock

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage
from openai.types.embedding import Embedding

from semantic_router.encoders import NimEncoder
from semantic_router.encoders.nvidia_nim import NIM_BASE_URL, nim_to_list


def embed_response(embeddings: list[list[float]]) -> CreateEmbeddingResponse:
    """Build a real OpenAI-shaped embedding response, as NIM's endpoint returns.

    Using the genuine SDK model means these tests fail if the response shape drifts.

    :param embeddings: One embedding per input document.
    :type embeddings: list[list[float]]
    :return: An embedding response.
    :rtype: CreateEmbeddingResponse
    """
    return CreateEmbeddingResponse(
        object="list",
        model="nvidia/nv-embedqa-e5-v5",
        usage=Usage(prompt_tokens=1, total_tokens=1),
        data=[
            Embedding(object="embedding", embedding=e, index=i)
            for i, e in enumerate(embeddings)
        ],
    )


@pytest.fixture
def mock_nim(mocker):
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
def encoder(mock_nim):
    return NimEncoder("nvidia/nv-embedqa-e5-v5", api_key="test_api_key")


class TestNimToList:
    def test_extracts_one_embedding_per_entry(self):
        assert nim_to_list(embed_response([[0.1, 0.2], [0.3, 0.4]])) == [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

    def test_raises_on_empty_response(self):
        with pytest.raises(ValueError, match="No embeddings found"):
            nim_to_list(embed_response([]))


class TestNimEncoderInit:
    def test_initialization_with_api_key(self, mock_nim):
        enc = NimEncoder("nvidia/nv-embedqa-e5-v5", api_key="test_api_key")
        assert enc.name == "nvidia/nv-embedqa-e5-v5"
        assert enc.type == "nvidia_nim"
        assert enc.score_threshold == 0.4

    def test_initialization_with_env_api_key(self, mock_nim, monkeypatch):
        monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test_api_key")
        _, _, sync_cls, _ = mock_nim
        NimEncoder("nvidia/nv-embedqa-e5-v5")
        assert sync_cls.call_args.kwargs["api_key"] == "test_api_key"

    def test_initialization_without_api_key(self, mock_nim, monkeypatch):
        monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Expected API key"):
            NimEncoder()

    def test_default_model_name(self, mock_nim, monkeypatch):
        monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test_api_key")
        # name must not carry the litellm "nvidia_nim/" prefix any more
        assert NimEncoder().name == "nvidia/nv-embedqa-e5-v5"

    def test_clients_point_at_nim_endpoint(self, mock_nim, monkeypatch):
        """Both clients must target NVIDIA, not the default OpenAI endpoint."""
        monkeypatch.delenv("NVIDIA_NIM_API_BASE", raising=False)
        _, _, sync_cls, async_cls = mock_nim
        NimEncoder("nvidia/nv-embedqa-e5-v5", api_key="test_api_key")
        assert NIM_BASE_URL == "https://integrate.api.nvidia.com/v1"
        sync_cls.assert_called_once_with(api_key="test_api_key", base_url=NIM_BASE_URL)
        async_cls.assert_called_once_with(api_key="test_api_key", base_url=NIM_BASE_URL)

    def test_base_url_override_parameter(self, mock_nim):
        """Self-hosted NIM containers are reached via base_url."""
        _, _, sync_cls, _ = mock_nim
        NimEncoder(
            "nvidia/nv-embedqa-e5-v5",
            api_key="test_api_key",
            base_url="http://localhost:8000/v1",
        )
        assert sync_cls.call_args.kwargs["base_url"] == "http://localhost:8000/v1"

    def test_base_url_override_env_var(self, mock_nim, monkeypatch):
        """NVIDIA_NIM_API_BASE is the variable the old litellm encoder read."""
        monkeypatch.setenv("NVIDIA_NIM_API_BASE", "http://env-host/v1")
        _, _, sync_cls, _ = mock_nim
        NimEncoder("nvidia/nv-embedqa-e5-v5", api_key="test_api_key")
        assert sync_cls.call_args.kwargs["base_url"] == "http://env-host/v1"


class TestNimEncoderSync:
    def test_encode_queries_uses_query_input_type(self, encoder, mock_nim):
        """nv-embedqa models are asymmetric: queries must not be sent as passages."""
        sync_client, _, _, _ = mock_nim
        assert encoder.encode_queries(["test"]) == [[0.1, 0.2, 0.3]]
        sync_client.embeddings.create.assert_called_once_with(
            input=["test"],
            model="nvidia/nv-embedqa-e5-v5",
            extra_body={"input_type": "query"},
        )

    def test_encode_documents_uses_passage_input_type(self, encoder, mock_nim):
        sync_client, _, _, _ = mock_nim
        assert encoder.encode_documents(["test"]) == [[0.1, 0.2, 0.3]]
        sync_client.embeddings.create.assert_called_once_with(
            input=["test"],
            model="nvidia/nv-embedqa-e5-v5",
            extra_body={"input_type": "passage"},
        )

    def test_queries_and_documents_differ(self, encoder, mock_nim):
        """Guard the asymmetry itself, not just each side in isolation."""
        sync_client, _, _, _ = mock_nim
        encoder.encode_queries(["test"])
        encoder.encode_documents(["test"])
        input_types = [
            call.kwargs["extra_body"]["input_type"]
            for call in sync_client.embeddings.create.call_args_list
        ]
        assert input_types == ["query", "passage"]

    def test_call_delegates_to_encode_queries(self, encoder, mock_nim):
        sync_client, _, _, _ = mock_nim
        assert encoder(["test"]) == [[0.1, 0.2, 0.3]]
        assert (
            sync_client.embeddings.create.call_args.kwargs["extra_body"]["input_type"]
            == "query"
        )

    def test_handles_multiple_inputs_correctly(self, encoder, mock_nim):
        sync_client, _, _, _ = mock_nim
        sync_client.embeddings.create.return_value = embed_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        assert encoder.encode_documents(["test1", "test2"]) == [[0.1, 0.2], [0.3, 0.4]]
        assert sync_client.embeddings.create.call_args.kwargs["input"] == [
            "test1",
            "test2",
        ]

    def test_kwargs_forwarded(self, encoder, mock_nim):
        sync_client, _, _, _ = mock_nim
        encoder.encode_queries(["test"], dimensions=256)
        assert sync_client.embeddings.create.call_args.kwargs["dimensions"] == 256

    def test_extra_body_is_merged_not_duplicated(self, encoder, mock_nim):
        """A caller-supplied extra_body must merge, not raise a duplicate kwarg."""
        sync_client, _, _, _ = mock_nim
        encoder.encode_queries(["test"], extra_body={"truncate": "END"})
        assert sync_client.embeddings.create.call_args.kwargs["extra_body"] == {
            "input_type": "query",
            "truncate": "END",
        }

    def test_extra_body_can_override_input_type(self, encoder, mock_nim):
        sync_client, _, _, _ = mock_nim
        encoder.encode_documents(["test"], extra_body={"input_type": "query"})
        assert sync_client.embeddings.create.call_args.kwargs["extra_body"] == {
            "input_type": "query"
        }

    def test_raises_error_on_api_failure(self, encoder, mock_nim):
        sync_client, _, _, _ = mock_nim
        sync_client.embeddings.create.side_effect = Exception("API call failed")
        with pytest.raises(ValueError, match="Nim API call failed"):
            encoder.encode_queries(["test"])

    def test_raises_error_on_empty_response(self, encoder, mock_nim):
        sync_client, _, _, _ = mock_nim
        sync_client.embeddings.create.return_value = embed_response([])
        with pytest.raises(ValueError, match="Nim API call failed"):
            encoder.encode_queries(["test"])


class TestNimEncoderAsync:
    @pytest.mark.asyncio
    async def test_aencode_queries_uses_query_input_type(self, encoder, mock_nim):
        _, async_client, _, _ = mock_nim
        assert await encoder.aencode_queries(["test"]) == [[0.1, 0.2, 0.3]]
        async_client.embeddings.create.assert_awaited_once_with(
            input=["test"],
            model="nvidia/nv-embedqa-e5-v5",
            extra_body={"input_type": "query"},
        )

    @pytest.mark.asyncio
    async def test_aencode_documents_uses_passage_input_type(self, encoder, mock_nim):
        _, async_client, _, _ = mock_nim
        assert await encoder.aencode_documents(["test"]) == [[0.1, 0.2, 0.3]]
        async_client.embeddings.create.assert_awaited_once_with(
            input=["test"],
            model="nvidia/nv-embedqa-e5-v5",
            extra_body={"input_type": "passage"},
        )

    @pytest.mark.asyncio
    async def test_acall_delegates_to_aencode_queries(self, encoder, mock_nim):
        _, async_client, _, _ = mock_nim
        assert await encoder.acall(["test"]) == [[0.1, 0.2, 0.3]]
        assert (
            async_client.embeddings.create.call_args.kwargs["extra_body"]["input_type"]
            == "query"
        )

    @pytest.mark.asyncio
    async def test_handles_multiple_inputs_correctly(self, encoder, mock_nim):
        _, async_client, _, _ = mock_nim
        async_client.embeddings.create.return_value = embed_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        result = await encoder.aencode_documents(["test1", "test2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_extra_body_is_merged_not_duplicated(self, encoder, mock_nim):
        _, async_client, _, _ = mock_nim
        await encoder.aencode_documents(["test"], extra_body={"truncate": "END"})
        assert async_client.embeddings.create.call_args.kwargs["extra_body"] == {
            "input_type": "passage",
            "truncate": "END",
        }

    @pytest.mark.asyncio
    async def test_uses_async_client(self, encoder, mock_nim):
        """The async path must not fall back to the blocking client."""
        sync_client, _, _, _ = mock_nim
        await encoder.aencode_queries(["test"])
        sync_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_error_on_api_failure(self, encoder, mock_nim):
        _, async_client, _, _ = mock_nim
        async_client.embeddings.create.side_effect = Exception("API call failed")
        with pytest.raises(ValueError, match="Nim API call failed"):
            await encoder.aencode_documents(["test"])
