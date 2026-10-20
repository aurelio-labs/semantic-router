import subprocess
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from aurelio_sdk import AsyncAurelioClient, AurelioClient, EmbeddingResponse

from semantic_router.encoders import AurelioSparseEncoder


def embed_response(embeddings: list[dict]) -> EmbeddingResponse:
    """Build a real Aurelio embedding response so tests fail if the SDK shape changes.

    :param embeddings: One ``{"indices": [...], "values": [...]}`` per input document.
    :type embeddings: list[dict]
    :return: An Aurelio embedding response.
    :rtype: EmbeddingResponse
    """
    return EmbeddingResponse.model_validate(
        {
            "model": "bm25",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
            "data": [{"index": i, "embedding": e} for i, e in enumerate(embeddings)],
        }
    )


@pytest.fixture
def encoder(mocker):
    enc = AurelioSparseEncoder(api_key="test_api_key")
    response = embed_response([{"indices": [1, 5], "values": [0.5, 0.25]}])
    mocker.patch.object(enc.client, "embedding", Mock(return_value=response))
    mocker.patch.object(enc.async_client, "embedding", AsyncMock(return_value=response))
    return enc


def test_import_does_not_configure_root_logger():
    """aurelio_sdk calls logging.basicConfig on import, so semantic_router must
    not import it until an AurelioSparseEncoder is created."""
    code = (
        "import logging, sys\n"
        "import semantic_router, semantic_router.encoders, semantic_router.schema\n"
        "assert logging.getLogger().handlers == [], logging.getLogger().handlers\n"
        "assert 'aurelio_sdk' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


class TestAurelioSparseEncoderInit:
    def test_initialization_with_api_key(self):
        enc = AurelioSparseEncoder(api_key="test_api_key")
        assert enc.name == "bm25"
        assert enc.type == "sparse"
        assert isinstance(enc.client, AurelioClient)
        assert isinstance(enc.async_client, AsyncAurelioClient)
        assert enc.client.api_key == "test_api_key"

    def test_initialization_with_env_api_key(self, monkeypatch):
        monkeypatch.setenv("AURELIO_API_KEY", "env_api_key")
        enc = AurelioSparseEncoder()
        assert enc.client.api_key == "env_api_key"

    def test_initialization_without_api_key(self, monkeypatch):
        monkeypatch.delenv("AURELIO_API_KEY", raising=False)
        with pytest.raises(ValueError, match="AURELIO_API_KEY"):
            AurelioSparseEncoder()

    def test_clients_are_not_serialized(self):
        enc = AurelioSparseEncoder(api_key="test_api_key")
        assert "client" not in enc.model_dump()
        assert "async_client" not in enc.model_dump()


class TestAurelioSparseEncoderSync:
    def test_encode_queries(self, encoder):
        result = encoder.encode_queries(["test"])
        assert [e.to_dict() for e in result] == [{1: 0.5, 5: 0.25}]
        encoder.client.embedding.assert_called_once_with(
            input=["test"], model="bm25", input_type="queries"
        )

    def test_encode_documents(self, encoder):
        result = encoder.encode_documents(["test"])
        assert [e.to_dict() for e in result] == [{1: 0.5, 5: 0.25}]
        assert encoder.client.embedding.call_args.kwargs["input_type"] == "documents"

    def test_call_delegates_to_encode_queries(self, encoder):
        encoder(["test"])
        assert encoder.client.embedding.call_args.kwargs["input_type"] == "queries"

    def test_fit_not_supported(self, encoder):
        with pytest.raises(NotImplementedError):
            encoder.fit(["test"])


class TestAurelioSparseEncoderAsync:
    @pytest.mark.asyncio
    async def test_aencode_queries(self, encoder):
        result = await encoder.aencode_queries(["test"])
        assert [e.to_dict() for e in result] == [{1: 0.5, 5: 0.25}]
        encoder.async_client.embedding.assert_awaited_once_with(
            input=["test"], model="bm25", input_type="queries"
        )

    @pytest.mark.asyncio
    async def test_aencode_documents(self, encoder):
        await encoder.aencode_documents(["test"])
        kwargs = encoder.async_client.embedding.call_args.kwargs
        assert kwargs["input_type"] == "documents"

    @pytest.mark.asyncio
    async def test_acall_delegates_to_aencode_queries(self, encoder):
        await encoder.acall(["test"])
        kwargs = encoder.async_client.embedding.call_args.kwargs
        assert kwargs["input_type"] == "queries"
