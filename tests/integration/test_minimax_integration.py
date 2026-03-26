import os
import time

import pytest

from semantic_router.encoders.minimax import MiniMaxEncoder
from semantic_router.llms.minimax import MiniMaxLLM
from semantic_router.schema import Message


def has_valid_minimax_api_key():
    """Check if a valid MiniMax API key is available."""
    api_key = os.environ.get("MINIMAX_API_KEY")
    return api_key is not None and api_key.strip() != ""


@pytest.fixture
def minimax_encoder():
    if not has_valid_minimax_api_key():
        pytest.skip("MINIMAX_API_KEY not set")
    return MiniMaxEncoder()


@pytest.fixture
def minimax_llm():
    if not has_valid_minimax_api_key():
        pytest.skip("MINIMAX_API_KEY not set")
    return MiniMaxLLM()


class TestMiniMaxEncoderIntegration:
    @pytest.mark.skipif(
        not has_valid_minimax_api_key(), reason="MINIMAX_API_KEY required"
    )
    def test_encoder_init_success(self, minimax_encoder):
        assert minimax_encoder._api_key is not None
        assert minimax_encoder.name == "embo-01"

    @pytest.mark.skipif(
        not has_valid_minimax_api_key(), reason="MINIMAX_API_KEY required"
    )
    def test_encoder_dims(self, minimax_encoder):
        time.sleep(2)
        embeddings = minimax_encoder(["test document"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    @pytest.mark.skipif(
        not has_valid_minimax_api_key(), reason="MINIMAX_API_KEY required"
    )
    def test_encoder_multiple_docs(self, minimax_encoder):
        time.sleep(2)
        embeddings = minimax_encoder(["hello world", "foo bar baz"])
        assert len(embeddings) == 2


class TestMiniMaxLLMIntegration:
    @pytest.mark.skipif(
        not has_valid_minimax_api_key(), reason="MINIMAX_API_KEY required"
    )
    def test_llm_init_success(self, minimax_llm):
        assert minimax_llm._client is not None
        assert minimax_llm.name == "MiniMax-M2.5"

    @pytest.mark.skipif(
        not has_valid_minimax_api_key(), reason="MINIMAX_API_KEY required"
    )
    def test_llm_call(self, minimax_llm):
        time.sleep(2)
        messages = [Message(role="user", content="Say hello")]
        result = minimax_llm(messages)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skipif(
        not has_valid_minimax_api_key(), reason="MINIMAX_API_KEY required"
    )
    @pytest.mark.asyncio
    async def test_llm_acall(self, minimax_llm):
        time.sleep(2)
        messages = [Message(role="user", content="Say hello")]
        result = await minimax_llm.acall(messages)
        assert isinstance(result, str)
        assert len(result) > 0
