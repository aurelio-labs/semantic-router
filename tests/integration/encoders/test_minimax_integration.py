import pytest

from semantic_router.encoders.minimax import MiniMaxEncoder

# Calls the paid MiniMax API, so it runs with the other live tests only.
pytestmark = pytest.mark.live("MINIMAX_API_KEY")


@pytest.fixture
def minimax_encoder():
    return MiniMaxEncoder()


class TestMiniMaxEncoderIntegration:
    def test_encoder_init_success(self, minimax_encoder):
        assert minimax_encoder._api_key is not None
        assert minimax_encoder.name == "embo-01"

    def test_encoder_dims(self, minimax_encoder):
        embeddings = minimax_encoder(["test document"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

    def test_encoder_multiple_docs(self, minimax_encoder):
        embeddings = minimax_encoder(["hello world", "foo bar baz"])
        assert len(embeddings) == 2
