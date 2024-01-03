import pytest

from semantic_router.schemas.encoder import (
    CohereEncoder,
    Encoder,
    EncoderType,
    OpenAIEncoder,
)


class TestEncoderDataclass:
    def test_encoder_initialization_openai(self, mocker):
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
        encoder = Encoder(type="openai", name="test-engine")
        assert encoder.type == EncoderType.OPENAI
        assert isinstance(encoder.model, OpenAIEncoder)

    def test_encoder_initialization_cohere(self, mocker):
        mocker.patch.dict("os.environ", {"COHERE_API_KEY": "test"})
        encoder = Encoder(type="cohere", name="test-engine")
        assert encoder.type == EncoderType.COHERE
        assert isinstance(encoder.model, CohereEncoder)

    def test_encoder_initialization_unsupported_type(self):
        with pytest.raises(ValueError):
            Encoder(type="unsupported", name="test-engine")

    def test_encoder_initialization_huggingface(self):
        with pytest.raises(NotImplementedError):
            Encoder(type="huggingface", name="test-engine")

    def test_encoder_call_method(self, mocker):
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
        mocker.patch(
            "semantic_router.encoders.openai.OpenAIEncoder.__call__",
            return_value=[0.1, 0.2, 0.3],
        )
        encoder = Encoder(type="openai", name="test-engine")
        result = encoder(["test"])
        assert result == [0.1, 0.2, 0.3]
