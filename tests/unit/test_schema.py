import pytest
from pydantic.v1 import ValidationError

from semantic_router.schema import (
    CohereEncoder,
    Encoder,
    EncoderType,
    Message,
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


class TestMessageDataclass:
    def test_message_creation(self):
        message = Message(role="user", content="Hello!")
        assert message.role == "user"
        assert message.content == "Hello!"

        with pytest.raises(ValidationError):
            Message(user_role="invalid_role", message="Hello!")

    def test_message_to_openai(self):
        message = Message(role="user", content="Hello!")
        openai_format = message.to_openai()
        assert openai_format == {"role": "user", "content": "Hello!"}

        message = Message(role="invalid_role", content="Hello!")
        with pytest.raises(ValueError):
            message.to_openai()

    def test_message_to_cohere(self):
        message = Message(role="user", content="Hello!")
        cohere_format = message.to_cohere()
        assert cohere_format == {"role": "user", "message": "Hello!"}
