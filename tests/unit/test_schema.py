import pytest
from pydantic import ValidationError

from semantic_router.schema import (
    Message,
)


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
