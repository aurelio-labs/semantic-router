import pytest
from unittest.mock import Mock
from semantic_router.text import Conversation, Message
from semantic_router.splitters.consecutive_sim import ConsecutiveSimSplitter
from semantic_router.splitters.cumulative_sim import CumulativeSimSplitter
from semantic_router.encoders.cohere import (
    CohereEncoder,
)  # Adjust this import based on your project structure


@pytest.fixture
def conversation_instance():
    return Conversation()


@pytest.fixture
def cohere_encoder():
    # Initialize CohereEncoder with necessary arguments
    encoder = CohereEncoder(
        name="cohere_encoder", cohere_api_key="dummy_key", input_type="text"
    )
    return encoder


def test_add_new_messages(conversation_instance):
    initial_len = len(conversation_instance.messages)
    conversation_instance.add_new_messages([Message(role="user", content="Hello")])
    assert len(conversation_instance.messages) == initial_len + 1


def test_remove_topics(conversation_instance):
    conversation_instance.topics.append((1, "Sample Topic"))
    conversation_instance.remove_topics()
    assert len(conversation_instance.topics) == 0


def test_configure_splitter_consecutive_similarity(
    conversation_instance, cohere_encoder
):
    conversation_instance.configure_splitter(
        encoder=cohere_encoder, threshold=0.5, split_method="consecutive_similarity"
    )
    assert isinstance(conversation_instance.splitter, ConsecutiveSimSplitter)


def test_configure_splitter_cumulative_similarity(
    conversation_instance, cohere_encoder
):
    conversation_instance.configure_splitter(
        encoder=cohere_encoder, threshold=0.5, split_method="cumulative_similarity"
    )
    assert isinstance(conversation_instance.splitter, CumulativeSimSplitter)


def test_configure_splitter_invalid_method(conversation_instance, cohere_encoder):
    with pytest.raises(ValueError):
        conversation_instance.configure_splitter(
            encoder=cohere_encoder, threshold=0.5, split_method="invalid_method"
        )


def test_split_by_topic_without_configuring_splitter(conversation_instance):
    with pytest.raises(ValueError):
        conversation_instance.split_by_topic()


def test_split_by_topic_with_no_unclustered_messages(
    conversation_instance, cohere_encoder, capsys
):
    conversation_instance.configure_splitter(
        encoder=cohere_encoder, threshold=0.5, split_method="consecutive_similarity"
    )
    conversation_instance.splitter = Mock()
    conversation_instance.split_by_topic()
    captured = capsys.readouterr()
    assert "No unclustered messages to process." in captured.out
