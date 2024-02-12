from unittest.mock import Mock

import pytest

from semantic_router.encoders.cohere import (
    CohereEncoder,
)

# Adjust this import based on your project structure
from semantic_router.schema import DocumentSplit
from semantic_router.splitters.consecutive_sim import ConsecutiveSimSplitter
from semantic_router.splitters.cumulative_sim import CumulativeSimSplitter
from semantic_router.text import Conversation, Message


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


def test_get_last_message_and_topic_id_with_no_topics(conversation_instance):
    # Test the method when there are no topics in the conversation
    last_topic_id, last_message = conversation_instance.get_last_message_and_topic_id()
    assert (
        last_topic_id is None and last_message is None
    ), "Expected None for both topic ID and message when there are no topics"


def test_get_last_message_and_topic_id_with_topics(conversation_instance):
    # Add some topics to the conversation instance
    conversation_instance.topics.append((0, "First message"))
    conversation_instance.topics.append((1, "Second message"))
    conversation_instance.topics.append((2, "Third message"))

    # Test the method when there are topics in the conversation
    last_topic_id, last_message = conversation_instance.get_last_message_and_topic_id()
    assert (
        last_topic_id == 2 and last_message == "Third message"
    ), "Expected last topic ID and message to match the last topic added"


def test_determine_topic_start_index_no_existing_topics(conversation_instance):
    # Scenario where there are no existing topics
    new_topics = [
        DocumentSplit(docs=["User: Hello!"], is_triggered=True, triggered_score=0.4)
    ]
    start_index = conversation_instance.determine_topic_start_index(
        new_topics, None, None
    )
    assert (
        start_index == 1
    ), "Expected start index to be 1 when there are no existing topics"


def test_determine_topic_start_index_with_existing_topics_not_including_last_message(
    conversation_instance,
):
    # Scenario where existing topics do not include the last message
    conversation_instance.topics.append((0, "First message"))
    new_topics = [
        DocumentSplit(docs=["User: Hello!"], is_triggered=True, triggered_score=0.4)
    ]
    start_index = conversation_instance.determine_topic_start_index(
        new_topics, 0, "Non-existent last message"
    )
    assert (
        start_index == 1
    ), "Expected start index to increment when last message is not in new topics"


def test_determine_topic_start_index_with_existing_topics_including_last_message(
    conversation_instance,
):
    # Scenario where the first new topic includes the last message
    conversation_instance.topics.append((0, "First message"))
    new_topics = [
        DocumentSplit(
            docs=["First message", "Another message"],
            is_triggered=True,
            triggered_score=0.4,
        )
    ]
    start_index = conversation_instance.determine_topic_start_index(
        new_topics, 0, "First message"
    )
    assert (
        start_index == 0
    ), "Expected start index to be the same as last topic ID when last message is included in new topics"


def test_determine_topic_start_index_increment_from_last_topic_id(
    conversation_instance,
):
    # Scenario to test increment from the last topic ID when last message is not in new topics
    conversation_instance.topics.append((1, "First message"))
    conversation_instance.topics.append((2, "Second message"))
    new_topics = [
        DocumentSplit(docs=["User: Hello!"], is_triggered=True, triggered_score=0.4)
    ]
    start_index = conversation_instance.determine_topic_start_index(
        new_topics, 2, "Non-existent last message"
    )
    assert start_index == 3, "Expected start index to be last topic ID + 1"
