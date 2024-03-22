from unittest.mock import Mock, create_autospec

import numpy as np
import pytest

from semantic_router.encoders.base import BaseEncoder
from semantic_router.encoders.cohere import CohereEncoder
from semantic_router.schema import Message
from semantic_router.splitters.base import BaseSplitter
from semantic_router.splitters.consecutive_sim import ConsecutiveSimSplitter
from semantic_router.splitters.cumulative_sim import CumulativeSimSplitter
from semantic_router.text import Conversation


def test_consecutive_sim_splitter():
    # Create a Mock object for the encoder
    mock_encoder = Mock()
    mock_encoder.return_value = np.array([[1, 0], [1, 0.1], [0, 1]])

    cohere_encoder = CohereEncoder(
        name="",
        cohere_api_key="a",
        input_type="",
    )
    # Instantiate the ConsecutiveSimSplitter with the mock encoder
    splitter = ConsecutiveSimSplitter(encoder=cohere_encoder, score_threshold=0.9)
    splitter.encoder = mock_encoder

    # Define some documents
    docs = ["doc1", "doc2", "doc3"]

    # Use the splitter to split the documents
    splits = splitter(docs)

    # Verify the splits
    assert len(splits) == 2, "Expected two splits based on the similarity threshold"
    assert splits[0].docs == [
        "doc1",
        "doc2",
    ], "First split does not match expected documents"
    assert splits[1].docs == ["doc3"], "Second split does not match expected documents"


def test_cumulative_sim_splitter():
    # Mock the BaseEncoder
    mock_encoder = Mock()
    # Adjust the side_effect to simulate the encoder's behavior for cumulative document comparisons
    # This simplistic simulation assumes binary embeddings for demonstration purposes
    # Define a side_effect function for the mock encoder
    mock_encoder.side_effect = lambda x: (
        [[0.5, 0]] if "doc1" in x or "doc1\ndoc2" in x or "doc2" in x else [[0, 0.5]]
    )

    # Instantiate the CumulativeSimSplitter with the mock encoder
    cohere_encoder = CohereEncoder(
        name="",
        cohere_api_key="a",
        input_type="",
    )
    splitter = CumulativeSimSplitter(encoder=cohere_encoder, score_threshold=0.9)
    splitter.encoder = mock_encoder

    # Define some documents
    docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    # Use the splitter to split the documents
    splits = splitter(docs)

    # Verify the splits
    # The expected outcome needs to match the logic defined in your mock_encoder's side_effect
    assert len(splits) == 2, f"{len(splits)}"
    assert splits[0].docs == [
        "doc1",
        "doc2",
    ], "First split does not match expected documents"
    assert splits[1].docs == [
        "doc3",
        "doc4",
        "doc5",
    ], "Second split does not match expected documents"


def test_split_by_topic_consecutive_similarity():
    mock_encoder = Mock()
    mock_encoder.return_value = [[0.5, 0], [0, 0.5]]

    messages = [
        Message(role="User", content="What is the latest news?"),
        Message(role="Assistant", content="How is the weather today?"),
    ]
    conversation = Conversation(messages=messages)

    cohere_encoder = CohereEncoder(
        name="",
        cohere_api_key="a",
        input_type="",
    )

    conversation.configure_splitter(
        encoder=cohere_encoder, threshold=0.5, split_method="consecutive_similarity"
    )
    conversation.splitter.encoder = mock_encoder

    topics, new_topics = conversation.split_by_topic()

    assert len(new_topics) == 2
    assert new_topics[0].docs == ["User: What is the latest news?"]
    assert new_topics[1].docs == ["Assistant: How is the weather today?"]


def test_split_by_topic_cumulative_similarity():
    mock_encoder = Mock()
    mock_encoder.side_effect = lambda x: (
        [[0.5, 0]] if "User: What is the latest news?" in x else [[0, 0.5]]
    )

    messages = [
        Message(role="User", content="What is the latest news?"),
        Message(role="Assistant", content="How is the weather today?"),
    ]
    conversation = Conversation(messages=messages)

    cohere_encoder = CohereEncoder(
        name="",
        cohere_api_key="a",
        input_type="",
    )

    conversation.configure_splitter(
        encoder=cohere_encoder, threshold=0.5, split_method="cumulative_similarity"
    )
    conversation.splitter.encoder = mock_encoder

    topics, new_topics = conversation.split_by_topic()

    # Assertions may need to be adjusted based on the expected behavior of the cumulative similarity splitter
    assert len(new_topics) == 2


def test_split_by_topic_no_messages():
    mock_encoder = create_autospec(BaseEncoder)
    conversation = Conversation()
    conversation.configure_splitter(
        encoder=mock_encoder, threshold=0.5, split_method="consecutive_similarity"
    )

    topics, new_topics = conversation.split_by_topic()

    assert len(new_topics) == 0
    assert len(topics) == 0


def test_split_by_topic_without_configuring_splitter():
    conversation = Conversation(messages=[Message(role="User", content="Hello")])

    with pytest.raises(ValueError):
        conversation.split_by_topic()


def test_consecutive_similarity_splitter_single_doc():
    mock_encoder = create_autospec(BaseEncoder)
    # Assuming any return value since it should not reach the point of using the encoder
    mock_encoder.return_value = np.array([[0.5, 0]])

    splitter = ConsecutiveSimSplitter(encoder=mock_encoder, score_threshold=0.5)

    docs = ["doc1"]
    with pytest.raises(ValueError) as excinfo:
        splitter(docs)
    assert "at least two are required" in str(excinfo.value)


def test_cumulative_similarity_splitter_single_doc():
    mock_encoder = create_autospec(BaseEncoder)
    # Assuming any return value since it should not reach the point of using the encoder
    mock_encoder.return_value = np.array([[0.5, 0]])

    splitter = CumulativeSimSplitter(encoder=mock_encoder, score_threshold=0.5)

    docs = ["doc1"]
    with pytest.raises(ValueError) as excinfo:
        splitter(docs)
    assert "at least two are required" in str(excinfo.value)


@pytest.fixture
def base_splitter_instance():
    # Now MockEncoder includes default values for required fields
    mock_encoder = Mock(spec=BaseEncoder)
    mock_encoder.name = "mock_encoder"
    mock_encoder.score_threshold = 0.5
    return BaseSplitter(name="test_splitter", encoder=mock_encoder, score_threshold=0.5)


def test_base_splitter_call_not_implemented(base_splitter_instance):
    with pytest.raises(NotImplementedError):
        base_splitter_instance(["document"])
