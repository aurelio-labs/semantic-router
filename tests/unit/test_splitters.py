import pytest
from unittest.mock import Mock
from semantic_router.utils.splitters import semantic_splitter


def test_semantic_splitter_consecutive_similarity_drop():
    # Mock the BaseEncoder
    mock_encoder = Mock()
    mock_encoder.return_value = [[0.5, 0], [0.5, 0], [0.5, 0], [0, 0.5], [0, 0.5]]

    docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    threshold = 0.5
    split_method = "consecutive_similarity_drop"

    result = semantic_splitter(mock_encoder, docs, threshold, split_method)

    assert result == {"split 1": ["doc1", "doc2", "doc3"], "split 2": ["doc4", "doc5"]}


def test_semantic_splitter_cumulative_similarity_drop():
    # Mock the BaseEncoder
    mock_encoder = Mock()
    mock_encoder.side_effect = (
        lambda x: [[0.5, 0]] if "doc1" in x or "doc1\ndoc2" in x else [[0, 0.5]]
    )

    docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    threshold = 0.5
    split_method = "cumulative_similarity_drop"

    result = semantic_splitter(mock_encoder, docs, threshold, split_method)

    assert result == {"split 1": ["doc1", "doc2"], "split 2": ["doc3", "doc4", "doc5"]}


def test_semantic_splitter_invalid_method():
    # Mock the BaseEncoder
    mock_encoder = Mock()

    docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    threshold = 0.5
    split_method = "invalid_method"

    with pytest.raises(ValueError):
        semantic_splitter(mock_encoder, docs, threshold, split_method)
