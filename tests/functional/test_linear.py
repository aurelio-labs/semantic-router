import numpy as np
import pytest

from semantic_router.linear import similarity_matrix, top_scores


@pytest.fixture
def ident_vector():
    return np.identity(10)[0]


@pytest.fixture
def test_index():
    return np.array([[3, 0, 0], [2, 1, 0], [0, 1, 0]])


def test_similarity_matrix__dimensionality():
    """Test that the similarity matrix is square."""
    xq = np.random.random((10,))  # 10-dimensional embedding vector
    index = np.random.random((100, 10))
    S = similarity_matrix(xq, index)
    assert S.shape == (100,)


def test_similarity_matrix__is_norm_max(ident_vector):
    """
    Using identical vectors should yield a maximum similarity of 1
    """
    index = np.repeat(np.atleast_2d(ident_vector), 3, axis=0)
    sim = similarity_matrix(ident_vector, index)
    assert sim.max() == 1.0


def test_similarity_matrix__is_norm_min(ident_vector):
    """
    Using orthogonal vectors should yield a minimum similarity of 0
    """
    orth_v = np.roll(np.atleast_2d(ident_vector), 1)
    index = np.repeat(orth_v, 3, axis=0)
    sim = similarity_matrix(ident_vector, index)
    assert sim.min() == 0.0


def test_top_scores__is_sorted(test_index):
    """
    Test that the top_scores function returns a sorted list of scores.
    """

    xq = test_index[0]  # should have max similarity

    sim = similarity_matrix(xq, test_index)
    _, idx = top_scores(sim, 3)

    # Scores and indexes should be sorted ascending
    assert np.array_equal(idx, np.array([2, 1, 0]))


def test_top_scores__scores(test_index):
    """
    Test that for a known vector and a known index, the top_scores function
    returns exactly the expected scores.
    """
    xq = test_index[0]  # should have max similarity

    sim = similarity_matrix(xq, test_index)
    scores, _ = top_scores(sim, 3)

    # Scores and indexes should be sorted ascending
    assert np.allclose(scores, np.array([0.0, 0.89442719, 1.0]))
