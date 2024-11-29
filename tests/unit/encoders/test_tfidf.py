import numpy as np
import pytest

from semantic_router.encoders import TfidfEncoder
from semantic_router.route import Route


@pytest.fixture
def tfidf_encoder():
    return TfidfEncoder()


class TestTfidfEncoder:
    def test_initialization(self, tfidf_encoder):
        assert tfidf_encoder.word_index == {}
        assert (tfidf_encoder.idf == np.array([])).all()

    def test_fit(self, tfidf_encoder):
        routes = [
            Route(
                name="test_route",
                utterances=["some docs", "and more docs", "and even more docs"],
            )
        ]
        tfidf_encoder.fit(routes)
        assert tfidf_encoder.word_index != {}
        assert not np.array_equal(tfidf_encoder.idf, np.array([]))

    def test_call_method(self, tfidf_encoder):
        routes = [
            Route(
                name="test_route",
                utterances=["some docs", "and more docs", "and even more docs"],
            )
        ]
        tfidf_encoder.fit(routes)
        result = tfidf_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sparse_emb.embedding, np.ndarray) for sparse_emb in result
        ), "Each item in result should be an array"

    def test_call_method_no_docs_tfidf_encoder(self, tfidf_encoder):
        with pytest.raises(ValueError):
            tfidf_encoder([])

    def test_call_method_no_word(self, tfidf_encoder):
        routes = [
            Route(
                name="test_route",
                utterances=["some docs", "and more docs", "and even more docs"],
            )
        ]
        tfidf_encoder.fit(routes)
        result = tfidf_encoder(["doc with fake word gta5jabcxyz"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sparse_emb.embedding, np.ndarray) for sparse_emb in result
        ), "Each item in result should be an array"

    def test_fit_with_strings(self, tfidf_encoder):
        routes = ["test a", "test b", "test c"]
        with pytest.raises(TypeError):
            tfidf_encoder.fit(routes)

    def test_call_method_with_uninitialized_model(self, tfidf_encoder):
        with pytest.raises(ValueError):
            tfidf_encoder(["test"])

    def test_compute_tf_no_word_index(self, tfidf_encoder):
        with pytest.raises(ValueError, match="Word index is not initialized."):
            tfidf_encoder._compute_tf(["some docs"])

    def test_compute_tf_with_word_in_word_index(self, tfidf_encoder):
        routes = [
            Route(
                name="test_route",
                utterances=["some docs", "and more docs", "and even more docs"],
            )
        ]
        tfidf_encoder.fit(routes)
        tf = tfidf_encoder._compute_tf(["some docs"])
        assert tf.shape == (1, len(tfidf_encoder.word_index))

    def test_compute_idf_no_word_index(self, tfidf_encoder):
        with pytest.raises(ValueError, match="Word index is not initialized."):
            tfidf_encoder._compute_idf(["some docs"])
