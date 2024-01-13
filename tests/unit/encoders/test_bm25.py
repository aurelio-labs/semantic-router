import pytest

from semantic_router.encoders import BM25Encoder


@pytest.fixture
def bm25_encoder():
    sparse_encoder = BM25Encoder(use_default_params=False)
    sparse_encoder.fit(
        ["The quick brown fox", "jumps over the lazy dog", "Hello, world!"]
    )
    return sparse_encoder


class TestBM25Encoder:
    def test_initialization(self, bm25_encoder):
        assert len(bm25_encoder.idx_mapping) != 0

    def test_fit(self, bm25_encoder):
        bm25_encoder.fit(["some docs", "and more docs", "and even more docs"])
        assert len(bm25_encoder.idx_mapping) != 0

    def test_call_method(self, bm25_encoder):
        result = bm25_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sublist, list) for sublist in result
        ), "Each item in result should be a list"

    def test_call_method_no_docs_bm25_encoder(self, bm25_encoder):
        with pytest.raises(ValueError):
            bm25_encoder([])

    def test_call_method_no_word(self, bm25_encoder):
        result = bm25_encoder(["doc with fake word gta5jabcxyz"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sublist, list) for sublist in result
        ), "Each item in result should be a list"

    def test_init_with_non_dict_doc_freq(self, mocker):
        mock_encoder = mocker.MagicMock()
        mock_encoder.get_params.return_value = {"doc_freq": "not a dict"}
        mocker.patch(
            "pinecone_text.sparse.BM25Encoder.default", return_value=mock_encoder
        )
        with pytest.raises(TypeError):
            BM25Encoder()

    def test_call_method_with_uninitialized_model_or_mapping(self, bm25_encoder):
        bm25_encoder.model = None
        with pytest.raises(ValueError):
            bm25_encoder(["test"])

    def test_fit_with_uninitialized_model(self, bm25_encoder):
        bm25_encoder.model = None
        with pytest.raises(ValueError):
            bm25_encoder.fit(["test"])
