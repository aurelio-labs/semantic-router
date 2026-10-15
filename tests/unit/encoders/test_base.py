import numpy as np
import pytest

from semantic_router.encoders import DenseEncoder, SparseEncoder


class TestDenseEncoder:
    @pytest.fixture
    def base_encoder(self):
        return DenseEncoder(name="TestEncoder")

    def test_base_encoder_initialization(self, base_encoder):
        assert base_encoder.name == "TestEncoder", "Initialization of name failed"

    def test_base_encoder_has_no_score_threshold(self):
        # v1 routers threshold on confidence, not on the encoder
        assert "score_threshold" not in DenseEncoder.model_fields

    def test_base_encoder_call_method_not_implemented(self, base_encoder):
        with pytest.raises(NotImplementedError):
            base_encoder(["some", "texts"])


class TestSparseEncoder:
    @pytest.fixture
    def base_encoder(self):
        return SparseEncoder(name="TestEncoder")

    def test_one_embedding_per_row(self, base_encoder):
        # all-zero rows (no token in the corpus) must keep their place in the
        # batch, including at the end
        arrays = np.array(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
            ]
        )
        embeds = base_encoder._array_to_sparse_embeddings(arrays)
        assert [e.to_dict() for e in embeds] == [
            {1: 2.0},
            {0: 0.0},
            {0: 1.0, 2: 3.0},
            {0: 0.0},
        ]

    def test_all_zero_rows_are_zero_embeddings(self, base_encoder):
        embeds = base_encoder._array_to_sparse_embeddings(np.zeros((3, 5)))
        assert [e.embedding.tolist() for e in embeds] == [[[0.0, 0.0]]] * 3
