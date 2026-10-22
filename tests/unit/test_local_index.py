import numpy as np
import pytest

from semantic_router.index.hybrid_local import HybridLocalIndex
from semantic_router.index.local import LocalIndex
from semantic_router.schema import SparseEmbedding


class TestLocalIndex:
    def test_local_index_add_metadata_length_mismatch(self):
        index = LocalIndex()
        with pytest.raises(
            ValueError,
            match=r"metadata_list length \(1\) must match embeddings length \(2\)",
        ):
            index.add(
                embeddings=[[1.0, 0.0], [0.0, 1.0]],
                routes=["a", "b"],
                utterances=["one", "two"],
                metadata_list=[{"source": "one"}],
            )

    def test_local_index_add_matching_metadata(self):
        index = LocalIndex()
        index.add(
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            routes=["a", "b"],
            utterances=["one", "two"],
            metadata_list=[{"source": "one"}, {"source": "two"}],
        )
        assert len(index) == 2
        utterances = index.get_utterances()
        assert len(utterances) == 2
        assert utterances[0].metadata == {"source": "one"}
        assert utterances[1].metadata == {"source": "two"}

    def test_adding_a_pair_again_replaces_it(self):
        index = LocalIndex()
        index.add(
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            routes=["a", "b"],
            utterances=["one", "two"],
            metadata_list=[{"v": 1}, {"v": 1}],
        )
        index.add(
            embeddings=[[0.6, 0.8]],
            routes=["a"],
            utterances=["one"],
            metadata_list=[{"v": 2}],
        )
        assert len(index) == 2
        assert [u.metadata for u in index.get_utterances()] == [{"v": 2}, {"v": 1}]
        assert index.index.tolist()[0] == [0.6, 0.8]


def hybrid_index() -> HybridLocalIndex:
    index = HybridLocalIndex()
    index.add(
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],
        routes=["a", "b", "c"],
        utterances=["one", "two", "three"],
        sparse_embeddings=[
            SparseEmbedding.from_dict({0: 1.0}),
            SparseEmbedding.from_dict({1: 1.0}),
            SparseEmbedding.from_dict({2: 1.0}),
        ],
    )
    return index


def hybrid_query(index: HybridLocalIndex, dense, sparse: dict, alpha: float):
    """query the way IndexedRouter does: alpha on dense, the rest on sparse"""
    scores, routes = index.query(  # type: ignore[misc]
        vector=np.array(dense) * alpha,
        sparse_vector={k: v * (1 - alpha) for k, v in sparse.items()},
        top_k=3,
    )
    return dict(zip(routes, scores.tolist()))


class TestHybridLocalIndex:
    def test_alpha_weights_the_dense_side(self):
        index = hybrid_index()
        low = hybrid_query(index, [1.0, 0.0], {1: 1.0}, alpha=0.2)
        high = hybrid_query(index, [1.0, 0.0], {1: 1.0}, alpha=0.8)
        # route a matches on dense only, so its score is alpha * cos = alpha
        assert low["a"] == pytest.approx(0.2)
        assert high["a"] == pytest.approx(0.8)
        # route b matches on sparse only, so its score is 1 - alpha
        assert low["b"] == pytest.approx(0.8)
        assert high["b"] == pytest.approx(0.2)

    def test_scores_stay_at_or_below_one_with_unit_norm_inputs(self):
        index = hybrid_index()
        for alpha in (0.0, 0.3, 0.5, 1.0):
            # a query identical to record c on both sides is the best possible match
            scores = hybrid_query(index, [0.6, 0.8], {2: 1.0}, alpha=alpha)
            assert max(scores.values()) == pytest.approx(1.0)
            assert all(s <= 1.0 + 1e-9 for s in scores.values())

    def test_stores_metadata(self):
        index = HybridLocalIndex()
        index.add(
            embeddings=[[1.0, 0.0]],
            routes=["a"],
            utterances=["one"],
            metadata_list=[{"sr_description": "A", "team": "x"}],
            sparse_embeddings=[SparseEmbedding.from_dict({0: 1.0})],
        )
        (route,) = index.get_routes()
        assert route.description == "A"
        assert route.metadata == {"team": "x"}

    def test_delete_keeps_sparse_rows_aligned(self):
        index = hybrid_index()
        index.delete("a")
        assert len(index.sparse_index or []) == len(index) == 2
        # the sparse side of b still lines up with b after a's row is gone
        scores = hybrid_query(index, [0.0, 0.0], {1: 1.0}, alpha=0.0)
        assert max(scores, key=lambda r: scores[r]) == "b"

    def test_adding_a_pair_again_replaces_its_sparse_row(self):
        index = hybrid_index()
        index.add(
            embeddings=[[1.0, 0.0]],
            routes=["a"],
            utterances=["one"],
            sparse_embeddings=[SparseEmbedding.from_dict({5: 1.0})],
        )
        assert len(index) == len(index.sparse_index or []) == 3
        scores = hybrid_query(index, [0.0, 0.0], {5: 1.0}, alpha=0.0)
        assert scores["a"] == pytest.approx(1.0)
