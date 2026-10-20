import pytest

from semantic_router.index.local import LocalIndex


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
        utterances = index.get_utterances(include_metadata=True)
        assert len(utterances) == 2
        assert utterances[0].metadata == {"source": "one"}
        assert utterances[1].metadata == {"source": "two"}
