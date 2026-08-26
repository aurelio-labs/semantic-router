import sys
from types import SimpleNamespace

import pytest

from semantic_router.encoders import FastEmbedEncoder


class TestFastEmbedEncoder:
    def test_local_files_only_is_passed_to_fastembed(self, monkeypatch):
        captured = {}

        class FakeTextEmbedding:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setitem(
            sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        )

        FastEmbedEncoder(local_files_only=True)

        assert captured["local_files_only"] is True

    def test_fastembed_encoder(self):
        pytest.importorskip("fastembed")
        encode = FastEmbedEncoder()
        test_docs = ["This is a test", "This is another test"]
        embeddings = encode(test_docs)
        assert isinstance(embeddings, list)
