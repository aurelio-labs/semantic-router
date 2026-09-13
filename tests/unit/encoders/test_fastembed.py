import pytest

from semantic_router.encoders import FastEmbedEncoder

_ = pytest.importorskip("fastembed")


class TestFastEmbedEncoder:
    def test_fastembed_encoder(self):
        encode = FastEmbedEncoder()
        test_docs = ["This is a test", "This is another test"]
        embeddings = encode(test_docs)
        assert isinstance(embeddings, list)

    @pytest.mark.asyncio
    async def test_fastembed_encoder_acall(self):
        encode = FastEmbedEncoder()
        test_docs = ["This is a test", "This is another test"]
        sync_embeddings = encode(test_docs)
        async_embeddings = await encode.acall(test_docs)
        assert isinstance(async_embeddings, list)
        assert async_embeddings == sync_embeddings
