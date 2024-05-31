from semantic_router.encoders import FastEmbedEncoder


class TestFastEmbedEncoder:
    def test_fastembed_encoder(self):
        if False:
            # temporarily disabled until fastembed added back as dep
            encode = FastEmbedEncoder()
            test_docs = ["This is a test", "This is another test"]

            embeddings = encode(test_docs)
            assert isinstance(embeddings, list)
