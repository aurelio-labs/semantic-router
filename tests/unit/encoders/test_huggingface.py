import pytest
import numpy as np
from semantic_router.encoders.huggingface import HuggingFaceEncoder


class TestHuggingFaceEncoder:
    def test_huggingface_encoder(self):
        encoder = HuggingFaceEncoder()
        test_docs = ["This is a test", "This is another test"]
        embeddings = encoder(test_docs)
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(test_docs)
        assert all(isinstance(embedding, list) for embedding in embeddings)
        assert all(len(embedding) > 0 for embedding in embeddings)

    def test_huggingface_encoder_normalized_embeddings(self):
        encoder = HuggingFaceEncoder()
        docs = ["This is a test document.", "Another test document."]
        unnormalized_embeddings = encoder(docs, normalize_embeddings=False)
        normalized_embeddings = encoder(docs, normalize_embeddings=True)
        assert len(unnormalized_embeddings) == len(normalized_embeddings)

        for unnormalized, normalized in zip(
            unnormalized_embeddings, normalized_embeddings
        ):
            norm_unnormalized = np.linalg.norm(unnormalized, ord=2)
            norm_normalized = np.linalg.norm(normalized, ord=2)
            # Ensure the norm of the normalized embeddings is approximately 1
            assert np.isclose(norm_normalized, 1.0)
            # Ensure the normalized embeddings are actually normalized versions of unnormalized embeddings
            np.testing.assert_allclose(
                normalized,
                np.divide(unnormalized, norm_unnormalized),
                rtol=1e-5,
                atol=1e-5,  # Adjust tolerance levels
            )

    def test_huggingface_encoder_invalid_pooling_strategy(self):
        encoder = HuggingFaceEncoder()
        docs = ["This is a test document.", "Another test document."]
        with pytest.raises(ValueError):
            encoder(docs, pooling_strategy="invalid_strategy")
