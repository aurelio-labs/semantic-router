import os
from unittest.mock import patch

import numpy as np
import pytest

_ = pytest.importorskip("transformers")

from semantic_router.encoders.huggingface import HuggingFaceEncoder  # noqa: E402

test_model_name = "aurelio-ai/sr-test-huggingface"


class TestHuggingFaceEncoder:
    def test_huggingface_encoder_import_errors_transformers(self):
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError) as error:
                HuggingFaceEncoder()

        assert "Please install transformers to use HuggingFaceEncoder" in str(
            error.value
        )

    def test_huggingface_encoder_import_errors_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError) as error:
                HuggingFaceEncoder()

        assert "Please install transformers to use HuggingFaceEncoder" in str(
            error.value
        )

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_huggingface_encoder_mean_pooling(self):
        encoder = HuggingFaceEncoder(name=test_model_name)
        test_docs = ["This is a test", "This is another test"]
        embeddings = encoder(test_docs, pooling_strategy="mean")
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(test_docs)
        assert all(isinstance(embedding, list) for embedding in embeddings)
        assert all(len(embedding) > 0 for embedding in embeddings)

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_huggingface_encoder_max_pooling(self):
        encoder = HuggingFaceEncoder(name=test_model_name)
        test_docs = ["This is a test", "This is another test"]
        embeddings = encoder(test_docs, pooling_strategy="max")
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(test_docs)
        assert all(isinstance(embedding, list) for embedding in embeddings)
        assert all(len(embedding) > 0 for embedding in embeddings)

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_huggingface_encoder_normalized_embeddings(self):
        encoder = HuggingFaceEncoder(name=test_model_name)
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
