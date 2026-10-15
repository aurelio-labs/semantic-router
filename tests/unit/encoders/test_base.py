import pytest

from semantic_router.encoders import DenseEncoder


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
