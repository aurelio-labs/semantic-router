import pytest

from semantic_router.encoders import DenseEncoder


class TestDenseEncoder:
    @pytest.fixture
    def base_encoder(self):
        return DenseEncoder(name="TestEncoder", score_threshold=0.5)

    def test_base_encoder_initialization(self, base_encoder):
        assert base_encoder.name == "TestEncoder", "Initialization of name failed"
        assert base_encoder.score_threshold == 0.5

    def test_base_encoder_call_method_not_implemented(self, base_encoder):
        with pytest.raises(NotImplementedError):
            base_encoder(["some", "texts"])
