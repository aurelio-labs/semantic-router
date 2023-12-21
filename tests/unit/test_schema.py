import pytest

from semantic_router.schemas.encoder import (
    CohereEncoder,
    Encoder,
    EncoderType,
    OpenAIEncoder,
)

from semantic_router.schemas.route import (
    Route,
)

from semantic_router.schemas.semantic_space import (
    SemanticSpace,
)


class TestEncoderDataclass:
    def test_encoder_initialization_openai(self, mocker):
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
        encoder = Encoder(type="openai", name="test-engine")
        assert encoder.type == EncoderType.OPENAI
        assert isinstance(encoder.model, OpenAIEncoder)

    def test_encoder_initialization_cohere(self, mocker):
        mocker.patch.dict("os.environ", {"COHERE_API_KEY": "test"})
        encoder = Encoder(type="cohere", name="test-engine")
        assert encoder.type == EncoderType.COHERE
        assert isinstance(encoder.model, CohereEncoder)

    def test_encoder_initialization_unsupported_type(self):
        with pytest.raises(ValueError):
            Encoder(type="unsupported", name="test-engine")

    def test_encoder_initialization_huggingface(self):
        with pytest.raises(NotImplementedError):
            Encoder(type="huggingface", name="test-engine")

    def test_encoder_call_method(self, mocker):
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
        mocker.patch(
            "semantic_router.encoders.openai.OpenAIEncoder.__call__",
            return_value=[0.1, 0.2, 0.3],
        )
        encoder = Encoder(type="openai", name="test-engine")
        result = encoder(["test"])
        assert result == [0.1, 0.2, 0.3]


class TestSemanticSpaceDataclass:
    def test_semanticspace_initialization(self):
        semantic_space = SemanticSpace()
        assert semantic_space.id == ""
        assert semantic_space.routes == []

    def test_semanticspace_add_route(self):
        route = Route(name="test", utterances=["hello", "hi"], description="greeting")
        semantic_space = SemanticSpace()
        semantic_space.add(route)

        assert len(semantic_space.routes) == 1
        assert semantic_space.routes[0].name == "test"
        assert semantic_space.routes[0].utterances == ["hello", "hi"]
        assert semantic_space.routes[0].description == "greeting"
