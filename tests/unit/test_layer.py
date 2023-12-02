import pytest

from semantic_router.encoders import BaseEncoder, CohereEncoder, OpenAIEncoder
from semantic_router.layer import (
    DecisionLayer,
    HybridDecisionLayer,
)  # Replace with the actual module name
from semantic_router.schema import Decision


def mock_encoder_call(utterances):
    # Define a mapping of utterances to return values
    mock_responses = {
        "Hello": [0.1, 0.2, 0.3],
        "Hi": [0.4, 0.5, 0.6],
        "Goodbye": [0.7, 0.8, 0.9],
        "Bye": [1.0, 1.1, 1.2],
        "Au revoir": [1.3, 1.4, 1.5],
    }
    return [mock_responses.get(u, [0, 0, 0]) for u in utterances]


@pytest.fixture
def base_encoder():
    return BaseEncoder(name="test-encoder")


@pytest.fixture
def cohere_encoder(mocker):
    mocker.patch.object(CohereEncoder, "__call__", side_effect=mock_encoder_call)
    return CohereEncoder(name="test-cohere-encoder", cohere_api_key="test_api_key")


@pytest.fixture
def openai_encoder(mocker):
    mocker.patch.object(OpenAIEncoder, "__call__", side_effect=mock_encoder_call)
    return OpenAIEncoder(name="test-openai-encoder", openai_api_key="test_api_key")


@pytest.fixture
def decisions():
    return [
        Decision(name="Decision 1", utterances=["Hello", "Hi"]),
        Decision(name="Decision 2", utterances=["Goodbye", "Bye", "Au revoir"]),
    ]


class TestDecisionLayer:
    def test_initialization(self, openai_encoder, decisions):
        decision_layer = DecisionLayer(encoder=openai_encoder, decisions=decisions)
        assert decision_layer.score_threshold == 0.82
        assert len(decision_layer.index) == 5
        assert len(set(decision_layer.categories)) == 2

    def test_initialization_different_encoders(self, cohere_encoder, openai_encoder):
        decision_layer_cohere = DecisionLayer(encoder=cohere_encoder)
        assert decision_layer_cohere.score_threshold == 0.3

        decision_layer_openai = DecisionLayer(encoder=openai_encoder)
        assert decision_layer_openai.score_threshold == 0.82

    def test_add_decision(self, openai_encoder):
        decision_layer = DecisionLayer(encoder=openai_encoder)
        decision = Decision(name="Decision 3", utterances=["Yes", "No"])
        decision_layer.add(decision)
        assert len(decision_layer.index) == 2
        assert len(set(decision_layer.categories)) == 1

    def test_add_multiple_decisions(self, openai_encoder, decisions):
        decision_layer = DecisionLayer(encoder=openai_encoder)
        for decision in decisions:
            decision_layer.add(decision)
        assert len(decision_layer.index) == 5
        assert len(set(decision_layer.categories)) == 2

    def test_query_and_classification(self, openai_encoder, decisions):
        decision_layer = DecisionLayer(encoder=openai_encoder, decisions=decisions)
        query_result = decision_layer("Hello")
        assert query_result in ["Decision 1", "Decision 2"]

    def test_query_with_no_index(self, openai_encoder):
        decision_layer = DecisionLayer(encoder=openai_encoder)
        assert decision_layer("Anything") is None

    def test_semantic_classify(self, openai_encoder, decisions):
        decision_layer = DecisionLayer(encoder=openai_encoder, decisions=decisions)
        classification, score = decision_layer._semantic_classify(
            [
                {"decision": "Decision 1", "score": 0.9},
                {"decision": "Decision 2", "score": 0.1},
            ]
        )
        assert classification == "Decision 1"
        assert score == [0.9]

    def test_semantic_classify_multiple_decisions(self, openai_encoder, decisions):
        decision_layer = DecisionLayer(encoder=openai_encoder, decisions=decisions)
        classification, score = decision_layer._semantic_classify(
            [
                {"decision": "Decision 1", "score": 0.9},
                {"decision": "Decision 2", "score": 0.1},
                {"decision": "Decision 1", "score": 0.8},
            ]
        )
        assert classification == "Decision 1"
        assert score == [0.9, 0.8]

    def test_pass_threshold(self, openai_encoder):
        decision_layer = DecisionLayer(encoder=openai_encoder)
        assert not decision_layer._pass_threshold([], 0.5)
        assert decision_layer._pass_threshold([0.6, 0.7], 0.5)

    def test_failover_score_threshold(self, base_encoder):
        decision_layer = DecisionLayer(encoder=base_encoder)
        assert decision_layer.score_threshold == 0.82


class TestHybridDecisionLayer:
    def test_initialization(self, openai_encoder, decisions):
        decision_layer = HybridDecisionLayer(
            encoder=openai_encoder, decisions=decisions
        )
        assert decision_layer.score_threshold == 0.82
        assert len(decision_layer.index) == 5
        assert len(set(decision_layer.categories)) == 2

    def test_initialization_different_encoders(self, cohere_encoder, openai_encoder):
        decision_layer_cohere = HybridDecisionLayer(encoder=cohere_encoder)
        assert decision_layer_cohere.score_threshold == 0.3

        decision_layer_openai = HybridDecisionLayer(encoder=openai_encoder)
        assert decision_layer_openai.score_threshold == 0.82

    def test_add_decision(self, openai_encoder):
        decision_layer = HybridDecisionLayer(encoder=openai_encoder)
        decision = Decision(name="Decision 3", utterances=["Yes", "No"])
        decision_layer.add(decision)
        assert len(decision_layer.index) == 2
        assert len(set(decision_layer.categories)) == 1

    def test_add_multiple_decisions(self, openai_encoder, decisions):
        decision_layer = HybridDecisionLayer(encoder=openai_encoder)
        for decision in decisions:
            decision_layer.add(decision)
        assert len(decision_layer.index) == 5
        assert len(set(decision_layer.categories)) == 2

    def test_query_and_classification(self, openai_encoder, decisions):
        decision_layer = HybridDecisionLayer(
            encoder=openai_encoder, decisions=decisions
        )
        query_result = decision_layer("Hello")
        assert query_result in ["Decision 1", "Decision 2"]

    def test_query_with_no_index(self, openai_encoder):
        decision_layer = HybridDecisionLayer(encoder=openai_encoder)
        assert decision_layer("Anything") is None

    def test_semantic_classify(self, openai_encoder, decisions):
        decision_layer = HybridDecisionLayer(
            encoder=openai_encoder, decisions=decisions
        )
        classification, score = decision_layer._semantic_classify(
            [
                {"decision": "Decision 1", "score": 0.9},
                {"decision": "Decision 2", "score": 0.1},
            ]
        )
        assert classification == "Decision 1"
        assert score == [0.9]

    def test_semantic_classify_multiple_decisions(self, openai_encoder, decisions):
        decision_layer = HybridDecisionLayer(
            encoder=openai_encoder, decisions=decisions
        )
        classification, score = decision_layer._semantic_classify(
            [
                {"decision": "Decision 1", "score": 0.9},
                {"decision": "Decision 2", "score": 0.1},
                {"decision": "Decision 1", "score": 0.8},
            ]
        )
        assert classification == "Decision 1"
        assert score == [0.9, 0.8]

    def test_pass_threshold(self, openai_encoder):
        decision_layer = HybridDecisionLayer(encoder=openai_encoder)
        assert not decision_layer._pass_threshold([], 0.5)
        assert decision_layer._pass_threshold([0.6, 0.7], 0.5)

    def test_failover_score_threshold(self, base_encoder):
        decision_layer = HybridDecisionLayer(encoder=base_encoder)
        assert decision_layer.score_threshold == 0.82


# Add more tests for edge cases and error handling as needed.
