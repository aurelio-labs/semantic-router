import pytest

from semantic_router.encoders import (
    AzureOpenAIEncoder,
    DenseEncoder,
    BM25Encoder,
    CohereEncoder,
    OpenAIEncoder,
    TfidfEncoder,
)
from semantic_router.routers import HybridRouter
from semantic_router.route import Route


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
def base_encoder(mocker):
    mock_base_encoder = DenseEncoder(name="test-encoder", score_threshold=0.5)
    mocker.patch.object(DenseEncoder, "__call__", return_value=[[0.1, 0.2, 0.3]])
    return mock_base_encoder


@pytest.fixture
def cohere_encoder(mocker):
    mocker.patch.object(CohereEncoder, "__call__", side_effect=mock_encoder_call)
    return CohereEncoder(name="test-cohere-encoder", cohere_api_key="test_api_key")


@pytest.fixture
def openai_encoder(mocker):
    mocker.patch.object(OpenAIEncoder, "__call__", side_effect=mock_encoder_call)
    return OpenAIEncoder(name="text-embedding-3-small", openai_api_key="test_api_key")


@pytest.fixture
def azure_encoder(mocker):
    mocker.patch.object(AzureOpenAIEncoder, "__call__", side_effect=mock_encoder_call)
    return AzureOpenAIEncoder(
        deployment_name="test-deployment",
        azure_endpoint="test_endpoint",
        api_key="test_api_key",
        api_version="test_version",
        model="test_model",
    )

@pytest.fixture
def bm25_encoder():
    #mocker.patch.object(BM25Encoder, "__call__", side_effect=mock_encoder_call)
    return BM25Encoder(name="test-bm25-encoder")


@pytest.fixture
def tfidf_encoder():
    #mocker.patch.object(TfidfEncoder, "__call__", side_effect=mock_encoder_call)
    return TfidfEncoder(name="test-tfidf-encoder")


@pytest.fixture
def routes():
    return [
        Route(name="Route 1", utterances=[
            "Hello we need this text to be a little longer for our sparse encoders",
            "In this case they need to learn from recurring tokens, ie words."
        ]),
        Route(name="Route 2", utterances=[
            "We give ourselves several examples from our encoders to learn from.",
            "But given this is only an example we don't need too many",
            "Just enough to test that our sparse encoders work as expected"
        ]),
    ]


sparse_encoder = TfidfEncoder()
sparse_encoder.fit(
    [
        Route(
            name="Route 1",
            utterances=[
                "The quick brown fox jumps over the lazy dog",
                "some other useful text containing words like fox and dog"
            ],
        ),
        Route(name="Route 2", utterances=["Hello, world!"]),
    ]
)


class TestHybridRouter:
    def test_initialization(self, openai_encoder, routes):
        route_layer = HybridRouter(
            encoder=openai_encoder,
            sparse_encoder=sparse_encoder,
            routes=routes,
            top_k=10,
            alpha=0.8,
        )
        assert route_layer.index is not None and route_layer.routes is not None
        assert openai_encoder.score_threshold == 0.3
        assert route_layer.score_threshold == 0.3
        assert route_layer.top_k == 10
        assert route_layer.alpha == 0.8
        assert route_layer.index.route_names is None
        assert len(route_layer.routes) == 2

    def test_initialization_different_encoders(self, cohere_encoder, openai_encoder):
        route_layer_cohere = HybridRouter(
            encoder=cohere_encoder, sparse_encoder=sparse_encoder
        )
        assert route_layer_cohere.score_threshold == 0.3

        route_layer_openai = HybridRouter(
            encoder=openai_encoder, sparse_encoder=sparse_encoder
        )
        assert route_layer_openai.score_threshold == 0.3

    def test_add_route(self, openai_encoder, routes):
        route_layer = HybridRouter(
            encoder=openai_encoder, sparse_encoder=sparse_encoder
        )
        route_layer.add(routes=routes[0])
        assert route_layer.index is not None, "route_layer.index is None"
        assert route_layer.routes is not None, "route_layer.routes is None"
        assert len(route_layer.routes) == 1, "route_layer.routes is not 1"

    def test_add_multiple_routes(self, openai_encoder, routes):
        route_layer = HybridRouter(
            encoder=openai_encoder, sparse_encoder=sparse_encoder
        )
        route_layer.add(routes=routes)
        assert route_layer.index is not None, "route_layer.index is None"
        assert route_layer.routes is not None, "route_layer.routes is None"
        assert len(route_layer.routes) == 2, "route_layer.routes is not 2"

    def test_query_and_classification(self, openai_encoder, routes):
        route_layer = HybridRouter(
            encoder=openai_encoder, sparse_encoder=sparse_encoder, routes=routes
        )
        query_result = route_layer("Hello")
        assert query_result in ["Route 1", "Route 2"]

    def test_query_with_no_index(self, openai_encoder):
        route_layer = HybridRouter(
            encoder=openai_encoder, sparse_encoder=sparse_encoder
        )
        assert isinstance(
            route_layer.sparse_encoder, BM25Encoder
        ) or isinstance(
            route_layer.sparse_encoder, TfidfEncoder
        ), (
            f"route_layer.sparse_encoder is {route_layer.sparse_encoder.__class__.__name__} "
            "not BM25Encoder or TfidfEncoder"
        )
        assert route_layer("Anything") is None

    def test_semantic_classify(self, openai_encoder, routes):
        route_layer = HybridRouter(
            encoder=openai_encoder, sparse_encoder=sparse_encoder, routes=routes
        )
        classification, score = route_layer._semantic_classify(
            [
                {"route": "Route 1", "score": 0.9},
                {"route": "Route 2", "score": 0.1},
            ]
        )
        assert classification == "Route 1"
        assert score == [0.9]

    def test_semantic_classify_multiple_routes(self, openai_encoder, routes):
        route_layer = HybridRouter(
            encoder=openai_encoder, sparse_encoder=sparse_encoder, routes=routes
        )
        classification, score = route_layer._semantic_classify(
            [
                {"route": "Route 1", "score": 0.9},
                {"route": "Route 2", "score": 0.1},
                {"route": "Route 1", "score": 0.8},
            ]
        )
        assert classification == "Route 1"
        assert score == [0.9, 0.8]

    def test_pass_threshold(self, openai_encoder):
        route_layer = HybridRouter(
            encoder=openai_encoder, sparse_encoder=sparse_encoder
        )
        assert not route_layer._pass_threshold([], 0.5)
        assert route_layer._pass_threshold([0.6, 0.7], 0.5)

    def test_failover_score_threshold(self, base_encoder):
        route_layer = HybridRouter(encoder=base_encoder, sparse_encoder=sparse_encoder)
        assert base_encoder.score_threshold == 0.50
        assert route_layer.score_threshold == 0.50

    def test_add_route_tfidf(self, cohere_encoder, tfidf_encoder, routes):
        hybrid_route_layer = HybridRouter(
            encoder=cohere_encoder,
            sparse_encoder=tfidf_encoder,
            routes=routes[:-1],
        )
        hybrid_route_layer.add(routes=routes[-1])
        all_utterances = [
            utterance for route in routes for utterance in route.utterances
        ]
        assert hybrid_route_layer.index.sparse_index is not None, "sparse_index is None"
        assert len(hybrid_route_layer.index.sparse_index) == len(all_utterances), "sparse_index length mismatch"

    def test_setting_aggregation_methods(self, openai_encoder, routes):
        for agg in ["sum", "mean", "max"]:
            route_layer = HybridRouter(
                encoder=openai_encoder,
                sparse_encoder=sparse_encoder,
                routes=routes,
                aggregation=agg,
            )
            assert route_layer.aggregation == agg

    def test_semantic_classify_multiple_routes_with_different_aggregation(
        self, openai_encoder, routes
    ):
        route_scores = [
            {"route": "Route 1", "score": 0.5},
            {"route": "Route 1", "score": 0.5},
            {"route": "Route 1", "score": 0.5},
            {"route": "Route 1", "score": 0.5},
            {"route": "Route 2", "score": 0.4},
            {"route": "Route 2", "score": 0.6},
            {"route": "Route 2", "score": 0.8},
            {"route": "Route 3", "score": 0.1},
            {"route": "Route 3", "score": 1.0},
        ]
        for agg in ["sum", "mean", "max"]:
            route_layer = HybridRouter(
                encoder=openai_encoder,
                sparse_encoder=sparse_encoder,
                routes=routes,
                aggregation=agg,
            )
            classification, score = route_layer._semantic_classify(route_scores)

            if agg == "sum":
                assert classification == "Route 1"
                assert score == [0.5, 0.5, 0.5, 0.5]
            elif agg == "mean":
                assert classification == "Route 2"
                assert score == [0.4, 0.6, 0.8]
            elif agg == "max":
                assert classification == "Route 3"
                assert score == [0.1, 1.0]


# Add more tests for edge cases and error handling as needed.
