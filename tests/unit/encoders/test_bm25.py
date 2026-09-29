import os

import numpy as np
import pytest

from semantic_router.encoders import BM25Encoder
from semantic_router.route import Route
from semantic_router.tokenizers import BaseTokenizer

UTTERANCES = [
    "Hello we need this text to be a little longer for our sparse encoders",
    "In this case they need to learn from recurring tokens, ie words.",
    "We give ourselves several examples from our encoders to learn from.",
    "But given this is only an example we don't need too many",
    "Just enough to test that our sparse encoders work as expected",
]


@pytest.fixture
def bm25_encoder():
    sparse_encoder = BM25Encoder(use_default_params=True)
    sparse_encoder.fit(
        [
            Route(
                name="test_route",
                utterances=[
                    "The quick brown fox",
                    "jumps over the lazy dog",
                    "Hello, world!",
                ],
            )
        ]
    )
    return sparse_encoder


@pytest.fixture
def routes():
    return [
        Route(name="Route 1", utterances=[UTTERANCES[0], UTTERANCES[1]]),
        Route(name="Route 2", utterances=[UTTERANCES[2], UTTERANCES[3], UTTERANCES[4]]),
    ]


@pytest.mark.skipif(
    os.environ.get("RUN_HF_TESTS") is None,
    reason="Set RUN_HF_TESTS=1 to run. This test downloads models from Hugging Face which can time out in CI.",
)
class TestBM25Encoder:
    def test_initialization(self, bm25_encoder):
        assert bm25_encoder._tokenizer is not None

    def test_fit(self, bm25_encoder, routes):
        bm25_encoder.fit(routes)
        assert bm25_encoder._tokenizer is not None

    def test_fit_with_strings(self, bm25_encoder):
        route_strings = ["test a", "test b", "test c"]
        with pytest.raises(TypeError):
            bm25_encoder.fit(route_strings)

    def test_call_method(self, bm25_encoder):
        result = bm25_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sparse_emb.embedding, np.ndarray) for sparse_emb in result
        ), "Each item in result should be an array"

    def test_call_method_no_docs_bm25_encoder(self, bm25_encoder):
        with pytest.raises(ValueError):
            bm25_encoder([])

    def test_call_method_no_word(self, bm25_encoder):
        result = bm25_encoder(["doc with fake word gta5jabcxyz"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sparse_emb.embedding, np.ndarray) for sparse_emb in result
        ), "Each item in result should be an array"

    def test_call_method_with_uninitialized_model_or_mapping(self, bm25_encoder):
        bm25_encoder._tokenizer = None
        with pytest.raises(ValueError):
            bm25_encoder(["test"])

    def test_fit_with_uninitialized_model(self, bm25_encoder, routes):
        bm25_encoder._tokenizer = None
        with pytest.raises(ValueError):
            bm25_encoder.fit(routes)

    def test_encode_queries(self, bm25_encoder):
        queries = ["quick brown", "lazy dog", "hello world"]
        results = bm25_encoder.encode_queries(queries)

        assert len(results) == len(queries)
        assert all([isinstance(result.embedding, np.ndarray) for result in results])

    def test_encode_queries_empty_list(self, bm25_encoder):
        with pytest.raises(ValueError, match="No documents provided for encoding"):
            bm25_encoder.encode_queries([])

    def test_encode_queries_unfitted(self):
        encoder = BM25Encoder(use_default_params=True)
        with pytest.raises(ValueError, match="Encoder not fitted"):
            encoder.encode_queries(["test query"])

    def test_encode_documents(self, bm25_encoder):
        documents = ["quick brown", "lazy dog", "hello world"]
        results = bm25_encoder.encode_documents(documents)

        assert len(results) == len(documents)
        assert all([isinstance(result.embedding, np.ndarray) for result in results])

    def test_encode_documents_empty_list(self, bm25_encoder):
        with pytest.raises(ValueError, match="No documents provided for encoding"):
            bm25_encoder.encode_documents([])

    def test_encode_documents_unfitted(self):
        encoder = BM25Encoder(use_default_params=True)
        with pytest.raises(ValueError, match="Encoder not fitted"):
            encoder.encode_documents(["test document"])

    def test_encode_documents_batch_size(self, bm25_encoder):
        documents = ["quick brown", "lazy dog", "hello world", "test document"]
        batch_size = 2
        results = bm25_encoder.encode_documents(documents, batch_size=batch_size)

        assert len(results) == len(documents)
        assert all(isinstance(result.embedding, np.ndarray) for result in results)


class WordTokenizer(BaseTokenizer):
    """Deterministic word-level tokenizer, so the formula tests below need no
    model download. Token id 0 is reserved for padding, matching the convention
    :class:`BM25Encoder` relies on.
    """

    def __init__(self, vocab: list[str]) -> None:
        super().__init__()
        self._vocab = {word: idx + 1 for idx, word in enumerate(vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + 1

    def tokenize(self, texts, pad: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        batch = [
            [self._vocab[word] for word in text.split() if word in self._vocab]
            for text in texts
        ]
        width = max(len(ids) for ids in batch)
        return np.array([ids + [0] * (width - len(ids)) for ids in batch])


# "eta" is in the tokenizer vocabulary but never appears in CORPUS, so it
# exercises the df=0 path.
VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]

# Document lengths 2, 3 and 4, so avgdl is exactly 3.0.
CORPUS = [
    "alpha beta",
    "beta gamma delta",
    "gamma delta epsilon zeta",
]
AVG_DOC_LEN = 3.0


def atire_tf_component(tf: float, doc_len: int, k1: float, b: float) -> float:
    """The document-side factor of the ATIRE BM25 formula (paper section 4.1).

    Written out longhand as a reference, independent of the vectorised
    implementation under test.
    """
    return ((k1 + 1.0) * tf) / (k1 * ((1.0 - b) + b * (doc_len / AVG_DOC_LEN)) + tf)


@pytest.fixture
def word_encoder():
    encoder = BM25Encoder(tokenizer=WordTokenizer(VOCAB), use_default_params=False)
    encoder.fit([Route(name="corpus", utterances=CORPUS)])
    return encoder


class TestBM25ATIREFormula:
    """Guards the ATIRE BM25 formula itself, rather than just output shapes."""

    def test_fit_computes_avg_doc_len(self, word_encoder):
        assert word_encoder.corpus_size == len(CORPUS)
        assert word_encoder._avg_doc_len == pytest.approx(AVG_DOC_LEN)

    @pytest.mark.parametrize(
        "document,doc_len,term,tf",
        [
            ("alpha", 1, "alpha", 1),
            ("alpha beta gamma", 3, "beta", 1),
            ("alpha alpha beta", 3, "alpha", 2),
            # ~3x the average length: the case where an inverted length
            # normalisation drives the denominator through zero.
            ("alpha beta gamma delta epsilon zeta alpha beta gamma", 9, "alpha", 2),
        ],
    )
    def test_encode_documents_matches_atire(
        self, word_encoder, document, doc_len, term, tf
    ):
        token_id = VOCAB.index(term) + 1
        encoded = word_encoder.encode_documents([document])[0].to_dict()

        expected = atire_tf_component(tf, doc_len, word_encoder.k1, word_encoder.b)
        assert encoded[token_id] == pytest.approx(expected)

    def test_encode_documents_always_positive(self, word_encoder):
        """The denominator must stay positive at any document length."""
        documents = [" ".join(["alpha"] + VOCAB * n) for n in range(1, 20)]
        for embedding in word_encoder.encode_documents(documents):
            values = embedding.embedding[:, 1]
            assert np.all(values > 0.0)

    def test_repeated_query_terms_are_weighted_by_frequency(self, word_encoder):
        """The paper sums over query term occurrences, so a term repeated in the
        query must carry proportionally more weight than a term appearing once.
        """
        # "alpha" and "epsilon" both appear in exactly one CORPUS document, so
        # their IDF is equal and any weight difference comes from query
        # term frequency alone.
        alpha_id = VOCAB.index("alpha") + 1
        epsilon_id = VOCAB.index("epsilon") + 1

        once = word_encoder.encode_queries(["alpha epsilon"])[0].to_dict()
        twice = word_encoder.encode_queries(["alpha alpha epsilon"])[0].to_dict()

        assert once[alpha_id] == pytest.approx(once[epsilon_id])

        # Repeating "alpha" must double its weight relative to "epsilon".
        assert twice[alpha_id] / twice[epsilon_id] == pytest.approx(2.0)
        assert twice[alpha_id] > once[alpha_id]

    def test_query_weights_sum_to_inverse_k1_plus_one(self, word_encoder):
        """Deviation from the paper: query weights are L1 normalised and then
        scaled by 1/(k1+1), so that scores land in [0, 1].
        """
        expected = 1.0 / (word_encoder.k1 + 1.0)
        for query in ["alpha", "alpha beta", "alpha alpha beta gamma"]:
            embedding = word_encoder.encode_queries([query])[0]
            assert embedding.embedding[:, 1].sum() == pytest.approx(expected)

    def test_scores_are_bounded_to_unit_interval(self, word_encoder):
        """The reason for the 1/(k1+1) scaling: scores share the [0, 1] range of
        the cosine similarities returned by the dense encoders.
        """
        queries = ["alpha", "alpha beta", "alpha alpha zeta", "gamma delta epsilon"]
        documents = CORPUS + [
            "alpha",
            "alpha alpha alpha",
            " ".join(VOCAB * 4),
            "gamma",
        ]
        encoded_docs = [d.to_dict() for d in word_encoder.encode_documents(documents)]

        for query in queries:
            weights = word_encoder.encode_queries([query])[0].to_dict()
            for doc, doc_weights in zip(documents, encoded_docs):
                score = sum(v * doc_weights.get(k, 0.0) for k, v in weights.items())
                assert 0.0 <= score <= 1.0, f"{query!r} vs {doc!r} scored {score}"

    def test_query_normalisation_is_ranking_neutral(self, word_encoder):
        """Normalisation is a positive constant per query, so it must not
        reorder documents.
        """
        query = "alpha alpha gamma zeta"
        documents = CORPUS + ["alpha gamma", "zeta zeta alpha", " ".join(VOCAB)]
        encoded_docs = [d.to_dict() for d in word_encoder.encode_documents(documents)]

        normalised = word_encoder.encode_queries([query])[0].to_dict()
        # Undo both normalisation steps to recover the paper's raw weights.
        total = sum(normalised.values())
        raw = {k: v / total for k, v in normalised.items()}

        def rank(weights):
            scores = [
                sum(v * d.get(k, 0.0) for k, v in weights.items()) for d in encoded_docs
            ]
            return list(np.argsort(-np.array(scores)))

        assert rank(normalised) == rank(raw)

    def test_query_terms_outside_corpus_are_dropped(self, word_encoder):
        """Terms absent from the corpus have df=0, so no IDF can be computed and
        they must not contribute weight.
        """
        eta_id = VOCAB.index("eta") + 1
        weights = word_encoder.encode_queries(["alpha eta"])[0].to_dict()

        assert weights.get(eta_id, 0.0) == pytest.approx(0.0)
        # The remaining weight still carries the full normalised mass.
        assert sum(weights.values()) == pytest.approx(1.0 / (word_encoder.k1 + 1.0))

    def test_longer_documents_score_lower(self, word_encoder):
        """A single occurrence of a term is worth less in a longer document."""
        documents = [
            "alpha",
            "alpha beta",
            "alpha beta gamma",
            "alpha beta gamma delta",
            "alpha beta gamma delta epsilon",
            "alpha beta gamma delta epsilon zeta",
        ]
        alpha_id = VOCAB.index("alpha") + 1
        scores = [
            embedding.to_dict()[alpha_id]
            for embedding in word_encoder.encode_documents(documents)
        ]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > scores[-1]
