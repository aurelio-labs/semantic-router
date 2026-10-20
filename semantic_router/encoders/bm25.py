import asyncio
from functools import partial
from typing import Any, List

import numpy as np

from semantic_router.encoders.base import (
    AsymmetricSparseMixin,
    FittableMixin,
    SparseEncoder,
)
from semantic_router.route import Route
from semantic_router.schema import SparseEmbedding
from semantic_router.tokenizers import BaseTokenizer, PretrainedTokenizer
from semantic_router.utils.logger import logger


class BM25Encoder(SparseEncoder, FittableMixin, AsymmetricSparseMixin):
    """BM25Encoder, running a vectorized version of ATIRE BM25 algorithm

    Concept:
    - BM25 uses scoring between queries & corpus to retrieve the most relevant documents ∈ corpus
    - most vector databases (VDB) store embedded documents and score them versus received queries for retrieval
    - we need to break up the BM25 formula into `encode_queries` and `encode_documents`, with the latter to be stored in VDB
    - dot product of `encode_queries(q)` and `encode_documents([D_0, D_1, ...])` is the BM25 score of the documents `[D_0, D_1, ...]` for the given query `q`
    - we train a BM25 encoder's normalization parameters on a sufficiently large corpus to capture target language distribution
    - these trained parameter allow us to balance TF & IDF of query & documents for retrieval (read more on how BM25 fixes issues with TF-IDF)

    The reference formula is section 4.1 of the ATIRE paper:

        RSV_d = Σ_{t ∈ q} log(N / df_t) · (k1 + 1) · tf_td
                           / (k1 · ((1 - b) + b · L_d / L_avg) + tf_td)

    Deviations from the paper
    -------------------------
    This implementation is faithful to the formula above on the document side,
    but deviates from it in three documented ways. Each is deliberate; see the
    inline comments at the point of use for the details.

    1. IDF smoothing. We compute `log((N + 1) / (df_t + 0.5))` rather than
       `log(N / df_t)`, to avoid a division by zero for unseen terms. This
       preserves the strictly-positive IDF that the ATIRE variant exists for.

    2. Query weight normalization. The paper produces an unbounded relevance
       score, which is fine for a search engine that only ranks within a single
       query. A router instead compares scores against a fixed
       `score_threshold`, so scores must be comparable *across* queries. We
       therefore L1-normalize the query weights and scale them by
       `1 / (k1 + 1)`, which bounds the query·document dot product to [0, 1] --
       the same range as the cosine similarity returned by the dense encoders,
       so the two can be blended in `HybridRouter` and thresholded alike.

       This is ranking-neutral within a query (it is a single positive constant
       per query) but it does discard cross-query IDF magnitude: two queries
       with different total IDF mass can produce the same score.

    3. Parameter defaults. We default to `k1=1.5, b=0.75` rather than the
       paper's `k1=0.9, b=0.4`. See the `k1` and `b` parameter docs below.

    ATIRE Paper: https://www.cs.otago.ac.nz/research/student-publications/atire-opensource.pdf
    Pinecone Implementation: https://github.com/pinecone-io/pinecone-text/blob/8399f9ff28c4652766c35165c0db9b0eff309077/pinecone_text/sparse/bm25_encoder.py

    :param k1: normalizer parameter that limits how much a single query term `q_i ∈ q` can affect score for document `D_n`.
        Defaults to 1.5, the standard Robertson value, *not* the 0.9 reported in the ATIRE paper. The paper's value was
        trained on the INEX 2008 Wikipedia collection, whose documents average 878 words (paper table 5), whereas this
        encoder is fitted on route utterances averaging on the order of ten tokens. Transferring the paper's parameters
        to utterance-length text is not obviously more correct, and changing the default would silently shift the scores
        that existing `score_threshold` values were tuned against.
    :type k1: float
    :param b: normalizer parameter that balances the effect of a single document length compared to the average document
        length. Defaults to 0.75 rather than the paper's 0.4, for the same reason as `k1`. Note that `b` is considerably
        more influential on short text: over utterances spanning 0.3x to 2.2x the average length, `b=0.75` spreads the
        document factor by ~2.2x versus ~1.4x for `b=0.4`. Lower it towards the paper's value if length is over-weighted
        for your route set.
    :type b: float
    :param corpus_size: number of documents in the trained corpus
    :type corpus_size: int, optional
    :param _avg_doc_len: float representing the average document length in the trained corpus
    :type _avg_doc_len: float, optional
    :param _documents_containing_word: (1, tokenizer.vocab_size) shaped array, denoting how many documents contain `token ∈ vocab`
    :type _documents_containing_word: class:`numpy.ndarray`, optional

    """

    type: str = "sparse"
    k1: float = 1.5
    b: float = 0.75
    corpus_size: int | None = None
    _tokenizer: BaseTokenizer | None
    _avg_doc_len: np.float64 | float | None
    _documents_containing_word: np.ndarray | None

    def __init__(
        self,
        tokenizer: BaseTokenizer | None = None,
        name: str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        corpus_size: int | None = None,
        avg_doc_len: float | None = None,
        use_default_params: bool = True,
    ) -> None:
        if name is None:
            name = "bm25"
        super().__init__(name=name)

        self.k1 = k1
        self.b = b
        self.corpus_size = corpus_size
        self._avg_doc_len = np.float64(avg_doc_len) if avg_doc_len else None
        if use_default_params and not tokenizer:
            logger.info("Initializing default BM25 model parameters.")
            self._tokenizer = PretrainedTokenizer("google-bert/bert-base-uncased")
        elif tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            raise ValueError(
                "Tokenizer not provided. Provide a tokenizer or set `use_default_params` to True"
            )

    def _fit_validate(self, routes: List[Route]):
        if not isinstance(routes, list) or not isinstance(routes[0], Route):
            raise TypeError("`routes` parameter must be a list of Route objects.")

    def fit(self, routes: List[Route]) -> "BM25Encoder":
        """Trains the encoder weights on the provided routes.

        :param routes: List of routes to train the encoder on.
        :type routes: List[Route]
        """
        if not self._tokenizer:
            raise ValueError(
                "BM25 encoder not initialized. Provide a tokenizer or set `use_default_params` to True"
            )
        self._fit_validate(routes)
        utterances = [utterance for route in routes for utterance in route.utterances]
        utterance_ids = self._tokenizer.tokenize(utterances, pad=True)
        corpus = self._tf(utterance_ids)

        self.corpus_size = len(utterances)

        # Calculate document lengths and average
        doc_lengths = corpus.sum(axis=1)
        self._avg_doc_len = doc_lengths.mean()

        # Calculate document frequencies
        documents_containing_word = np.atleast_2d((corpus > 0).sum(axis=0))

        documents_containing_word[:, 0] *= 0
        self._documents_containing_word = documents_containing_word

        return self

    def _tf(self, docs: np.ndarray) -> np.ndarray:
        """Returns term frequency of query terms in trained corpus

        :param docs: 2D shaped array of each document's token ids
        :type docs: numpy.ndarray
        :return: Matrix where value @ (m, n) represents how many times token id `n` appears in document `m`
        :rtype: numpy.ndarray
        """
        if self._tokenizer is None:
            raise ValueError(
                "Tokenizer not provided. Provide a tokenizer or set `use_default_params` to True"
            )
        vocab_size = self._tokenizer.vocab_size
        # `np.bincount` doesn't return a consistent shape, we need to ensure minlength
        bincount = partial(np.bincount, minlength=vocab_size)
        # Bincount returns element count
        # e.g. [0, 1, 1, 3, 1] => [1, 3, 0, 1]
        tf = np.apply_along_axis(bincount, 1, docs)

        # only change the values of existing non-zero terms, so that the operation doesn't change the sparsity of the matrix, and we don't get a warning that our operation is expensive
        # We use `0` as a padding, so ignore it's term frequency
        tf[:, 0] *= 0  # type: ignore
        return tf

    def _df(self, queries: np.ndarray) -> np.ndarray:
        """Returns the amount of times each token in the query appears in trained corpus

        This is done in a faster, vectorized way, instead of looping through each query

        :param queries: 2D shaped array of each query token ids
        :type queries: numpy.ndarray
        :return: Matrix where value @ (m, n) represents how many times token id `n` in query `m` appears in the trained corpus
        :rtype: numpy.ndarray
        """
        if self._documents_containing_word is None:
            raise ValueError(
                "Encoder not fitted. `BM25Encoder.fit` a corpus, or `BM25Encoder.load` a pretrained encoder."
            )
        if self._tokenizer is None:
            raise ValueError(
                "Tokenizer not provided. Provide a tokenizer or set `use_default_params` to True"
            )
        n = queries.shape[0]
        # Create row indices -> [[0], [1], [2], ...] (n, 1) shaped matrix
        row_indices = np.arange(n)[:, None]

        # Create a (len(queries), vocab_size) mask
        mask = np.zeros(
            (n, self._tokenizer.vocab_size), dtype=bool
        )  # (n_queries, vocab_size)
        # Fill the mask with `True` only for the token ids that appear in each query
        mask[row_indices, queries] = True

        # Repeat the `(1, vocab_size)` shaped _documents_containing_word matrix `len(queries)` times -> (len(queries), vocab_size) shape
        # Multiply the repeated matrix with the mask to result in the document frequency of each token in each query
        query_df = mask * self._documents_containing_word
        return query_df

    def encode_queries(self, queries: list[str]) -> list[SparseEmbedding]:
        """Returns BM25 scores for queries using precomputed corpus scores.

        :param queries: List of queries to encode
        :type queries: list
        :return: BM25 scores for each query against the corpus
        :rtype: list[SparseEmbedding]
        """
        if (
            self.corpus_size is None
            or self._avg_doc_len is None
            or self._documents_containing_word is None
        ):
            raise ValueError(
                "Encoder not fitted. Please `.fit` the model on a provided corpus or load a pretrained encoder"
            )
        if not self._tokenizer:
            raise ValueError(
                "BM25 encoder not initialized. Provide a tokenizer or set `use_default_params` to True"
            )
        if queries == []:
            raise ValueError("No documents provided for encoding")

        # Convert queries to token counts
        queries_ids = self._tokenizer.tokenize(queries, pad=True)
        df = self._df(queries_ids)  # (batch_size, vocab_size)
        N = self.corpus_size

        # DEVIATION FROM THE ATIRE PAPER: the paper specifies `log(N / df_t)`,
        # whereas we compute `log((N + 1) / (df_t + 0.5))`. The `+ 0.5` avoids a
        # division by zero and the `+ 1` keeps the numerator above the
        # denominator, so the result is strictly positive. This preserves the
        # property the ATIRE variant was introduced for (IDF is never negative,
        # see footnote 1 of the paper) and stays monotonically decreasing in
        # `df_t`, but it is a smoothed variant rather than the paper's formula.
        # It weights common terms slightly less harshly than `log(N / df_t)`.
        df = df + np.where(df > 0, 0.5, 0)
        idf = np.divide(N + 1, df, out=np.zeros_like(df), where=df != 0)
        idf = np.log(
            idf, out=np.zeros_like(df), where=df != 0
        )  # (batch_size, vocab_size)

        # The paper sums over the query terms `t ∈ q`, so a term repeated in the
        # query contributes once per occurrence. Weighting IDF by the query term
        # frequency is equivalent, and keeps the operation vectorised.
        qtf = self._tf(queries_ids)  # (batch_size, vocab_size)
        weighted_idf = idf * qtf

        # DEVIATION FROM THE ATIRE PAPER: the paper's RSV is an unbounded sum,
        # which is fine when only ranking within one query. A router compares
        # scores against a fixed `score_threshold`, so they must be comparable
        # across queries too. Two steps make that so:
        #
        #   1. L1-normalize, so the weights sum to 1 regardless of how long the
        #      query is or how rare its terms are.
        #   2. Divide by `k1 + 1`, so the weights sum to `1 / (k1 + 1)`.
        #
        # `encode_documents` returns values in (0, k1 + 1] (it keeps the paper's
        # `(k1 + 1)` numerator), so the dot product of the two is bounded to
        # [0, 1] -- the same range as the cosine similarity produced by the
        # dense encoders. That lets `HybridRouter` blend the two and lets one
        # threshold apply to both. Without the `k1 + 1` divisor the sparse score
        # would reach `k1 + 1` (2.5 at the default `k1`) and would dominate the
        # dense component of the hybrid score.
        #
        # Both steps are a single positive constant per query, so neither
        # changes the ranking of documents for that query.
        denominator = weighted_idf.sum(axis=1)[:, np.newaxis] * (self.k1 + 1.0)
        idf_norm = np.divide(
            weighted_idf,
            denominator,
            out=np.zeros_like(weighted_idf),
            where=weighted_idf != 0,
        )

        return self._array_to_sparse_embeddings(idf_norm)

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int | None = None,
    ) -> list[SparseEmbedding]:
        r"""Returns document term frequency normed by itself & average trained corpus length
        (This is the right-hand side of the BM25 equation, which gets matmul-ed with the query IDF component)

        LaTeX: $\frac{(k_1 + 1) \times f(d_i, D)}{f(d_i, D) + k_1 \times (1 - b + b \times \frac{|D|}{avgdl})}$
        where:
            f(d_i, D) is frequency of term `d_i ∈ D`
            |D| is the document length
            avgdl is average document length in trained corpus

        :param documents: List of queries to encode
        :type documents: list
        :return: Encoded queries (as either sparse or dict)
        :rtype: list[SparseEmbedding]
        """
        if (
            self.corpus_size is None
            or self._avg_doc_len is None
            or self._documents_containing_word is None
        ):
            raise ValueError(
                "Encoder not fitted. Please `.fit` the model on a provided corpus or load a pretrained encoder"
            )
        if not self._tokenizer:
            raise ValueError(
                "BM25 encoder not initialized. Provide a tokenizer or set `use_default_params` to True"
            )
        if documents == []:
            raise ValueError("No documents provided for encoding")
        batch_size = batch_size or len(documents)
        queries_ids = self._tokenizer.tokenize(documents, pad=True)
        tf = self._tf(queries_ids)  # (batch_size, vocab_size)
        tf_sum = tf.sum(axis=1)  # (batch_size, 1)
        tf_normed = ((self.k1 + 1.0) * tf) / (
            self.k1
            * ((1.0 - self.b) + self.b * (tf_sum[:, np.newaxis] / self._avg_doc_len))
            + tf
        )  # (batch_size, vocab_size)

        return self._array_to_sparse_embeddings(tf_normed)

    def model(self, docs: List[str]) -> list[SparseEmbedding]:
        """Encode documents using BM25, with different encoding for queries vs documents to be indexed.

        :param docs: List of documents to encode
        :param is_query: If True, use query encoding, else use document encoding
        :return: List of sparse embeddings
        """
        if not self._tokenizer:
            raise ValueError(
                "Encoder not fitted. `BM25.index` a corpus, or `BM25.load` a pretrained encoder."
            )
        if (
            self.corpus_size is None
            or self._avg_doc_len is None
            or self._documents_containing_word is None
        ):
            raise ValueError(
                "Encoder not fitted. Please `.fit` the model on a provided corpus or load a pretrained encoder"
            )

        return self.encode_queries(docs)

    async def aencode_queries(self, docs: List[str]) -> List[SparseEmbedding]:
        # While this is a CPU-bound operation, and doesn't benefit from asyncio
        # we provide this method to abide by the `SparseEncoder` superclass
        return await asyncio.to_thread(lambda: self.encode_queries(docs))

    async def aencode_documents(self, docs: List[str]) -> List[SparseEmbedding]:
        # While this is a CPU-bound operation, and doesn't benefit from asyncio
        # we provide this method to abide by the `SparseEncoder` superclass
        return await asyncio.to_thread(lambda: self.encode_documents(docs))

    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        return self.encode_queries(docs)

    async def acall(self, docs: List[Any]) -> List[SparseEmbedding]:
        return await asyncio.to_thread(lambda: self.__call__(docs))
