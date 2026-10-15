import asyncio
import string
from collections import Counter
from typing import Dict, List

import numpy as np

from semantic_router.encoders import SparseEncoder
from semantic_router.encoders.base import FittableMixin
from semantic_router.route import Route
from semantic_router.schema import SparseEmbedding


class TfidfEncoder(SparseEncoder, FittableMixin):
    idf: np.ndarray = np.array([])
    # TODO: add option to use default params like with BM25Encoder
    word_index: Dict = {}

    def __init__(self, name: str | None = None):
        if name is None:
            name = "tfidf"
        super().__init__(name=name)
        self.word_index = {}
        self.idf = np.array([])

    def __call__(self, docs: List[str]) -> list[SparseEmbedding]:
        if len(self.word_index) == 0 or self.idf.size == 0:
            raise ValueError("Vectorizer is not initialized.")
        if len(docs) == 0:
            raise ValueError("No documents to encode.")

        docs = [self._preprocess(doc) for doc in docs]
        tf = self._compute_tf(docs)
        tfidf = tf * self.idf
        # L2-normalize after weighting, so a query·document score is a cosine in
        # [0, 1], the same range as the dense scores it is blended with
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        tfidf = np.divide(tfidf, norms, out=np.zeros_like(tfidf), where=norms != 0)
        return self._array_to_sparse_embeddings(tfidf)

    async def acall(self, docs: List[str]) -> List[SparseEmbedding]:
        return await asyncio.to_thread(lambda: self.__call__(docs))

    def fit(self, routes: List[Route]):
        """Trains the encoder weights on the provided routes.

        :param routes: List of routes to train the encoder on.
        :type routes: List[Route]
        """
        self._fit_validate(routes=routes)
        docs = []
        for route in routes:
            for doc in route.utterances:
                docs.append(self._preprocess(doc))  # type: ignore
        self.word_index = self._build_word_index(docs)
        if len(self.word_index) == 0:
            raise ValueError(f"Too little data to fit {self.__class__.__name__}.")
        self.idf = self._compute_idf(docs)

    def _fit_validate(self, routes: List[Route]):
        if not isinstance(routes, list) or not isinstance(routes[0], Route):
            raise TypeError("`routes` parameter must be a list of Route objects.")

    def _build_word_index(self, docs: List[str]) -> Dict:
        # sorted, so a word gets the same id in every process (set order depends
        # on PYTHONHASHSEED, and ids stored in a remote index must stay valid)
        words = sorted({word for doc in docs for word in doc.split()})
        word_index = {word: i for i, word in enumerate(words)}
        return word_index

    def _compute_tf(self, docs: List[str]) -> np.ndarray:
        if len(self.word_index) == 0:
            raise ValueError("Word index is not initialized.")
        tf = np.zeros((len(docs), len(self.word_index)))
        for i, doc in enumerate(docs):
            word_counts = Counter(doc.split())
            for word, count in word_counts.items():
                if word in self.word_index:
                    tf[i, self.word_index[word]] = count
        # L2 normalization; a doc with no words in the index stays all zero
        norms = np.linalg.norm(tf, axis=1, keepdims=True)
        tf = np.divide(tf, norms, out=np.zeros_like(tf), where=norms != 0)
        return tf

    def _compute_idf(self, docs: List[str]) -> np.ndarray:
        if len(self.word_index) == 0:
            raise ValueError("Word index is not initialized.")
        idf = np.zeros(len(self.word_index))
        for doc in docs:
            words = set(doc.split())
            for word in words:
                if word in self.word_index:
                    idf[self.word_index[word]] += 1
        # smoothed as in scikit-learn: always >= 1, so a word in every document
        # still counts, and never with a negative weight
        idf = np.log((1 + len(docs)) / (1 + idf)) + 1
        return idf

    def _preprocess(self, doc: str) -> str:
        lowercased_doc = doc.lower()
        no_punctuation_doc = lowercased_doc.translate(
            str.maketrans("", "", string.punctuation)
        )
        return no_punctuation_doc
