import string
from collections import Counter
from typing import Dict, List

import numpy as np
from numpy import ndarray
from numpy.linalg import norm

from semantic_router.encoders import BaseEncoder
from semantic_router.route import Route


class TfidfEncoder(BaseEncoder):
    idf: ndarray = np.array([])
    word_index: Dict = {}

    def __init__(self, name: str = "tfidf", score_threshold: float = 0.82):
        # TODO default score_threshold not thoroughly tested, should optimize
        super().__init__(name=name, score_threshold=score_threshold)
        self.word_index = {}
        self.idf = np.array([])

    def __call__(self, docs: List[str]) -> List[List[float]]:
        if len(self.word_index) == 0 or self.idf.size == 0:
            raise ValueError("Vectorizer is not initialized.")
        if len(docs) == 0:
            raise ValueError("No documents to encode.")

        docs = [self._preprocess(doc) for doc in docs]
        tf = self._compute_tf(docs)
        tfidf = tf * self.idf
        return tfidf.tolist()

    def fit(self, routes: List[Route]):
        docs = []
        for route in routes:
            for doc in route.utterances:
                docs.append(self._preprocess(doc))  # type: ignore
        self.word_index = self._build_word_index(docs)
        self.idf = self._compute_idf(docs)

    def _build_word_index(self, docs: List[str]) -> Dict:
        words = set()
        for doc in docs:
            for word in doc.split():
                words.add(word)
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
        # L2 normalization
        tf = tf / norm(tf, axis=1, keepdims=True)
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
        idf = np.log(len(docs) / (idf + 1))
        return idf

    def _preprocess(self, doc: str) -> str:
        lowercased_doc = doc.lower()
        no_punctuation_doc = lowercased_doc.translate(
            str.maketrans("", "", string.punctuation)
        )
        return no_punctuation_doc
