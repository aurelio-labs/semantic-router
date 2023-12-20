import numpy as np
from collections import Counter
from semantic_router.encoders import BaseEncoder
from semantic_router.schema import Route
from numpy.linalg import norm


class TfidfEncoder(BaseEncoder):
    idf: dict | None = None
    word_index: dict | None = None

    def __init__(self, name: str = "tfidf"):
        super().__init__(name=name)
        self.word_index = None
        self.idf = None

    def __call__(self, docs: list[str]) -> list[list[float]]:
        if self.word_index is None or self.idf is None:
            raise ValueError("Vectorizer is not initialized.")
        if len(docs) == 0:
            raise ValueError("No documents to encode.")

        tf = self._compute_tf(docs)
        tfidf = tf * self.idf
        return tfidf.tolist()

    def fit(self, routes: list[Route]):
        docs = []
        for route in routes:
            for utterance in route.utterances:
                docs.append(utterance)
        self.word_index = self._build_word_index(docs)
        self.idf = self._compute_idf(docs)

    def _build_word_index(self, docs: list[str]) -> dict:
        words = set()
        for doc in docs:
            for word in doc.split():
                words.add(word)
        word_index = {word: i for i, word in enumerate(words)}
        return word_index

    def _compute_tf(self, docs: list[str]) -> np.ndarray:
        tf = np.zeros((len(docs), len(self.word_index)))
        for i, doc in enumerate(docs):
            word_counts = Counter(doc.split())
            for word, count in word_counts.items():
                if word in self.word_index:
                    tf[i, self.word_index[word]] = count
        # L2 normalization
        tf = tf / norm(tf, axis=1, keepdims=True)
        return tf

    def _compute_idf(self, docs: list[str]) -> np.ndarray:
        idf = np.zeros(len(self.word_index))
        for doc in docs:
            words = set(doc.split())
            for word in words:
                if word in self.word_index:
                    idf[self.word_index[word]] += 1
        idf = np.log(len(docs) / (idf + 1))
        return idf
