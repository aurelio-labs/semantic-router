from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from semantic_router.encoders import BaseEncoder
from semantic_router.schema import Route


class TfidfEncoder(BaseEncoder):
    vectorizer: TfidfVectorizer | None = None

    def __init__(self, name: str = "tfidf"):
        super().__init__(name=name)
        self.vectorizer = TfidfVectorizer()

    def __call__(self, docs: list[str]) -> list[list[float]]:
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not initialized.")
        if len(docs) == 0:
            raise ValueError("No documents to encode.")

        embeds = self.vectorizer.transform(docs).toarray()
        return embeds.tolist()

    def fit(self, routes: list[Route]):
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not initialized.")
        docs = self._get_all_utterances(routes)
        self.vectorizer.fit(docs)

    def _get_all_utterances(self, routes: list[Route]) -> list[str]:
        utterances = []
        for route in routes:
            for utterance in route.utterances:
                utterances.append(utterance)
        return utterances
