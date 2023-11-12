import numpy as np

from semantic_router.rankers import (
    BaseRanker,
    CohereRanker
)
from semantic_router.retrievers import (
    BaseRetriever,
    CohereRetriever
)
from semantic_router.matchers import BaseMatcher
from semantic_router.schema import Decision


class TwoStageMatcher(BaseMatcher):
    def __init__(
        self,
        retriever: BaseRetriever | None,
        ranker: BaseRanker | None,
        top_k: int = 25,
        top_n: int = 5
    ):
        super().__init__(
            retriever=retriever, ranker=ranker, top_k=top_k, top_n=top_n
        )
        if retriever is None:
            retriever = CohereRetriever(
                name="embed-english-v3.0",
                top_k=top_k
            )
        if ranker is None:
            ranker = CohereRanker(
                name="rerank-english-v2.0",
                top_n=top_n
            )
    
    def __call__(self, query: str, decisions: list[Decision]) -> str:
        pass

    def add(self, decision: Decision):
        self._add_decision(decision=decision)

    def _add_decision(self, decision: Decision):
        # create embeddings for first stage
        embeds = self.retriever(decision.utterances)
        # create a decision array for decision categories
        if self.categories is None:
            self.categories = np.array([decision.name] * len(embeds))
        else:
            str_arr = np.array([decision.name] * len(embeds))
            self.categories = np.concatenate([self.categories, str_arr])
        # create utterance array (the index)
        if self.index is None:
            self.index = np.array(embeds)
        else:
            embed_arr = np.array(embeds)
            self.index = np.concatenate([self.index, embed_arr])

    