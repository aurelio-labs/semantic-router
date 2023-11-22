import numpy as np
from numpy.linalg import norm

from semantic_router.encoders import (
    BaseEncoder,
    CohereEncoder,
    OpenAIEncoder,
    BM25Encoder
)
from semantic_router.schema import Decision


class DecisionLayer:
    index = None
    categories = None
    score_threshold = 0.82

    def __init__(self, encoder: BaseEncoder, decisions: list[Decision] = []):
        self.encoder = encoder
        # decide on default threshold based on encoder
        if isinstance(encoder, OpenAIEncoder):
            self.score_threshold = 0.82
        elif isinstance(encoder, CohereEncoder):
            self.score_threshold = 0.3
        else:
            self.score_threshold = 0.82
        # if decisions list has been passed, we initialize index now
        if decisions:
            # initialize index now
            for decision in decisions:
                self._add_decision(decision=decision)

    def __call__(self, text: str) -> str | None:
        results = self._query(text)
        top_class, top_class_scores = self._semantic_classify(results)
        passed = self._pass_threshold(top_class_scores, self.score_threshold)
        if passed:
            return top_class
        else:
            return None

    def add(self, decision: Decision):
        self._add_decision(decision=decision)

    def _add_decision(self, decision: Decision):
        # create embeddings
        embeds = self.encoder(decision.utterances)

        # create decision array
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

    def _query(self, text: str, top_k: int = 5):
        """Given some text, encodes and searches the index vector space to
        retrieve the top_k most similar records.
        """
        # create query vector
        xq = np.array(self.encoder([text]))
        xq = np.squeeze(xq)  # Reduce to 1d array.

        if self.index is not None:
            index_norm = norm(self.index, axis=1)
            xq_norm = norm(xq.T)
            sim = np.dot(self.index, xq.T) / (index_norm * xq_norm)
            # get indices of top_k records
            top_k = min(top_k, sim.shape[0])
            idx = np.argpartition(sim, -top_k)[-top_k:]
            scores = sim[idx]
            # get the utterance categories (decision names)
            decisions = self.categories[idx] if self.categories is not None else []
            return [
                {"decision": d, "score": s.item()} for d, s in zip(decisions, scores)
            ]
        else:
            return []

    def _semantic_classify(self, query_results: list[dict]) -> tuple[str, list[float]]:
        scores_by_class = {}
        for result in query_results:
            score = result["score"]
            decision = result["decision"]
            if decision in scores_by_class:
                scores_by_class[decision].append(score)
            else:
                scores_by_class[decision] = [score]

        # Calculate total score for each class
        total_scores = {
            decision: sum(scores) for decision, scores in scores_by_class.items()
        }
        top_class = max(total_scores, key=lambda x: total_scores[x], default=None)

        # Return the top class and its associated scores
        return str(top_class), scores_by_class.get(top_class, [])

    def _pass_threshold(self, scores: list[float], threshold: float) -> bool:
        if scores:
            return max(scores) > threshold
        else:
            return False


class HybridDecisionLayer:
    index = None
    categories = None
    score_threshold = 0.82

    def __init__(
        self,
        encoder: BaseEncoder,
        decisions: list[Decision] = [],
        alpha: float = 0.3
    ):
        self.encoder = encoder
        self.sparse_encoder = BM25Encoder()
        self.alpha = alpha
        # decide on default threshold based on encoder
        if isinstance(encoder, OpenAIEncoder):
            self.score_threshold = 0.82
        elif isinstance(encoder, CohereEncoder):
            self.score_threshold = 0.3
        else:
            self.score_threshold = 0.82
        # if decisions list has been passed, we initialize index now
        if decisions:
            # initialize index now
            for decision in decisions:
                self._add_decision(decision=decision)

    def __call__(self, text: str) -> str | None:
        results = self._query(text)
        top_class, top_class_scores = self._semantic_classify(results)
        passed = self._pass_threshold(top_class_scores, self.score_threshold)
        if passed:
            return top_class
        else:
            return None

    def add(self, decision: Decision):
        self._add_decision(decision=decision)

    def _add_decision(self, decision: Decision):
        # create embeddings
        dense_embeds = self.encoder(decision.utterances) * self.alpha
        sparse_embeds = self.sparse_encoder(decision.utterances) * (1 - self.alpha)
        # concatenate vectors to create hybrid vecs
        embeds = np.concatenate([
            dense_embeds, sparse_embeds
        ], axis=1)

        # create decision array
        if self.categories is None:
            self.categories = np.array([decision.name] * len(embeds))
            self.utterances = np.array(decision.utterances)
        else:
            str_arr = np.array([decision.name] * len(embeds))
            self.categories = np.concatenate([self.categories, str_arr])
            self.utterances = np.concatenate([
                self.utterances,
                np.array(decision.utterances)
            ])
        # create utterance array (the dense index)
        if self.index is None:
            self.index = np.array(dense_embeds)
        else:
            embed_arr = np.array(dense_embeds)
            self.index = np.concatenate([self.index, embed_arr])
        # create sparse utterance array
        if self.sparse_index is None:
            self.sparse_index = np.array(sparse_embeds)
        else:
            sparse_embed_arr = np.array(sparse_embeds)
            self.sparse_index = np.concatenate([
                self.sparse_index, sparse_embed_arr
            ])

    def _query(self, text: str, top_k: int = 5):
        """Given some text, encodes and searches the index vector space to
        retrieve the top_k most similar records.
        """
        # create dense query vector
        xq_d = np.array(self.encoder([text]))
        xq_d = np.squeeze(xq_d)  # Reduce to 1d array.
        # create sparse query vector
        xq_s = np.array(self.sparse_encoder([text]))
        xq_s = np.squeeze(xq_s)
        # convex scaling
        xq_d, xq_s = self._convex_scaling(xq_d, xq_s)
        # concatenate to create single hybrid vec
        xq = np.concatenate([xq_d, xq_s], axis=1)

        if self.index is not None:
            index_norm = norm(self.index, axis=1)
            xq_norm = norm(xq.T)
            sim = np.dot(self.index, xq.T) / (index_norm * xq_norm)
            # get indices of top_k records
            top_k = min(top_k, sim.shape[0])
            idx = np.argpartition(sim, -top_k)[-top_k:]
            scores = sim[idx]
            # get the utterance categories (decision names)
            decisions = self.categories[idx] if self.categories is not None else []
            return [
                {"decision": d, "score": s.item()} for d, s in zip(decisions, scores)
            ]
        else:
            return []
        
    def _convex_scaling(self, dense: list[float], sparse: list[float]):
        # scale sparse and dense vecs
        dense = dense * self.alpha
        sparse = sparse * (1 - self.alpha)
        return dense, sparse

    def _semantic_classify(self, query_results: list[dict]) -> tuple[str, list[float]]:
        scores_by_class = {}
        for result in query_results:
            score = result["score"]
            decision = result["decision"]
            if decision in scores_by_class:
                scores_by_class[decision].append(score)
            else:
                scores_by_class[decision] = [score]

        # Calculate total score for each class
        total_scores = {
            decision: sum(scores) for decision, scores in scores_by_class.items()
        }
        top_class = max(total_scores, key=lambda x: total_scores[x], default=None)

        # Return the top class and its associated scores
        return str(top_class), scores_by_class.get(top_class, [])

    def _pass_threshold(self, scores: list[float], threshold: float) -> bool:
        if scores:
            return max(scores) > threshold
        else:
            return False