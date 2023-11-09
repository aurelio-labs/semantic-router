import numpy as np
from numpy.linalg import norm

from semantic_router.encoders import BaseEncoder, CohereEncoder, OpenAIEncoder
from semantic_router.schema import Decision


class DecisionLayer:
    index = None
    categories = None
    similarity_threshold = 0.82

    def __init__(self, encoder: BaseEncoder, decisions: list[Decision] = []):
        self.encoder = encoder
        # decide on default threshold based on encoder
        if isinstance(encoder, OpenAIEncoder):
            self.similarity_threshold = 0.82
        elif isinstance(encoder, CohereEncoder):
            self.similarity_threshold = 0.3
        else:
            self.similarity_threshold = 0.82
        # if decisions list has been passed, we initialize index now
        if decisions:
            # initialize index now
            for decision in decisions:
                self._add_decision(decision=decision)

    def __call__(self, text: str) -> str | None:
        results = self._query(text)
        top_class, top_class_scores = self._semantic_classify(results)
        passed = self._pass_threshold(top_class_scores, self.similarity_threshold)
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
