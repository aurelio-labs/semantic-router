from semantic_router.encoders import BaseEncoder, OpenAIEncoder, CohereEncoder
from semantic_router.schema import Decision
import numpy as np
from numpy.linalg import norm


class DecisionLayer:
    index = None
    categories = None
    threshold = 0.82

    def __init__(self, encoder: BaseEncoder, decisions: list[Decision] = []):
        self.encoder = encoder
        # decide on default threshold based on encoder
        if isinstance(encoder, OpenAIEncoder):
            self.threshold = 0.82
        elif isinstance(encoder, CohereEncoder):
            self.threshold = 0.3
        else:
            self.threshold = 0.82
        # if decisions list has been passed, we initialize index now
        if decisions:
            # initialize index now
            for decision in decisions:
                self._add_decision(decision=decision)

    def __call__(self, text: str) -> str | None:
        results = self._query(text)
        top_class, top_class_scores  = self._semantic_classify(results)
        passed = self._pass_threshold(top_class_scores, self.threshold)
        if passed:
            return top_class
        else:
            return None

    def add(self, decision: Decision):
        self._add_decision(devision=decision)

    def _add_decision(self, decision: Decision):
        # create embeddings
        embeds = self.encoder(decision.utterances)

        # create decision array
        if self.categories is None:
            self.categories = np.array([decision.name]*len(embeds))
        else:
            str_arr = np.array([decision.name]*len(embeds))
            self.categories = np.concatenate([self.categories, str_arr])
        # create utterance array (the index)
        if self.index is None:
            self.index = np.array(embeds)
        else:
            embed_arr = np.array(embeds)
            self.index = np.concatenate([self.index, embed_arr])

    def _query(self, text: str, top_k: int=5):
        """Given some text, encodes and searches the index vector space to
        retrieve the top_k most similar records.
        """
        # create query vector
        xq = np.array(self.encoder([text]))
        xq = np.squeeze(xq) # Reduce to 1d array.
        sim = np.dot(self.index, xq.T) / (norm(self.index, axis=1)*norm(xq.T))
        # get indices of top_k records
        top_k = min(top_k, sim.shape[0])
        idx = np.argpartition(sim, -top_k)[-top_k:]
        scores = sim[idx]
        # get the utterance categories (decision names)
        decisions = self.categories[idx]
        return [
            {"decision": d, "score": s.item()} for d, s in zip(decisions, scores)
        ]

    def _semantic_classify(self, query_results: dict):
        scores_by_class = {}
        for result in query_results:
            score = result['score']
            decision = result['decision']
            if decision in scores_by_class:
                scores_by_class[decision].append(score)
            else:
                scores_by_class[decision] = [score]
        # Calculate total score for each class
        total_scores = {decision: sum(scores) for decision, scores in scores_by_class.items()}
        top_class = max(total_scores, key=total_scores.get, default=None)
        # Return the top class and its associated scores
        return str(top_class), scores_by_class.get(top_class, [])
    
    def _pass_threshold(self, scores: list[float], threshold: float):
        """Returns true if the threshold has been passed."""
        return max(scores) > threshold
