from decision_layer.encoders import BaseEncoder
from decision_layer.schema import Decision
import numpy as np
from numpy.linalg import norm
class DecisionLayer:
    index = None
    categories = None

    def __init__(self, encoder: BaseEncoder, decisions: list[Decision] = []):
        self.encoder = encoder
        self.embeddings_classified = False
        # if decisions list has been passed, we initialize index now
        if decisions:
            # initialize index now
            for decision in decisions:
                self._add_decision(decision=decision)


    def __call__(self, text: str, _method: str='raw', _threshold: float=0.5):
        results = self._query(text)
        decision = self._semantic_classify(results, _method=_method, _threshold=_threshold)
        # return decision
        return decision

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

    def _cosine_similarity(self, v1, v2):
        """Compute the dot product between two embeddings using numpy functions."""
        np_v1 = np.array(v1)
        np_v2 = np.array(v2)
        return np.dot(np_v1, np_v2) / (np.linalg.norm(np_v1) * np.linalg.norm(np_v2))

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
    

    def _semantic_classify(self, query_results: dict, _method: str='raw', _threshold: float=0.5):
        """Given some text, categorizes."""

        # Initialize score dictionaries
        scores_by_class = {}
        highest_score_by_class = {}

        # Define valid methods
        valid_methods = ['raw', 'tan', 'max_score_in_top_class']

        # Check if method is valid
        if _method not in valid_methods:
            raise ValueError(f"Invalid method: {_method}")

        # Apply the scoring system to the results and group by category
        for result in query_results:
            decision = result['decision']
            score = result['score']

            # Apply tan transformation if method is 'tan'
            if _method == 'tan':
                score = np.tan(score * (np.pi / 2))

            # Update scores_by_class
            scores_by_class[decision] = scores_by_class.get(decision, 0) + score

            # Update highest_score_by_class for 'max_score_in_top_class' method
            if _method == 'max_score_in_top_class':
                highest_score_by_class[decision] = max(score, highest_score_by_class.get(decision, 0))

        # Sort the categories by score in descending order
        sorted_classes = sorted(scores_by_class.items(), key=lambda x: x[1], reverse=True)

        # Determine if the score is sufficiently high
        predicted_class = None
        if sorted_classes:
            top_class, top_score = sorted_classes[0]
            if _method == 'max_score_in_top_class':
                top_score = highest_score_by_class[top_class]
            if top_score > _threshold:
                predicted_class = top_class

        # Return the category with the highest total score
        return predicted_class, scores_by_class