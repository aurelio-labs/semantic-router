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

    def __call__(self, text: str):

        results = self._query(text)
        decision = self.simple_categorise(results)
        # return decision
        raise NotImplementedError("To implement decision logic based on scores")


    def add(self, decision: Decision, dimensiona):
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
        # calculate cosine similarities
        sim = np.dot(self.index, xq.T) / (norm(self.index)*norm(xq.T))
        # get indices of top_k records
        idx = np.argpartition(sim.T[0], -top_k)[-top_k:]
        scores = sim[idx]
        # get the utterance categories (decision names)
        decisions = self.categories[idx]
        return [
            {"decision": d, "score": s.item()} for d, s in zip(decisions, scores)
        ]

    def simple_categorise(self, text: str, top_k: int=5, apply_tan: bool=True):
        """Given some text, categorises it based on the scores from _query."""
        # get the results from _query
        results = self._query(text, top_k)
        
        # apply the scoring system to the results and group by category
        scores_by_category = {}
        for result in results:
            score = np.tan(result['score'] * (np.pi / 2)) if apply_tan else result['score']
            if result['decision'] in scores_by_category:
                scores_by_category[result['decision']] += score
            else:
                scores_by_category[result['decision']] = score
        
        # sort the categories by score in descending order
        sorted_categories = sorted(scores_by_category.items(), key=lambda x: x[1], reverse=True)
        
        # return the category with the highest total score
        return sorted_categories[0][0] if sorted_categories else None
    

