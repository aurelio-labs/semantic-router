import numpy as np
from numpy.linalg import norm
from tqdm.auto import tqdm

from semantic_router.encoders import (
    BaseEncoder,
    BM25Encoder,
    CohereEncoder,
    OpenAIEncoder,
)
from semantic_router.schema import Route
from semantic_router.utils.logger import logger


class HybridRouteLayer:
    index = None
    sparse_index = None
    categories = None
    score_threshold = 0.82

    def __init__(
        self, encoder: BaseEncoder, routes: list[Route] = [], alpha: float = 0.3
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
        # if routes list has been passed, we initialize index now
        if routes:
            # initialize index now
            for route in tqdm(routes):
                self._add_route(route=route)

    def __call__(self, text: str) -> str | None:
        results = self._query(text)
        top_class, top_class_scores = self._semantic_classify(results)
        passed = self._pass_threshold(top_class_scores, self.score_threshold)
        if passed:
            return top_class
        else:
            return None

    def add(self, route: Route):
        self._add_route(route=route)

    def _add_route(self, route: Route):
        # create embeddings
        dense_embeds = np.array(self.encoder(route.utterances))  # * self.alpha
        sparse_embeds = np.array(
            self.sparse_encoder(route.utterances)
        )  # * (1 - self.alpha)

        # create route array
        if self.categories is None:
            self.categories = np.array([route.name] * len(route.utterances))
            self.utterances = np.array(route.utterances)
        else:
            str_arr = np.array([route.name] * len(route.utterances))
            self.categories = np.concatenate([self.categories, str_arr])
            self.utterances = np.concatenate(
                [self.utterances, np.array(route.utterances)]
            )
        # create utterance array (the dense index)
        if self.index is None:
            self.index = dense_embeds
        else:
            self.index = np.concatenate([self.index, dense_embeds])
        # create sparse utterance array
        if self.sparse_index is None:
            self.sparse_index = sparse_embeds
        else:
            self.sparse_index = np.concatenate([self.sparse_index, sparse_embeds])

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

        if self.index is not None and self.sparse_index is not None:
            # calculate dense vec similarity
            index_norm = norm(self.index, axis=1)
            xq_d_norm = norm(xq_d.T)
            sim_d = np.dot(self.index, xq_d.T) / (index_norm * xq_d_norm)
            # calculate sparse vec similarity
            sparse_norm = norm(self.sparse_index, axis=1)
            xq_s_norm = norm(xq_s.T)
            sim_s = np.dot(self.sparse_index, xq_s.T) / (sparse_norm * xq_s_norm)
            total_sim = sim_d + sim_s
            # get indices of top_k records
            top_k = min(top_k, total_sim.shape[0])
            idx = np.argpartition(total_sim, -top_k)[-top_k:]
            scores = total_sim[idx]
            # get the utterance categories (route names)
            routes = self.categories[idx] if self.categories is not None else []
            return [{"route": d, "score": s.item()} for d, s in zip(routes, scores)]
        else:
            logger.warning("No index found. Please add routes to the layer.")
            return []

    def _convex_scaling(self, dense: np.ndarray, sparse: np.ndarray):
        # scale sparse and dense vecs
        dense = np.array(dense) * self.alpha
        sparse = np.array(sparse) * (1 - self.alpha)
        return dense, sparse

    def _semantic_classify(self, query_results: list[dict]) -> tuple[str, list[float]]:
        scores_by_class: dict[str, list[float]] = {}
        for result in query_results:
            score = result["score"]
            route = result["route"]
            if route in scores_by_class:
                scores_by_class[route].append(score)
            else:
                scores_by_class[route] = [score]

        # Calculate total score for each class
        total_scores = {route: sum(scores) for route, scores in scores_by_class.items()}
        top_class = max(total_scores, key=lambda x: total_scores[x], default=None)

        # Return the top class and its associated scores
        if top_class is not None:
            return str(top_class), scores_by_class.get(top_class, [])
        else:
            logger.warning("No classification found for semantic classifier.")
            return "", []

    def _pass_threshold(self, scores: list[float], threshold: float) -> bool:
        if scores:
            return max(scores) > threshold
        else:
            return False
