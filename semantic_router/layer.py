import numpy as np

from semantic_router.encoders import (
    BaseEncoder,
    CohereEncoder,
    OpenAIEncoder,
)
from semantic_router.linear import similarity_matrix, top_scores
from semantic_router.schema import Route
from semantic_router.utils.logger import logger


class RouteLayer:
    index = None
    categories = None
    score_threshold = 0.82

    def __init__(self, encoder: BaseEncoder, routes: list[Route] = []):
        self.encoder = encoder
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
            self.add_routes(routes=routes)

    def __call__(self, text: str) -> str | None:
        results = self._query(text)
        top_class, top_class_scores = self._semantic_classify(results)
        passed = self._pass_threshold(top_class_scores, self.score_threshold)
        if passed:
            return top_class
        else:
            return None

    def add_route(self, route: Route):
        # create embeddings
        embeds = self.encoder(route.utterances)

        # create route array
        if self.categories is None:
            self.categories = np.array([route.name] * len(embeds))
        else:
            str_arr = np.array([route.name] * len(embeds))
            self.categories = np.concatenate([self.categories, str_arr])
        # create utterance array (the index)
        if self.index is None:
            self.index = np.array(embeds)
        else:
            embed_arr = np.array(embeds)
            self.index = np.concatenate([self.index, embed_arr])

    def add_routes(self, routes: list[Route]):
        # create embeddings for all routes
        all_utterances = [
            utterance for route in routes for utterance in route.utterances
        ]
        embedded_utterance = self.encoder(all_utterances)

        # create route array
        route_names = [route.name for route in routes for _ in route.utterances]
        route_array = np.array(route_names)
        self.categories = (
            np.concatenate([self.categories, route_array])
            if self.categories is not None
            else route_array
        )

        # create utterance array (the index)
        embed_utterance_arr = np.array(embedded_utterance)
        self.index = (
            np.concatenate([self.index, embed_utterance_arr])
            if self.index is not None
            else embed_utterance_arr
        )

    def _query(self, text: str, top_k: int = 5):
        """Given some text, encodes and searches the index vector space to
        retrieve the top_k most similar records.
        """
        # create query vector
        xq = np.array(self.encoder([text]))
        xq = np.squeeze(xq)  # Reduce to 1d array.

        if self.index is not None:
            # calculate similarity matrix
            sim = similarity_matrix(xq, self.index)
            scores, idx = top_scores(sim, top_k)
            # get the utterance categories (route names)
            routes = self.categories[idx] if self.categories is not None else []
            return [{"route": d, "score": s.item()} for d, s in zip(routes, scores)]
        else:
            logger.warning("No index found for route layer.")
            return []

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
