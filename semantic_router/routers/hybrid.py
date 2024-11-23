from typing import Any, Dict, List, Optional, Tuple
import asyncio
from pydantic.v1 import validator, Field

import numpy as np
from numpy.linalg import norm

from semantic_router.encoders import (
    BaseEncoder,
    BM25Encoder,
    TfidfEncoder,
)
from semantic_router.route import Route
from semantic_router.index.hybrid_local import HybridLocalIndex
from semantic_router.schema import RouteChoice
from semantic_router.utils.logger import logger
from semantic_router.routers.base import BaseRouteLayer
from semantic_router.llms import BaseLLM


class HybridRouteLayer(BaseRouteLayer):
    """A hybrid layer that uses both dense and sparse embeddings to classify routes.
    """
    # there are a few additional attributes for hybrid
    sparse_encoder: BM25Encoder = Field(default_factory=BM25Encoder)
    alpha: float = 0.3
    index: HybridLocalIndex = Field(default_factory=HybridLocalIndex)

    def __init__(
        self,
        encoder: BaseEncoder,
        sparse_encoder: Optional[BM25Encoder] = None,
        llm: Optional[BaseLLM] = None,
        routes: List[Route] = [],
        index: Optional[HybridLocalIndex] = None,
        top_k: int = 5,
        aggregation: str = "mean",
        auto_sync: Optional[str] = None,
        alpha: float = 0.3,
    ):
        super().__init__(
            encoder=encoder,
            llm=llm,
            routes=routes.copy(),
            index=index,
            top_k=top_k,
            aggregation=aggregation,
            auto_sync=auto_sync,
        )
        # initialize sparse encoder
        if sparse_encoder is None:
            logger.warning("No sparse_encoder provided. Using default BM25Encoder.")
            self.sparse_encoder = BM25Encoder()
        else:
            self.sparse_encoder = sparse_encoder
        # set alpha
        self.alpha = alpha
        # fit sparse encoder if needed
        if isinstance(self.sparse_encoder, TfidfEncoder) and hasattr(
            self.sparse_encoder, "fit"
        ):
            self.sparse_encoder.fit(routes)
        # initialize index if not provided
        # TODO: add check for hybrid compatible index
        if self.index is None:
            logger.warning("No index provided. Using default HybridLocalIndex.")
            self.index = HybridLocalIndex()
        # add routes if we have them
        if routes:
            for route in routes:
                self.add(route)

    @validator("sparse_encoder", pre=True, always=True)
    def set_sparse_encoder(cls, v):
        return v if v is not None else BM25Encoder()
    
    @validator("index", pre=True, always=True)
    def set_index(cls, v):
        return v if v is not None else HybridLocalIndex()

    def _encode(self, text: List[str]) -> Any:
        """Given some text, generates dense and sparse embeddings, then scales them
        using the chosen alpha value.
        """
        # TODO: should encode "content" rather than text
        # TODO: add alpha as a parameter
        # create dense query vector
        xq_d = np.array(self.encoder(text))
        #xq_d = np.squeeze(xq_d)  # Reduce to 1d array.
        # create sparse query vector
        xq_s = np.array(self.sparse_encoder(text))
        #xq_s = np.squeeze(xq_s)
        # convex scaling
        xq_d, xq_s = self._convex_scaling(xq_d, xq_s)
        return xq_d, xq_s

    async def _async_encode(self, text: List[str]) -> Any:
        """Given some text, generates dense and sparse embeddings, then scales them
        using the chosen alpha value.
        """
        # TODO: should encode "content" rather than text
        # TODO: add alpha as a parameter
        # async encode both dense and sparse
        dense_coro = self.encoder.acall(text)
        sparse_coro = self.sparse_encoder.acall(text)
        dense_vec, sparse_vec = await asyncio.gather(dense_coro, sparse_coro)
        # create dense query vector
        xq_d = np.array(dense_vec)
        #xq_d = np.squeeze(xq_d)  # reduce to 1d array
        # create sparse query vector
        xq_s = np.array(sparse_vec)
        #xq_s = np.squeeze(xq_s)
        # convex scaling
        xq_d, xq_s = self._convex_scaling(xq_d, xq_s)
        return xq_d, xq_s

    def __call__(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
        simulate_static: bool = False,
        route_filter: Optional[List[str]] = None,
        sparse_vector: Optional[List[float]] = None,
    ) -> RouteChoice:
        # if no vector provided, encode text to get vector
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector, potential_sparse_vector = self._encode(text=[text])
        if sparse_vector is None:
            if text is None:
                raise ValueError("Either text or sparse_vector must be provided")
            sparse_vector = potential_sparse_vector
        # TODO: add alpha as a parameter
        scores, route_names = self.index.query(
            vector=np.array(vector) if isinstance(vector, list) else vector,
            top_k=self.top_k,
            route_filter=route_filter,
            sparse_vector=np.array(sparse_vector) if isinstance(sparse_vector, list) else sparse_vector,
        )
        top_class, top_class_scores = self._semantic_classify(list(zip(scores, route_names)))
        passed = self._pass_threshold(top_class_scores, self.score_threshold)
        if passed:
            return RouteChoice(
                name=top_class,
                similarity_score=max(top_class_scores)
            )
        else:
            return RouteChoice()

    def add(self, route: Route):
        self.routes += [route]

        route_names = [route.name] * len(route.utterances)

        # create embeddings for all routes
        logger.info(f"Encoding route {route.name}")
        dense_embeds, sparse_embeds = self._encode(route.utterances)
        self.index.add(
            embeddings=dense_embeds,
            sparse_embeddings=sparse_embeds,
            routes=route_names,  # TODO: aligning names of routes v route_names
            utterances=route.utterances,
        )
        # TODO: in some places we say vector, sparse_vector and in others
        # TODO: we say embeddings, sparse_embeddings

    def _convex_scaling(self, dense: np.ndarray, sparse: np.ndarray):
        # scale sparse and dense vecs
        dense = np.array(dense) * self.alpha
        sparse = np.array(sparse) * (1 - self.alpha)
        return dense, sparse

    def _set_aggregation_method(self, aggregation: str = "sum"):
        if aggregation == "sum":
            return lambda x: sum(x)
        elif aggregation == "mean":
            return lambda x: np.mean(x)
        elif aggregation == "max":
            return lambda x: max(x)
        else:
            raise ValueError(
                f"Unsupported aggregation method chosen: {aggregation}. Choose either 'SUM', 'MEAN', or 'MAX'."
            )

    def _semantic_classify(self, query_results: List[Tuple]) -> Tuple[str, List[float]]:
        scores_by_class: Dict[str, List[float]] = {}
        for score, route in query_results:
            if route in scores_by_class:
                scores_by_class[route].append(score)
            else:
                scores_by_class[route] = [score]

        # Calculate total score for each class
        total_scores = {
            route: self.aggregation_method(scores)
            for route, scores in scores_by_class.items()
        }
        top_class = max(total_scores, key=lambda x: total_scores[x], default=None)

        # Return the top class and its associated scores
        if top_class is not None:
            return str(top_class), scores_by_class.get(top_class, [])
        else:
            logger.warning("No classification found for semantic classifier.")
            return "", []

    def _pass_threshold(self, scores: List[float], threshold: float) -> bool:
        if scores:
            return max(scores) > threshold
        else:
            return False
