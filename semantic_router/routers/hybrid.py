from typing import Dict, List, Optional
import asyncio
from pydantic import Field

import numpy as np

from semantic_router.encoders import (
    DenseEncoder,
    SparseEncoder,
    BM25Encoder,
    TfidfEncoder,
)
from semantic_router.route import Route
from semantic_router.index import BaseIndex, HybridLocalIndex
from semantic_router.schema import RouteChoice, SparseEmbedding, Utterance
from semantic_router.utils.logger import logger
from semantic_router.routers.base import BaseRouter, xq_reshape
from semantic_router.llms import BaseLLM


class HybridRouter(BaseRouter):
    """A hybrid layer that uses both dense and sparse embeddings to classify routes."""

    # there are a few additional attributes for hybrid
    sparse_encoder: Optional[SparseEncoder] = Field(default=None)
    alpha: float = 0.3

    def __init__(
        self,
        encoder: DenseEncoder,
        sparse_encoder: Optional[SparseEncoder] = None,
        llm: Optional[BaseLLM] = None,
        routes: List[Route] = [],
        index: Optional[HybridLocalIndex] = None,
        top_k: int = 5,
        aggregation: str = "mean",
        auto_sync: Optional[str] = None,
        alpha: float = 0.3,
    ):
        if index is None:
            logger.warning("No index provided. Using default HybridLocalIndex.")
            index = HybridLocalIndex()
        encoder = self._get_encoder(encoder=encoder)
        super().__init__(
            encoder=encoder,
            llm=llm,
            routes=routes,
            index=index,
            top_k=top_k,
            aggregation=aggregation,
            auto_sync=auto_sync,
        )
        # initialize sparse encoder
        self.sparse_encoder = self._get_sparse_encoder(sparse_encoder=sparse_encoder)
        # set alpha
        self.alpha = alpha
        # fit sparse encoder if needed
        if (
            isinstance(self.sparse_encoder, TfidfEncoder)
            and hasattr(self.sparse_encoder, "fit")
            and self.routes
        ):
            self.sparse_encoder.fit(self.routes)
        # run initialize index now if auto sync is active
        if self.auto_sync:
            self._init_index_state()

    def add(self, routes: List[Route] | Route):
        """Add a route to the local HybridRouter and index.

        :param route: The route to add.
        :type route: Route
        """
        # TODO: merge into single method within BaseRouter
        current_local_hash = self._get_hash()
        current_remote_hash = self.index._read_hash()
        if current_remote_hash.value == "":
            # if remote hash is empty, the index is to be initialized
            current_remote_hash = current_local_hash
        if isinstance(routes, Route):
            routes = [routes]
        # create embeddings for all routes
        route_names, all_utterances, all_function_schemas, all_metadata = (
            self._extract_routes_details(routes, include_metadata=True)
        )
        # TODO: to merge, self._encode should probably output a special
        # TODO Embedding type that can be either dense or hybrid
        dense_emb, sparse_emb = self._encode(all_utterances)
        self.index.add(
            embeddings=dense_emb.tolist(),
            routes=route_names,
            utterances=all_utterances,
            function_schemas=all_function_schemas,
            metadata_list=all_metadata,
            sparse_embeddings=sparse_emb,  # type: ignore
        )

        self.routes.extend(routes)
        if current_local_hash.value == current_remote_hash.value:
            self._write_hash()  # update current hash in index
        else:
            logger.warning(
                "Local and remote route layers were not aligned. Remote hash "
                f"not updated. Use `{self.__class__.__name__}.get_utterance_diff()` "
                "to see details."
            )

    def _execute_sync_strategy(self, strategy: Dict[str, Dict[str, List[Utterance]]]):
        """Executes the provided sync strategy, either deleting or upserting
        routes from the local and remote instances as defined in the strategy.

        :param strategy: The sync strategy to execute.
        :type strategy: Dict[str, Dict[str, List[Utterance]]]
        """
        if strategy["remote"]["delete"]:
            data_to_delete = {}  # type: ignore
            for utt_obj in strategy["remote"]["delete"]:
                data_to_delete.setdefault(utt_obj.route, []).append(utt_obj.utterance)
            # TODO: switch to remove without sync??
            self.index._remove_and_sync(data_to_delete)
        if strategy["remote"]["upsert"]:
            utterances_text = [utt.utterance for utt in strategy["remote"]["upsert"]]
            dense_emb, sparse_emb = self._encode(utterances_text)
            self.index.add(
                embeddings=dense_emb.tolist(),
                routes=[utt.route for utt in strategy["remote"]["upsert"]],
                utterances=utterances_text,
                function_schemas=[
                    utt.function_schemas for utt in strategy["remote"]["upsert"]  # type: ignore
                ],
                metadata_list=[utt.metadata for utt in strategy["remote"]["upsert"]],
                sparse_embeddings=sparse_emb,  # type: ignore
            )
        if strategy["local"]["delete"]:
            self._local_delete(utterances=strategy["local"]["delete"])
        if strategy["local"]["upsert"]:
            self._local_upsert(utterances=strategy["local"]["upsert"])
        # update hash
        self._write_hash()

    def _get_index(self, index: Optional[BaseIndex]) -> BaseIndex:
        if index is None:
            logger.warning("No index provided. Using default HybridLocalIndex.")
            index = HybridLocalIndex()
        else:
            index = index
        return index

    def _get_sparse_encoder(
        self, sparse_encoder: Optional[SparseEncoder]
    ) -> SparseEncoder:
        if sparse_encoder is None:
            logger.warning("No sparse_encoder provided. Using default BM25Encoder.")
            sparse_encoder = BM25Encoder()
        else:
            sparse_encoder = sparse_encoder
        return sparse_encoder

    def _encode(self, text: list[str]) -> tuple[np.ndarray, list[SparseEmbedding]]:
        """Given some text, generates dense and sparse embeddings, then scales them
        using the chosen alpha value.
        """
        if self.sparse_encoder is None:
            raise ValueError("self.sparse_encoder is not set.")
        # TODO: should encode "content" rather than text
        # TODO: add alpha as a parameter
        # create dense query vector
        xq_d = np.array(self.encoder(text))
        # xq_d = np.squeeze(xq_d)  # Reduce to 1d array.
        # create sparse query vector dict
        xq_s = self.sparse_encoder(text)
        # xq_s = np.squeeze(xq_s)
        # convex scaling
        xq_d, xq_s = self._convex_scaling(dense=xq_d, sparse=xq_s)
        return xq_d, xq_s

    async def _async_encode(
        self, text: List[str]
    ) -> tuple[np.ndarray, list[SparseEmbedding]]:
        """Given some text, generates dense and sparse embeddings, then scales them
        using the chosen alpha value.
        """
        if self.sparse_encoder is None:
            raise ValueError("self.sparse_encoder is not set.")
        # TODO: should encode "content" rather than text
        # TODO: add alpha as a parameter
        # async encode both dense and sparse
        dense_coro = self.encoder.acall(text)
        sparse_coro = self.sparse_encoder.acall(text)
        dense_vec, xq_s = await asyncio.gather(dense_coro, sparse_coro)
        # create dense query vector
        xq_d = np.array(dense_vec)
        # convex scaling
        xq_d, xq_s = self._convex_scaling(dense=xq_d, sparse=xq_s)
        return xq_d, xq_s

    def __call__(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float] | np.ndarray] = None,
        simulate_static: bool = False,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> RouteChoice:
        potential_sparse_vector: List[SparseEmbedding] | None = None
        # if no vector provided, encode text to get vector
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector, potential_sparse_vector = self._encode(text=[text])
        # convert to numpy array if not already
        vector = xq_reshape(vector)
        if sparse_vector is None:
            if text is None:
                raise ValueError("Either text or sparse_vector must be provided")
            sparse_vector = (
                potential_sparse_vector[0] if potential_sparse_vector else None
            )
        if sparse_vector is None:
            raise ValueError("Sparse vector is required for HybridLocalIndex.")
        # TODO: add alpha as a parameter
        scores, route_names = self.index.query(
            vector=vector,
            top_k=self.top_k,
            route_filter=route_filter,
            sparse_vector=sparse_vector,
        )
        top_class, top_class_scores = self._semantic_classify(
            [
                {"score": score, "route": route}
                for score, route in zip(scores, route_names)
            ]
        )
        passed = self._pass_threshold(top_class_scores, self.score_threshold)
        if passed:
            return RouteChoice(name=top_class, similarity_score=max(top_class_scores))
        else:
            return RouteChoice()

    def _convex_scaling(
        self, dense: np.ndarray, sparse: list[SparseEmbedding]
    ) -> tuple[np.ndarray, list[SparseEmbedding]]:
        # TODO: better way to do this?
        sparse_dicts = [sparse_vec.to_dict() for sparse_vec in sparse]
        # scale sparse and dense vecs
        scaled_dense = np.array(dense) * self.alpha
        scaled_sparse = []
        for sparse_dict in sparse_dicts:
            scaled_sparse.append(
                SparseEmbedding.from_dict(
                    {k: v * (1 - self.alpha) for k, v in sparse_dict.items()}
                )
            )
        return scaled_dense, scaled_sparse
