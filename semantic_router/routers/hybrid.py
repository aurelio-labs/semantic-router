from typing import Any, Dict, List, Optional, Union
from tqdm.auto import tqdm
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
from semantic_router.routers.base import BaseRouter, xq_reshape, threshold_random_search
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
        routes: Optional[List[Route]] = None,
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
        # initialize sparse encoder
        sparse_encoder = self._get_sparse_encoder(sparse_encoder=sparse_encoder)
        super().__init__(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            llm=llm,
            routes=routes,
            index=index,
            top_k=top_k,
            aggregation=aggregation,
            auto_sync=auto_sync,
        )
        # set alpha
        self.alpha = alpha
        # fit sparse encoder if needed
        if (
            isinstance(self.sparse_encoder, TfidfEncoder)
            and hasattr(self.sparse_encoder, "fit")
            and self.routes
        ):
            self.sparse_encoder.fit(self.routes)

    def _set_score_threshold(self):
        """Set the score threshold for the HybridRouter. Unlike the base router the
        encoder score threshold is not used directly. Instead, the dense encoder
        score threshold is multiplied by the alpha value, resulting in a lower
        score threshold. This is done to account for the difference in returned
        scores from the hybrid router.
        """
        if self.encoder.score_threshold is not None:
            self.score_threshold = self.encoder.score_threshold * self.alpha
            if self.score_threshold is None:
                logger.warning(
                    "No score threshold value found in encoder. Using the default "
                    "'None' value can lead to unexpected results."
                )

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
            sparse_embeddings=sparse_emb,
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
                sparse_embeddings=sparse_emb,
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
    ) -> Optional[SparseEncoder]:
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
        if not self.index.is_ready():
            raise ValueError("Index is not ready.")
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
            vector=vector[0],
            top_k=self.top_k,
            route_filter=route_filter,
            sparse_vector=sparse_vector,
        )
        query_results = [
            {"route": d, "score": s.item()} for d, s in zip(route_names, scores)
        ]
        # TODO JB we should probably make _semantic_classify consume arrays rather than
        # needing to convert to list here
        top_class, top_class_scores = self._semantic_classify(
            query_results=query_results
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

    def fit(
        self,
        X: List[str],
        y: List[str],
        batch_size: int = 500,
        max_iter: int = 500,
        local_execution: bool = False,
    ):
        original_index = self.index
        if self.sparse_encoder is None:
            raise ValueError("Sparse encoder is not set.")
        if local_execution:
            # Switch to a local index for fitting
            from semantic_router.index.hybrid_local import HybridLocalIndex

            remote_routes = self.index.get_utterances(include_metadata=True)
            # TODO Enhance by retrieving directly the vectors instead of embedding all utterances again
            routes, utterances, function_schemas, metadata = map(
                list, zip(*remote_routes)
            )
            embeddings = self.encoder(utterances)
            sparse_embeddings = self.sparse_encoder(utterances)
            self.index = HybridLocalIndex()
            self.index.add(
                embeddings=embeddings,
                sparse_embeddings=sparse_embeddings,
                routes=routes,
                utterances=utterances,
                metadata_list=metadata,
            )

        # convert inputs into array
        Xq_d: List[List[float]] = []
        Xq_s: List[SparseEmbedding] = []
        for i in tqdm(range(0, len(X), batch_size), desc="Generating embeddings"):
            emb_d = np.array(self.encoder(X[i : i + batch_size]))
            # TODO JB: for some reason the sparse encoder is receiving a tuple
            # like `("Hello",)`
            emb_s = self.sparse_encoder(X[i : i + batch_size])
            Xq_d.extend(emb_d)
            Xq_s.extend(emb_s)
        # initial eval (we will iterate from here)
        best_acc = self._vec_evaluate(Xq_d=np.array(Xq_d), Xq_s=Xq_s, y=y)
        best_thresholds = self.get_thresholds()
        # begin fit
        for _ in (pbar := tqdm(range(max_iter), desc="Training")):
            pbar.set_postfix({"acc": round(best_acc, 2)})
            # Find the best score threshold for each route
            thresholds = threshold_random_search(
                route_layer=self,
                search_range=0.8,
            )
            # update current route layer
            self._update_thresholds(route_thresholds=thresholds)
            # evaluate
            acc = self._vec_evaluate(Xq_d=np.array(Xq_d), Xq_s=Xq_s, y=y)
            # update best
            if acc > best_acc:
                best_acc = acc
                best_thresholds = thresholds
        # update route layer to best thresholds
        self._update_thresholds(route_thresholds=best_thresholds)

        if local_execution:
            # Switch back to the original index
            self.index = original_index

    def evaluate(self, X: List[str], y: List[str], batch_size: int = 500) -> float:
        """
        Evaluate the accuracy of the route selection.
        """
        if self.sparse_encoder is None:
            raise ValueError("Sparse encoder is not set.")
        Xq_d: List[List[float]] = []
        Xq_s: List[SparseEmbedding] = []
        for i in tqdm(range(0, len(X), batch_size), desc="Generating embeddings"):
            emb_d = np.array(self.encoder(X[i : i + batch_size]))
            emb_s = self.sparse_encoder(X[i : i + batch_size])
            Xq_d.extend(emb_d)
            Xq_s.extend(emb_s)

        accuracy = self._vec_evaluate(Xq_d=np.array(Xq_d), Xq_s=Xq_s, y=y)
        return accuracy

    def _vec_evaluate(  # type: ignore
        self,
        Xq_d: Union[List[float], Any],
        Xq_s: list[SparseEmbedding],
        y: List[str],
    ) -> float:
        """
        Evaluate the accuracy of the route selection.
        """
        correct = 0
        for xq_d, xq_s, target_route in zip(Xq_d, Xq_s, y):
            # We treate dynamic routes as static here, because when evaluating we use only vectors, and dynamic routes expect strings by default.
            route_choice = self(vector=xq_d, sparse_vector=xq_s, simulate_static=True)
            if route_choice.name == target_route:
                correct += 1
        accuracy = correct / len(Xq_d)
        return accuracy
