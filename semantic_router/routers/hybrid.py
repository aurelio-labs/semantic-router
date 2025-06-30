import asyncio
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import Field
from tqdm.auto import tqdm

from semantic_router.encoders import (
    BM25Encoder,
    DenseEncoder,
    SparseEncoder,
)
from semantic_router.encoders.base import (
    AsymmetricDenseMixin,
    AsymmetricSparseMixin,
    FittableMixin,
)
from semantic_router.encoders.encode_input_type import EncodeInputType
from semantic_router.index import BaseIndex, HybridLocalIndex
from semantic_router.llms import BaseLLM
from semantic_router.route import Route
from semantic_router.routers.base import BaseRouter, threshold_random_search, xq_reshape
from semantic_router.schema import RouteChoice, SparseEmbedding, Utterance
from semantic_router.utils.logger import logger


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
        init_async_index: bool = False,
    ):
        """Initialize the HybridRouter.

        :param encoder: The dense encoder to use.
        :type encoder: DenseEncoder
        :param sparse_encoder: The sparse encoder to use.
        :type sparse_encoder: Optional[SparseEncoder]
        """
        if index is None:
            logger.warning("No index provided. Using default HybridLocalIndex.")
            index = HybridLocalIndex()
        encoder = self._get_encoder(encoder=encoder)
        # initialize sparse encoder
        sparse_encoder = self._get_sparse_encoder(sparse_encoder=sparse_encoder)
        # fit sparse encoder if needed
        if isinstance(sparse_encoder, FittableMixin) and routes:
            sparse_encoder.fit(routes)
        super().__init__(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            llm=llm,
            routes=routes,
            index=index,
            top_k=top_k,
            aggregation=aggregation,
            auto_sync=auto_sync,
            init_async_index=init_async_index,
        )
        # set alpha
        self.alpha = alpha

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

        if self.sparse_encoder is None:
            raise ValueError("Sparse Encoder not initialised.")
        # TODO: merge into single method within BaseRouter
        current_local_hash = self._get_hash()
        current_remote_hash = self.index._read_hash()
        if current_remote_hash.value == "":
            # if remote hash is empty, the index is to be initialized
            current_remote_hash = current_local_hash
        if isinstance(routes, Route):
            routes = [routes]

        self.routes.extend(routes)
        if isinstance(self.sparse_encoder, FittableMixin) and self.routes:
            self.sparse_encoder.fit(self.routes)
        # create embeddings for all routes
        (
            route_names,
            all_utterances,
            all_function_schemas,
            all_metadata,
        ) = self._extract_routes_details(routes, include_metadata=True)
        # TODO: to merge, self._encode should probably output a special
        # TODO Embedding type that can be either dense or hybrid
        dense_emb, sparse_emb = self._encode(all_utterances, input_type="documents")
        self.index.add(
            embeddings=dense_emb.tolist(),
            routes=route_names,
            utterances=all_utterances,
            function_schemas=all_function_schemas,
            metadata_list=all_metadata,
            sparse_embeddings=sparse_emb,
        )

        if current_local_hash.value == current_remote_hash.value:
            self._write_hash()  # update current hash in index
        else:
            logger.warning(
                "Local and remote route layers were not aligned. Remote hash "
                f"not updated. Use `{self.__class__.__name__}.get_utterance_diff()` "
                "to see details."
            )

    async def aadd(self, routes: List[Route] | Route):
        """Add a route to the local HybridRouter and index asynchronously.

        :param routes: The route(s) to add.
        :type routes: List[Route] | Route
        """
        if self.sparse_encoder is None:
            raise ValueError("Sparse Encoder not initialised.")

        # TODO: merge into single method within BaseRouter
        current_local_hash = self._get_hash()
        current_remote_hash = await self.index._async_read_hash()
        if current_remote_hash.value == "":
            # if remote hash is empty, the index is to be initialized
            current_remote_hash = current_local_hash

        if isinstance(routes, Route):
            routes = [routes]

        self.routes.extend(routes)
        if isinstance(self.sparse_encoder, FittableMixin) and self.routes:
            self.sparse_encoder.fit(self.routes)

        # create embeddings for all routes
        (
            route_names,
            all_utterances,
            all_function_schemas,
            all_metadata,
        ) = self._extract_routes_details(routes, include_metadata=True)

        # TODO: to merge, self._encode should probably output a special
        # TODO Embedding type that can be either dense or hybrid
        dense_emb, sparse_emb = await self._async_encode(
            all_utterances, input_type="documents"
        )

        await self.index.aadd(
            embeddings=dense_emb.tolist(),
            routes=route_names,
            utterances=all_utterances,
            function_schemas=all_function_schemas,
            metadata_list=all_metadata,
            sparse_embeddings=sparse_emb,
        )

        if current_local_hash.value == current_remote_hash.value:
            await self._async_write_hash()  # update current hash in index
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
        if self.sparse_encoder is None:
            raise ValueError("Sparse Encoder not initialised.")
        if strategy["remote"]["delete"]:
            data_to_delete = {}  # type: ignore
            for utt_obj in strategy["remote"]["delete"]:
                data_to_delete.setdefault(utt_obj.route, []).append(utt_obj.utterance)
            # TODO: switch to remove without sync??
            self.index._remove_and_sync(data_to_delete)
        if strategy["remote"]["upsert"]:
            utterances_text = [utt.utterance for utt in strategy["remote"]["upsert"]]
            dense_emb, sparse_emb = self._encode(
                utterances_text, input_type="documents"
            )
            self.index.add(
                embeddings=dense_emb.tolist(),
                routes=[utt.route for utt in strategy["remote"]["upsert"]],
                utterances=utterances_text,
                function_schemas=[
                    utt.function_schemas  # type: ignore
                    for utt in strategy["remote"]["upsert"]
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
        if isinstance(self.sparse_encoder, FittableMixin) and self.routes:
            self.sparse_encoder.fit(self.routes)

    def _get_index(self, index: Optional[BaseIndex]) -> BaseIndex:
        """Get the index.

        :param index: The index to get.
        :type index: Optional[BaseIndex]
        :return: The index.
        :rtype: BaseIndex
        """
        if index is None:
            logger.warning("No index provided. Using default HybridLocalIndex.")
            index = HybridLocalIndex()
        else:
            index = index
        return index

    def _get_sparse_encoder(
        self, sparse_encoder: Optional[SparseEncoder]
    ) -> SparseEncoder:
        """Get the sparse encoder.

        :param sparse_encoder: The sparse encoder to get.
        :type sparse_encoder: Optional[SparseEncoder]
        :return: The sparse encoder.
        :rtype: Optional[SparseEncoder]
        """
        if sparse_encoder is None:
            logger.warning("No sparse_encoder provided. Using default BM25Encoder.")
            sparse_encoder = BM25Encoder()
        else:
            sparse_encoder = sparse_encoder
        return sparse_encoder

    def _encode(
        self, text: list[str], input_type: EncodeInputType
    ) -> tuple[np.ndarray, list[SparseEmbedding]]:
        """Given some text, generates dense and sparse embeddings, then scales them
        using the chosen alpha value.

        :param text: List of texts to encode
        :type text: List[str]
        :param input_type: Specify whether encoding 'queries' or 'documents', used in asymmetric retrieval
        :type input_type: semantic_router.encoders.encode_input_type.EncodeInputType
        :return: Tuple of dense and sparse embeddings
        """
        if self.sparse_encoder is None:
            raise ValueError("self.sparse_encoder is not set.")

        if isinstance(self.encoder, AsymmetricDenseMixin):
            match input_type:
                case "queries":
                    dense_v = self.encoder.encode_queries(text)
                case "documents":
                    dense_v = self.encoder.encode_documents(text)
        else:
            dense_v = self.encoder(text)
        xq_d = np.array(dense_v)  # type: ignore

        if isinstance(self.sparse_encoder, AsymmetricSparseMixin):
            match input_type:
                case "queries":
                    xq_s = self.sparse_encoder.encode_queries(text)
                case "documents":
                    xq_s = self.sparse_encoder.encode_documents(text)
        else:
            xq_s = self.sparse_encoder(text)

        # Convex scaling
        xq_d, xq_s = self._convex_scaling(dense=xq_d, sparse=xq_s)
        return xq_d, xq_s

    async def _async_encode(
        self, text: List[str], input_type: EncodeInputType
    ) -> tuple[np.ndarray, list[SparseEmbedding]]:
        """Given some text, generates dense and sparse embeddings, then scales them
        using the chosen alpha value.

        :param text: The text to encode.
        :type text: List[str]
        :param input_type: Specify whether encoding 'queries' or 'documents', used in asymmetric retrieval
        :type input_type: semantic_router.encoders.encode_input_type.EncodeInputType
        :return: A tuple of the dense and sparse embeddings.
        :rtype: tuple[np.ndarray, list[SparseEmbedding]]
        """
        if self.sparse_encoder is None:
            raise ValueError("self.sparse_encoder is not set.")
        # TODO: should encode "content" rather than text
        # TODO: add alpha as a parameter
        # async encode both dense and sparse

        if isinstance(self.encoder, AsymmetricDenseMixin):
            match input_type:
                case "queries":
                    dense_coro = self.encoder.aencode_queries(text)
                case "documents":
                    dense_coro = self.encoder.aencode_documents(text)
        else:
            dense_coro = self.encoder.acall(text)

        if isinstance(self.sparse_encoder, AsymmetricSparseMixin):
            match input_type:
                case "queries":
                    sparse_coro = self.sparse_encoder.aencode_queries(text)
                case "documents":
                    sparse_coro = self.sparse_encoder.aencode_documents(text)
        else:
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
        limit: int | None = 1,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> RouteChoice | list[RouteChoice]:
        """Call the HybridRouter.

        :param text: The text to encode.
        :type text: Optional[str]
        :param vector: The vector to encode.
        :type vector: Optional[List[float] | np.ndarray]
        :param simulate_static: Whether to simulate a static route.
        :type simulate_static: bool
        :param route_filter: The route filter to use.
        :type route_filter: Optional[List[str]]
        :param limit: The number of routes to return, defaults to 1. If set to None, no
            limit is applied and all routes are returned.
        :type limit: int | None
        :param sparse_vector: The sparse vector to use.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A RouteChoice or a list of RouteChoices.
        :rtype: RouteChoice | list[RouteChoice]
        """
        if not self.index.is_ready():
            raise ValueError("Index is not ready.")
        if self.sparse_encoder is None:
            raise ValueError("Sparse encoder is not set.")
        potential_sparse_vector: List[SparseEmbedding] | None = None
        # if no vector provided, encode text to get vector
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            xq_d = np.array(self.encoder([text]))
            xq_s = self.sparse_encoder([text])
            vector, potential_sparse_vector = self._convex_scaling(
                dense=xq_d, sparse=xq_s
            )
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
        # decide most relevant routes
        scored_routes = self._score_routes(query_results=query_results)
        route_choices = self._pass_routes(
            scored_routes=scored_routes,
            simulate_static=simulate_static,
            text=text,
            limit=limit,
        )
        return route_choices

    async def acall(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float] | np.ndarray] = None,
        limit: int | None = 1,
        simulate_static: bool = False,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> RouteChoice | list[RouteChoice]:
        """Asynchronously call the router to get a route choice.

        :param text: The text to route.
        :type text: Optional[str]
        :param vector: The vector to route.
        :type vector: Optional[List[float] | np.ndarray]
        :param simulate_static: Whether to simulate a static route (ie avoid dynamic route
            LLM calls during fit or evaluate).
        :type simulate_static: bool
        :param route_filter: The route filter to use.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to use.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: The route choice.
        :rtype: RouteChoice
        """
        if not (await self.index.ais_ready()):
            # TODO: need async version for qdrant
            await self._async_init_index_state()
        if self.sparse_encoder is None:
            raise ValueError("Sparse encoder is not set.")
        potential_sparse_vector: List[SparseEmbedding] | None = None
        # if no vector provided, encode text to get vector
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector, potential_sparse_vector = await self._async_encode(
                text=[text], input_type="queries"
            )
        # convert to numpy array if not already
        vector = xq_reshape(xq=vector)
        if sparse_vector is None:
            if text is None:
                raise ValueError("Either text or sparse_vector must be provided")
            sparse_vector = (
                potential_sparse_vector[0] if potential_sparse_vector else None
            )
        # get scores and routes
        scores, routes = await self.index.aquery(
            vector=vector[0],
            top_k=self.top_k,
            route_filter=route_filter,
            sparse_vector=sparse_vector,
        )
        query_results = [
            {"route": d, "score": s.item()} for d, s in zip(routes, scores)
        ]
        scored_routes = self._score_routes(query_results=query_results)
        return await self._async_pass_routes(
            scored_routes=scored_routes,
            simulate_static=simulate_static,
            text=text,
            limit=limit,
        )

    async def _async_execute_sync_strategy(
        self, strategy: Dict[str, Dict[str, List[Utterance]]]
    ):
        """Executes the provided sync strategy, either deleting or upserting
        routes from the local and remote instances as defined in the strategy.

        :param strategy: The sync strategy to execute.
        :type strategy: Dict[str, Dict[str, List[Utterance]]]
        """
        if self.sparse_encoder is None:
            raise ValueError("Sparse encoder is not set.")
        if strategy["remote"]["delete"]:
            data_to_delete = {}  # type: ignore
            for utt_obj in strategy["remote"]["delete"]:
                data_to_delete.setdefault(utt_obj.route, []).append(utt_obj.utterance)
            # TODO: switch to remove without sync??
            await self.index._async_remove_and_sync(data_to_delete)
        if strategy["remote"]["upsert"]:
            utterances_text = [utt.utterance for utt in strategy["remote"]["upsert"]]
            await self.index.aadd(
                embeddings=await self.encoder.acall(docs=utterances_text),
                sparse_embeddings=await self.sparse_encoder.acall(docs=utterances_text),
                routes=[utt.route for utt in strategy["remote"]["upsert"]],
                utterances=utterances_text,
                function_schemas=[
                    utt.function_schemas  # type: ignore
                    for utt in strategy["remote"]["upsert"]
                ],
                metadata_list=[utt.metadata for utt in strategy["remote"]["upsert"]],
            )
        if strategy["local"]["delete"]:
            # assumption is that with simple local delete we don't benefit from async
            self._local_delete(utterances=strategy["local"]["delete"])
        if strategy["local"]["upsert"]:
            # same assumption as with local delete above
            self._local_upsert(utterances=strategy["local"]["upsert"])
        # update hash
        await self._async_write_hash()

    def _convex_scaling(
        self, dense: np.ndarray, sparse: list[SparseEmbedding]
    ) -> tuple[np.ndarray, list[SparseEmbedding]]:
        """Convex scaling of the dense and sparse vectors.

        :param dense: The dense vector to scale.
        :type dense: np.ndarray
        :param sparse: The sparse vector to scale.
        :type sparse: list[SparseEmbedding]
        """
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
        """Fit the HybridRouter.

        :param X: The input data.
        :type X: List[str]
        :param y: The output data.
        :type y: List[str]
        :param batch_size: The batch size to use for fitting.
        :type batch_size: int
        :param max_iter: The maximum number of iterations to use for fitting.
        :type max_iter: int
        :param local_execution: Whether to execute the fitting locally.
        :type local_execution: bool
        """
        original_index = self.index
        if self.sparse_encoder is None:
            raise ValueError("Sparse encoder is not set.")
        if local_execution:
            # Switch to a local index for fitting
            from semantic_router.index.hybrid_local import HybridLocalIndex

            remote_utterances = self.index.get_utterances(include_metadata=True)
            # TODO Enhance by retrieving directly the vectors instead of embedding all utterances again
            routes = []
            utterances = []
            metadata = []
            for utterance in remote_utterances:
                routes.append(utterance.route)
                utterances.append(utterance.utterance)
                metadata.append(utterance.metadata)
            embeddings = (
                self.encoder(utterances)
                if not isinstance(self.encoder, AsymmetricDenseMixin)
                else self.encoder.encode_documents(utterances)
            )
            sparse_embeddings = (
                self.sparse_encoder(utterances)
                if not isinstance(self.sparse_encoder, AsymmetricSparseMixin)
                else self.sparse_encoder.encode_documents(utterances)
            )
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
            emb_d = np.array(
                self.encoder(X[i : i + batch_size])
                if not isinstance(self.encoder, AsymmetricDenseMixin)
                else self.encoder.encode_queries(X[i : i + batch_size])
            )
            # TODO JB: for some reason the sparse encoder is receiving a tuple
            # like `("Hello",)`
            emb_s = (
                self.sparse_encoder(X[i : i + batch_size])
                if not isinstance(self.sparse_encoder, AsymmetricSparseMixin)
                else self.sparse_encoder.encode_queries(X[i : i + batch_size])
            )

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
        """Evaluate the accuracy of the route selection.

        :param X: The input data.
        :type X: List[str]
        :param y: The output data.
        :type y: List[str]
        :param batch_size: The batch size to use for evaluation.
        :type batch_size: int
        :return: The accuracy of the route selection.
        :rtype: float
        """
        if self.sparse_encoder is None:
            raise ValueError("Sparse encoder is not set.")
        Xq_d: List[List[float]] = []
        Xq_s: List[SparseEmbedding] = []
        for i in tqdm(range(0, len(X), batch_size), desc="Generating embeddings"):
            emb_d = np.array(
                self.encoder(X[i : i + batch_size])
                if not isinstance(self.encoder, AsymmetricDenseMixin)
                else self.encoder.encode_queries(X[i : i + batch_size])
            )
            emb_s = (
                self.sparse_encoder(X[i : i + batch_size])
                if not isinstance(self.sparse_encoder, AsymmetricSparseMixin)
                else self.sparse_encoder.encode_queries(X[i : i + batch_size])
            )
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
        """Evaluate the accuracy of the route selection.

        :param Xq_d: The dense vectors to evaluate.
        :type Xq_d: Union[List[float], Any]
        :param Xq_s: The sparse vectors to evaluate.
        :type Xq_s: list[SparseEmbedding]
        :param y: The output data.
        :type y: List[str]
        :return: The accuracy of the route selection.
        :rtype: float
        """
        correct = 0
        for xq_d, xq_s, target_route in zip(Xq_d, Xq_s, y):
            # We treate dynamic routes as static here, because when evaluating we use only vectors, and dynamic routes expect strings by default.
            route_choice = self(vector=xq_d, sparse_vector=xq_s, simulate_static=True)
            if isinstance(route_choice, list):
                route_name = route_choice[0].name
            else:
                route_name = route_choice.name
            if route_name == target_route:
                correct += 1
        accuracy = correct / len(Xq_d)
        return accuracy
