"""a router that scores by similarity between the query and stored example utterances."""

from __future__ import annotations

import asyncio
from typing import Callable, Literal, Sequence

import numpy as np

from semantic_router.calibration import Calibrator, Method
from semantic_router.encoders import DenseEncoder, SparseEncoder
from semantic_router.encoders.base import (
    AsymmetricDenseMixin,
    AsymmetricSparseMixin,
    FittableMixin,
)
from semantic_router.index import BaseIndex, HybridLocalIndex, LocalIndex
from semantic_router.route import Route
from semantic_router.routers.base import BaseRouter, FitReport
from semantic_router.schema import SparseEmbedding
from semantic_router.utils.logger import logger

Kind = Literal["queries", "documents"]
Aggregation = Literal["mean", "max", "sum"]
_AGG: dict[str, Callable[[list[float]], float]] = {
    "mean": lambda v: float(np.mean(v)),
    "max": max,
    "sum": sum,
}


def _encode(enc: DenseEncoder | SparseEncoder, texts: list[str], kind: Kind) -> list:
    # asymmetric encoders embed queries and documents differently; the rest have one path
    if isinstance(enc, (AsymmetricDenseMixin, AsymmetricSparseMixin)):
        return (
            enc.encode_queries(texts)
            if kind == "queries"
            else enc.encode_documents(texts)
        )
    return enc(texts)


async def _aencode(
    enc: DenseEncoder | SparseEncoder, texts: list[str], kind: Kind
) -> list:
    try:
        if isinstance(enc, (AsymmetricDenseMixin, AsymmetricSparseMixin)):
            return await (
                enc.aencode_queries(texts)
                if kind == "queries"
                else enc.aencode_documents(texts)
            )
        return await enc.acall(texts)
    except NotImplementedError:
        # no async path on this encoder: run the sync one in a thread so the loop stays free
        return await asyncio.to_thread(_encode, enc, texts, kind)


def _record_metadata(route: Route) -> dict:
    # route-level fields ride on every record, so the index alone can rebuild the route
    meta = dict(route.metadata)
    if route.description is not None:
        meta["sr_description"] = route.description
    if route.confidence_threshold is not None:
        meta["sr_confidence_threshold"] = route.confidence_threshold
    return meta


def _merge(stored: list[Route], new: list[Route]) -> list[Route]:
    """the routes the index holds once `new` is upserted onto `stored`: a new route
    replaces a stored one's fields, and their utterances add up."""
    merged = {r.name: r for r in stored}
    for r in new:
        old = merged.get(r.name)
        if old is not None:
            extra = [u for u in r.utterances if u not in old.utterances]
            r = r.model_copy(update={"utterances": list(old.utterances) + extra})
        merged[r.name] = r
    return list(merged.values())


class IndexedRouter(BaseRouter):
    def __init__(
        self,
        encoder: DenseEncoder,
        routes: Sequence[Route] = (),
        index: BaseIndex | None = None,
        sparse_encoder: SparseEncoder | None = None,
        alpha: float = 0.3,
        top_k: int = 5,
        aggregation: Aggregation = "mean",
        confidence_threshold: float | None = None,
        calibrator: Calibrator | None = None,
    ):
        super().__init__(routes, confidence_threshold, calibrator)
        self.encoder = encoder
        self.sparse_encoder = sparse_encoder
        self.alpha = alpha
        self.top_k = top_k
        self.aggregate = _AGG[aggregation]
        # `is None`, not `or`: an empty index has len 0, so `or` would swap it for a local one
        if index is None:
            index = HybridLocalIndex() if sparse_encoder else LocalIndex()
        self.index = index
        # build synchronously so a constructed router is ready to call; the given routes
        # are upserted onto whatever the index already holds
        if isinstance(self.sparse_encoder, FittableMixin) and (
            corpus := _merge(self.routes, list(routes))
        ):
            self.sparse_encoder.fit(corpus)
        self._index(list(routes))
        # filled from the index by each call, in _scores_many
        self._snapshot: dict[str, Route] = {}

    @property
    def routes(self) -> list[Route]:
        # the index is the source of truth, so read it on every access: routes another
        # router wrote to a shared index show up here too
        if not hasattr(self, "index"):
            return []  # BaseRouter.__init__ runs before the index is set
        return self.index.get_routes()

    @routes.setter
    def routes(self, routes: list[Route]) -> None:
        # BaseRouter assigns routes on init and remove; an indexed router writes them to
        # the index (_index, _on_add, _on_remove), so there is nothing to keep here
        pass

    def _bar(self, name: str) -> float | None:
        # same rule as BaseRouter._bar, read from this call's snapshot of the index
        # rather than re-reading the index for every ranked route
        route = self._snapshot.get(name)
        if route is not None and route.confidence_threshold is not None:
            return route.confidence_threshold
        return self.confidence_threshold

    def _index(self, routes: list[Route]) -> None:
        names = [r.name for r in routes for _ in r.utterances]
        # encoders get the utterances as given (images for multimodal routes), the index text
        items = [u for r in routes for u in r.utterances]
        if not items:
            return
        sparse = (
            _encode(self.sparse_encoder, items, "documents")
            if self.sparse_encoder
            else None
        )
        self.index.add(
            embeddings=_encode(self.encoder, items, "documents"),
            routes=names,
            utterances=[str(u) for u in items],
            metadata_list=[_record_metadata(r) for r in routes for _ in r.utterances],
            sparse_embeddings=sparse,
        )

    async def _on_add(self, routes: list[Route]) -> None:
        if isinstance(self.sparse_encoder, FittableMixin):
            # a fitted sparse model changes its vocabulary with the corpus, which makes every
            # stored sparse vector stale; the only correct move is to refit and rebuild.
            # read the routes before deleting the index, which is where they live
            routes = _merge(self.routes, routes)
            self.sparse_encoder.fit(routes)
            self.index.delete_index()
        await asyncio.to_thread(self._index, routes)

    async def _on_remove(self, name: str) -> None:
        await self.index.adelete(name)

    async def _query(
        self, dense: list[float], sparse: SparseEmbedding | None
    ) -> dict[str, float]:
        vec = np.array(dense)
        if sparse is not None:
            # convex mix: alpha on dense, the rest on sparse, summed inside the hybrid index
            vec = vec * self.alpha
            sparse = SparseEmbedding.from_dict(
                {k: v * (1 - self.alpha) for k, v in sparse.to_dict().items()}
            )
        scores, names = await self.index.aquery(
            vector=vec, top_k=self.top_k, sparse_vector=sparse
        )
        hits: dict[str, list[float]] = {name: [] for name in self._snapshot}
        for n, s in zip(names, scores):
            hits.setdefault(n, []).append(float(s))
        return {n: self.aggregate(v) if v else 0.0 for n, v in hits.items()}

    async def afit(
        self, X: Sequence[str], y: Sequence[str | None], method: Method = "platt"
    ) -> FitReport:
        stored = {str(u) for r in self.routes for u in r.utterances}
        overlap = sum(q in stored for q in X)
        if overlap:
            # a stored utterance matches itself at ~1.0, so fitting on it teaches the
            # calibrator that real queries (which never score that high) are unsure
            logger.warning(
                f"{overlap}/{len(X)} training queries are stored utterances; they match "
                "themselves at ~1.0, so the calibrator will under-rate real queries, which "
                "score lower. fit on held-out queries."
            )
        return await super().afit(X, y, method)

    async def _scores(self, query: str) -> dict[str, float]:
        return (await self._scores_many([query]))[0]

    async def _scores_many(self, queries: Sequence[str]) -> list[dict[str, float]]:
        # one encoder call for the whole batch, then a cheap index query per vector
        texts = list(queries)
        # read the routes once per batch, alongside the embedding call: this call scores
        # every route in the index, and _bar reads each route's threshold from them
        dense, routes = await asyncio.gather(
            _aencode(self.encoder, texts, "queries"), self.index.aget_routes()
        )
        self._snapshot = {r.name: r for r in routes}
        sparse = (
            await _aencode(self.sparse_encoder, texts, "queries")
            if self.sparse_encoder
            else [None] * len(texts)
        )
        return list(
            await asyncio.gather(*(self._query(d, s) for d, s in zip(dense, sparse)))
        )
