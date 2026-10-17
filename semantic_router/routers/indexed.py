"""a router that scores by similarity between the query and stored example utterances."""

from __future__ import annotations

import asyncio
import time
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
from semantic_router.routers.base import BaseRouter, FitReport, in_sync_call
from semantic_router.schema import MatchedUtterance, SparseEmbedding
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
    if in_sync_call():
        # a sync call: use the encoder's sync client, so its async one never binds to
        # the shared loop
        return await asyncio.to_thread(_encode, enc, texts, kind)
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
        alpha: float = 0.7,
        top_k: int = 5,
        aggregation: Aggregation = "mean",
        confidence_threshold: float | None = None,
        calibrator: Calibrator | None = None,
        refresh_interval: float | None = None,
    ):
        # the routes as last read from the index, and when (time.monotonic)
        self._cache: dict[str, Route] = {}
        self._read_at = 0.0
        super().__init__(routes, confidence_threshold, calibrator)
        self.encoder = encoder
        self.sparse_encoder = sparse_encoder
        self.alpha = alpha
        self.top_k = top_k
        self.aggregate = _AGG[aggregation]
        # seconds a call trusts the cached routes before re-reading the index; None never
        # re-reads, so only refresh() and this router's own add / remove do
        self.refresh_interval = refresh_interval
        # `is None`, not `or`: an empty index has len 0, so `or` would swap it for a local one
        if index is None:
            index = HybridLocalIndex() if sparse_encoder else LocalIndex()
        self.index = index
        # build synchronously so a constructed router is ready to call; the given routes
        # are upserted onto whatever the index already holds
        if isinstance(self.sparse_encoder, FittableMixin) and (
            corpus := _merge(self.index.get_routes(), list(routes))
        ):
            self.sparse_encoder.fit(corpus)
        self._index(list(routes))
        self.refresh()

    @property
    def routes(self) -> list[Route]:
        """the routes in the index, as last read from it.

        the index is the only place routes are written, and this view is only ever filled
        from it: on build, after this router's add / remove, on refresh(), and per call
        once the cache is older than refresh_interval. a route another router wrote shows
        up here after a refresh, but a query finds it in the index straight away.
        """
        return list(self._cache.values())

    @routes.setter
    def routes(self, routes: list[Route]) -> None:
        # BaseRouter assigns routes on init and remove; an indexed router writes them to
        # the index (_index, _on_add, _on_remove) and reads them back, so there is
        # nothing to keep here
        pass

    def refresh(self) -> None:
        """re-read the routes from the index, picking up other writers' changes."""
        self._fill(self.index.get_routes())

    async def arefresh(self) -> None:
        """re-read the routes from the index, picking up other writers' changes."""
        self._fill(await self._aget_routes())

    async def _aget_routes(self) -> list[Route]:
        # a sync call uses the index's sync client (see in_sync_call)
        if in_sync_call():
            return await asyncio.to_thread(self.index.get_routes)
        return await self.index.aget_routes()

    def _fill(self, routes: list[Route]) -> None:
        # one assignment, so a concurrent call sees either the old routes or the new
        self._cache = {r.name: r for r in routes}
        self._read_at = time.monotonic()

    def _stale(self) -> bool:
        return (
            self.refresh_interval is not None
            and time.monotonic() - self._read_at >= self.refresh_interval
        )

    def _bar(self, name: str) -> float | None:
        # same rule as BaseRouter._bar, by name from the cache. a route that isn't cached
        # yet (another router added it) is held to the router's threshold until a refresh
        route = self._cache.get(name)
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
            # read every stored route fresh before deleting the index, which is where
            # they live
            routes = _merge(await self._aget_routes(), routes)
            self.sparse_encoder.fit(routes)
            self.index.delete_index()
        await asyncio.to_thread(self._index, routes)
        await self.arefresh()

    async def _on_remove(self, name: str) -> None:
        if in_sync_call():
            await asyncio.to_thread(self.index.delete, name)
        else:
            await self.index.adelete(name)
        await self.arefresh()

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
        if in_sync_call():
            result = await asyncio.to_thread(
                self.index.query,
                vector=vec,
                top_k=self.top_k,
                sparse_vector=sparse,
                return_utterances=True,
            )
        else:
            result = await self.index.aquery(
                vector=vec,
                top_k=self.top_k,
                sparse_vector=sparse,
                return_utterances=True,
            )
        # result is either (scores, names) or (scores, names, utterances) depending on
        # whether the index honours return_utterances; older implementations may ignore it
        if len(result) == 3:
            scores_arr, names, utterances = result
        else:
            scores_arr, names = result
            utterances = []
        scores = [float(s) for s in scores_arr]
        # every cached route scores, 0.0 if it isn't a hit; a hit on a route that isn't
        # cached yet still counts, because the search always runs against the index
        hits: dict[str, list[float]] = {name: [] for name in self._cache}
        matched_utts: dict[str, list[MatchedUtterance]] = {}
        for n, s, u in zip(names, scores, utterances):
            hits.setdefault(n, []).append(float(s))
            matched_utts.setdefault(n, []).append(
                MatchedUtterance(text=str(u), score=float(s), route=n)
            )
        # store per-route utterances for BaseRouter._choose to inject into RouteChoice
        self._last_utterances = matched_utts if utterances else None
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
        if self._stale():
            # re-read the routes alongside the embedding call, so it adds no latency
            dense, _ = await asyncio.gather(
                _aencode(self.encoder, texts, "queries"), self.arefresh()
            )
        else:
            dense = await _aencode(self.encoder, texts, "queries")
        sparse = (
            await _aencode(self.sparse_encoder, texts, "queries")
            if self.sparse_encoder
            else [None] * len(texts)
        )
        return list(
            await asyncio.gather(*(self._query(d, s) for d, s in zip(dense, sparse)))
        )
