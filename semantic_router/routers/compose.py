"""compose routers: fall through a chain, or merge several opinions into one."""

from __future__ import annotations

import asyncio
from typing import Sequence

from semantic_router.route import Route
from semantic_router.routers.base import BaseRouter, QueryResult


class Chain(BaseRouter):
    """try routers in order; the first whose best confidence clears its bar answers.

    ``Chain`` routes by delegating to its sub-routers in sequence. Because it never
    runs an index query itself, ``matched_utterances`` on each ``RouteChoice`` is
    always ``None`` — utterance-level matches only surface on the
    ``IndexedRouter`` directly.
    """

    def __init__(
        self,
        routers: Sequence[BaseRouter],
        confidence_threshold: float | None = None,
    ):
        self.routers = list(routers)
        super().__init__([], confidence_threshold)

    @property
    def routes(self) -> list[Route]:
        # the first router's routes, read live, so later index changes show through
        return self.routers[0].routes

    @routes.setter
    def routes(self, value: list[Route]) -> None:
        pass  # a chain holds no routes of its own

    @property
    def calibrated(self) -> bool:
        return all(r.calibrated for r in self.routers)

    async def _scores(
        self, query: str, return_utterances: bool = False
    ) -> QueryResult:
        return (
            await self._scores_many([query], return_utterances=return_utterances)
        )[0]

    async def _scores_many(
        self, queries: Sequence[str], return_utterances: bool = False
    ) -> list[QueryResult]:
        # each router scores, in one batch, only the queries the routers before it
        # weren't sure of
        out: dict[int, dict[str, float]] = {}
        pending = list(range(len(queries)))
        for r in self.routers:
            raw_qrs = await r._scores_many([queries[i] for i in pending])
            raws = [qr.scores for qr in raw_qrs]
            unsure = []
            for i, raw in zip(pending, raws):
                probs = r._probs(raw)
                # the top route by raw score: a calibrator never reorders
                top = max(raw, key=raw.__getitem__)
                # the top route's own bar, else this router's, else the chain's
                bar = r._bar(top)
                if bar is None:
                    bar = self.confidence_threshold
                if r is self.routers[-1] or bar is None or probs[top] >= bar:
                    # in raw-score order, so routes the calibrator ties keep their order
                    order = sorted(raw, key=raw.__getitem__, reverse=True)
                    out[i] = {n: probs[n] for n in order}
                else:
                    unsure.append(i)
            pending = unsure
            if not pending:
                break
        return [QueryResult(scores=out[i]) for i in range(len(queries))]


class Merge(BaseRouter):
    """ask every router at once and take a weighted mean of their confidences per route.

    ``Merge`` routes by blending confidences from its sub-routers. Because it never
    runs an index query itself, ``matched_utterances`` on each ``RouteChoice`` is
    always ``None`` — utterance-level matches only surface on the
    ``IndexedRouter`` directly.
    """

    def __init__(
        self,
        routers: Sequence[BaseRouter],
        weights: list[float] | None = None,
        confidence_threshold: float | None = None,
    ):
        self.routers = list(routers)
        self.weights = weights or [1.0] * len(routers)
        super().__init__([], confidence_threshold)

    @property
    def routes(self) -> list[Route]:
        # the first router's routes, read live, so later index changes show through
        return self.routers[0].routes

    @routes.setter
    def routes(self, value: list[Route]) -> None:
        pass  # a merge holds no routes of its own

    @property
    def calibrated(self) -> bool:
        return all(r.calibrated for r in self.routers)

    async def _scores(
        self, query: str, return_utterances: bool = False
    ) -> QueryResult:
        return (
            await self._scores_many([query], return_utterances=return_utterances)
        )[0]

    async def _scores_many(
        self, queries: Sequence[str], return_utterances: bool = False
    ) -> list[QueryResult]:
        # every router scores the whole batch at once, so an indexed router makes one
        # embed call for it rather than one per query
        results = await asyncio.gather(*(r._scores_many(queries) for r in self.routers))
        total = sum(self.weights)
        out = []
        for per_router in zip(*results):
            per_router_scores = [qr.scores for qr in per_router]
            probs = [r._probs(s) for r, s in zip(self.routers, per_router_scores)]
            names = dict.fromkeys(n for p in probs for n in p)
            out.append(
                QueryResult(
                    scores={
                        n: sum(w * p.get(n, 0.0) for w, p in zip(self.weights, probs))
                        / total
                        for n in names
                    }
                )
            )
        return out
