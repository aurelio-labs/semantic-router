"""compose routers: fall through a chain, or merge several opinions into one."""

from __future__ import annotations

import asyncio
import math
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

    async def _scores(self, query: str, return_utterances: bool = False) -> QueryResult:
        return (await self._scores_many([query], return_utterances=return_utterances))[
            0
        ]

    async def _scores_many(
        self, queries: Sequence[str], return_utterances: bool = False
    ) -> list[QueryResult]:
        # each router scores, in one batch, only the queries the routers before it
        # weren't sure of
        out: dict[int, dict[str, float]] = {}
        out_utts: dict[int, dict[str, list]] = {}
        pending = list(range(len(queries)))
        for r in self.routers:
            raw_qrs = await r._scores_many(
                [queries[i] for i in pending], return_utterances=return_utterances
            )
            raws = [qr.scores for qr in raw_qrs]
            unsure = []
            for i, raw, qr in zip(pending, raws, raw_qrs):
                if not raw:
                    out[i] = {}
                    unsure.append(i)
                    continue
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
                    # forward matched_utterances from the winning sub-router, if any
                    if qr.matched_utterances is not None:
                        out_utts[i] = dict(qr.matched_utterances)
                else:
                    unsure.append(i)
            pending = unsure
            if not pending:
                break
        results = []
        for i in range(len(queries)):
            scores = out.get(i, {})
            utts = out_utts.get(i)
            results.append(QueryResult(scores=scores, matched_utterances=utts))
        return results


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
        self.weights = [1.0] * len(self.routers) if weights is None else list(weights)
        self._shares()  # fail at construction, not at the first call
        super().__init__([], confidence_threshold)

    def _shares(self) -> list[float]:
        """each router's share of the mean, from the current `weights`."""
        weights = self.weights
        if len(weights) != len(self.routers):
            # zip would drop routers past a short list, and a long one only adds to the total
            raise ValueError(
                f"got {len(weights)} weights for {len(self.routers)} routers; "
                "give one weight per router"
            )
        if any(not (math.isfinite(w) and w >= 0) for w in weights) or not any(
            w > 0 for w in weights
        ):
            raise ValueError(
                f"weights must be finite, 0 or more, and not all 0, got {weights}"
            )
        # scaled by the largest first, so huge weights can't overflow the sum
        top = max(weights)
        scaled = [w / top for w in weights]
        total = sum(scaled)
        return [w / total for w in scaled]

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

    async def _scores(self, query: str, return_utterances: bool = False) -> QueryResult:
        return (await self._scores_many([query], return_utterances=return_utterances))[
            0
        ]

    async def _scores_many(
        self, queries: Sequence[str], return_utterances: bool = False
    ) -> list[QueryResult]:
        # every router scores the whole batch at once, so an indexed router makes one
        # embed call for it rather than one per query
        shares = self._shares()  # before any router runs, so bad weights cost nothing
        results = await asyncio.gather(
            *(
                r._scores_many(queries, return_utterances=return_utterances)
                for r in self.routers
            )
        )
        out = []
        for per_router in zip(*results):
            per_router_scores = [qr.scores for qr in per_router]
            probs = [r._probs(s) for r, s in zip(self.routers, per_router_scores)]
            names = dict.fromkeys(n for p in probs for n in p)
            scores_out = {
                n: sum(w * p.get(n, 0.0) for w, p in zip(shares, probs)) for n in names
            }
            # merge matched_utterances from all sub-routers that expose them
            merged_utts: dict[str, list] | None = None
            for qr in per_router:
                if qr.matched_utterances is not None:
                    if merged_utts is None:
                        merged_utts = {}
                    for route, utts in qr.matched_utterances.items():
                        merged_utts.setdefault(route, []).extend(utts)
            out.append(QueryResult(scores=scores_out, matched_utterances=merged_utts))
        return out
