"""compose routers: fall through a chain, or merge several opinions into one."""

from __future__ import annotations

import asyncio
from typing import Sequence

from semantic_router.route import Route
from semantic_router.routers.base import BaseRouter


class Chain(BaseRouter):
    """try routers in order; the first whose best confidence clears its bar answers."""

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

    async def _scores(self, query: str) -> dict[str, float]:
        for r in self.routers:
            raw = await r._scores(query)
            probs = r._probs(raw)
            # the top route by raw score: a calibrator never reorders
            top = max(raw, key=raw.__getitem__)
            # the top route's own bar, else this router's, else the chain's
            bar = r._bar(top)
            if bar is None:
                bar = self.confidence_threshold
            if r is self.routers[-1] or bar is None or probs[top] >= bar:
                # in raw-score order, so routes the calibrator ties keep their ranking
                return {
                    n: probs[n] for n in sorted(raw, key=raw.__getitem__, reverse=True)
                }
        raise AssertionError("unreachable")


class Merge(BaseRouter):
    """ask every router at once and take a weighted mean of their confidences per route."""

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

    async def _scores(self, query: str) -> dict[str, float]:
        results = await asyncio.gather(*(r._scores(query) for r in self.routers))
        probs = [r._probs(s) for r, s in zip(self.routers, results)]
        names = {n for p in probs for n in p}
        total = sum(self.weights)
        return {
            n: sum(w * p.get(n, 0.0) for w, p in zip(self.weights, probs)) / total
            for n in names
        }
