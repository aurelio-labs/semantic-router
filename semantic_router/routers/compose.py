"""compose routers: fall through a chain, or merge several opinions into one."""

from __future__ import annotations

import asyncio
from typing import Sequence

from semantic_router.routers.base import BaseRouter


class Chain(BaseRouter):
    """try routers in order; the first whose best confidence clears its bar answers."""

    def __init__(
        self,
        routers: Sequence[BaseRouter],
        confidence_threshold: float | None = None,
    ):
        super().__init__(routers[0].routes, confidence_threshold)
        self.routers = list(routers)

    @property
    def calibrated(self) -> bool:
        return all(r.calibrated for r in self.routers)

    async def _scores(self, query: str) -> dict[str, float]:
        for r in self.routers:
            probs = r._probs(await r._scores(query))
            top = max(probs, key=probs.__getitem__)
            # the top route's own bar, else this router's, else the chain's
            bar = r._bar(top)
            if bar is None:
                bar = self.confidence_threshold
            if r is self.routers[-1] or bar is None or probs[top] >= bar:
                return probs
        raise AssertionError("unreachable")


class Merge(BaseRouter):
    """ask every router at once and take a weighted mean of their confidences per route."""

    def __init__(
        self,
        routers: Sequence[BaseRouter],
        weights: list[float] | None = None,
        confidence_threshold: float | None = None,
    ):
        super().__init__(routers[0].routes, confidence_threshold)
        self.routers = list(routers)
        self.weights = weights or [1.0] * len(routers)

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
