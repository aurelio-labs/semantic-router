"""the routing contract. a router scores every route for a query; everything else is shared."""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Coroutine, Sequence, TypeVar

import numpy as np

from semantic_router.calibration import Calibrator, Method, ece
from semantic_router.route import Route
from semantic_router.schema import RouteChoice
from semantic_router.utils.logger import logger

T = TypeVar("T")


_loop: asyncio.AbstractEventLoop | None = None


def run(coro: Coroutine[object, object, T]) -> T:
    """run a coroutine from sync code, including from inside a running loop (notebooks).

    every sync call shares one background loop, so pooled connections survive between calls
    instead of being torn down with a fresh `asyncio.run` each time.
    """
    global _loop
    try:
        running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if _loop is not None and running is _loop:
        coro.close()
        raise RuntimeError(
            "use `await` here: a sync call from inside the router loop would deadlock"
        )
    if _loop is None:
        _loop = asyncio.new_event_loop()
        threading.Thread(
            target=_loop.run_forever, name="semantic-router", daemon=True
        ).start()
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def in_sync_call() -> bool:
    """true while a sync method's coroutine runs on the shared background loop.

    routers use their clients' sync APIs there: an async client used on that loop binds
    to it, and then fails when the caller later awaits on a loop of their own.
    """
    try:
        return asyncio.get_running_loop() is _loop
    except RuntimeError:
        return False


@dataclass(frozen=True)
class FitReport:
    method: str
    n: int
    accuracy: float  # top-1 accuracy on queries that have a route, before any threshold
    ece: float  # calibration error of the stated confidences


class BaseRouter(ABC):
    # true when raw scores are already probabilities (system one models)
    probabilistic = False
    concurrency = 8  # cap on in-flight requests when scoring many queries

    def __init__(
        self,
        routes: Sequence[Route],
        confidence_threshold: float | None = None,
        calibrator: Calibrator | None = None,
    ):
        self.routes = list(routes)
        self.confidence_threshold = confidence_threshold
        self.calibrator = calibrator
        # the labelled queries behind the calibrator, kept so a route change can refit it
        self._fit_data: tuple[list[str], list[str | None], Method] | None = None

    @abstractmethod
    async def _scores(self, query: str) -> dict[str, float]:
        """raw score for every route, higher means more likely."""

    async def _scores_many(self, queries: Sequence[str]) -> list[dict[str, float]]:
        # default fans out with a concurrency cap; indexed routers override to batch-embed
        sem = asyncio.Semaphore(self.concurrency)

        async def one(q: str) -> dict[str, float]:
            async with sem:
                return await self._scores(q)

        return list(await asyncio.gather(*(one(q) for q in queries)))

    @property
    def calibrated(self) -> bool:
        return self.calibrator is not None or self.probabilistic

    def _probs(self, scores: dict[str, float]) -> dict[str, float]:
        # confidence per route: calibrated if we can, raw otherwise
        if self.calibrator is None:
            return scores
        names = list(scores)
        p = self.calibrator(np.array([scores[n] for n in names]))
        return dict(zip(names, map(float, p)))

    def _bar(self, name: str) -> float | None:
        """the confidence a route needs: its own threshold, else the router's."""
        for r in self.routes:
            if r.name == name and r.confidence_threshold is not None:
                return r.confidence_threshold
        return self.confidence_threshold

    def _choose(
        self, scores: dict[str, float], limit: int | None, threshold: float | None
    ) -> RouteChoice | list[RouteChoice]:
        probs = self._probs(scores)
        # rank by the raw score: calibration says how sure, it must never reorder routes
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        picks = []
        for n in ranked:
            c = probs[n]
            # a per-call threshold wins over the route's and the router's
            bar = self._bar(n) if threshold is None else threshold
            if bar is None or c >= bar:
                picks.append(
                    RouteChoice(name=n, confidence=c, calibrated=self.calibrated)
                )
        picks = picks[:limit]
        if limit == 1:
            # an empty pick still says whether the bar it missed was a calibrated one
            return picks[0] if picks else RouteChoice(calibrated=self.calibrated)
        return picks

    async def acall(
        self, query: str, limit: int | None = 1, threshold: float | None = None
    ) -> RouteChoice | list[RouteChoice]:
        return self._choose(await self._scores(query), limit, threshold)

    def __call__(
        self, query: str, limit: int | None = 1, threshold: float | None = None
    ) -> RouteChoice | list[RouteChoice]:
        return run(self.acall(query, limit, threshold))

    async def afit(
        self, X: Sequence[str], y: Sequence[str | None], method: Method = "platt"
    ) -> FitReport:
        """calibrate on labelled queries. a label of None means no route should match."""
        scored = await self._scores_many(X)
        top = [max(s, key=lambda k: s[k]) for s in scored]
        raw = np.array([s[t] for s, t in zip(scored, top)])
        correct = np.array([t == truth for t, truth in zip(top, y)], dtype=float)
        self.calibrator = Calibrator.fit(raw, correct, method)
        self._fit_data = (list(X), list(y), method)
        # accuracy over queries that have a route; a None label can only ever be a miss here
        labelled = np.array([truth is not None for truth in y])
        acc = float(correct[labelled].mean()) if labelled.any() else float("nan")
        return FitReport(method, len(X), acc, ece(self.calibrator(raw), correct))

    def fit(
        self, X: Sequence[str], y: Sequence[str | None], method: Method = "platt"
    ) -> FitReport:
        return run(self.afit(X, y, method))

    async def aevaluate(self, X: Sequence[str], y: Sequence[str | None]) -> float:
        """end-to-end accuracy with the threshold applied; None labels are right when nothing passes."""
        picks = [self._choose(s, 1, None) for s in await self._scores_many(X)]
        return float(np.mean([getattr(p, "name", None) == t for p, t in zip(picks, y)]))

    def evaluate(self, X: Sequence[str], y: Sequence[str | None]) -> float:
        return run(self.aevaluate(X, y))

    async def aadd(self, routes: Route | Sequence[Route]) -> None:
        new = [routes] if isinstance(routes, Route) else list(routes)
        self.routes.extend(new)
        await self._on_add(new)
        await self._refit()

    def add(self, routes: Route | Sequence[Route]) -> None:
        run(self.aadd(routes))

    async def aremove(self, name: str) -> None:
        self.routes = [r for r in self.routes if r.name != name]
        await self._on_remove(name)
        await self._refit()

    def remove(self, name: str) -> None:
        run(self.aremove(name))

    async def _refit(self) -> None:
        """the route set changed, so the calibrator no longer describes it: refit it or drop it.

        refitting re-scores the stored training queries: one batched embed for an indexed
        router, one call per query for a decision router.
        """
        if self.calibrator is None:
            return
        if self._fit_data is None:
            # passed in or loaded, so there are no labelled queries here to refit on
            logger.warning(
                "routes changed and there are no labelled queries to refit the calibrator on; "
                "it was dropped, so confidences are raw until you call fit() again"
            )
            self.calibrator = None
            return
        X, y, method = self._fit_data
        names = {r.name for r in self.routes}
        # a label for a route that is gone now means no route should match
        y = [t if t in names else None for t in y]
        try:
            await self.afit(X, y, method)
        except ValueError as e:
            logger.warning(
                f"could not refit the calibrator after a route change ({e}); it was dropped"
            )
            self.calibrator = None
            return
        if unlabelled := sorted(names - set(y)):
            logger.warning(
                f"refitted the calibrator, but {unlabelled} have no labelled queries; "
                "fit() again with some so their confidence is honest"
            )

    def to_yaml(self, path: str | Path) -> None:
        """save the router to a yaml file; `load_router` reads it back. keys are never written."""
        from semantic_router.config import save_router

        save_router(router=self, path=path, format="yaml")

    def to_json(self, path: str | Path) -> None:
        """save the router to a json file; `load_router` reads it back. keys are never written."""
        from semantic_router.config import save_router

        save_router(router=self, path=path, format="json")

    async def _on_add(self, routes: list[Route]) -> None:
        """hook for routers that keep state per route (an index)."""

    async def _on_remove(self, name: str) -> None:
        """hook for routers that keep state per route (an index)."""
