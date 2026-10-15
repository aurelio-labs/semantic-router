"""a router that asks a system one model which route applies. one call answers many questions."""

from __future__ import annotations

import asyncio
import os
import random
from typing import Sequence

import aiohttp

from semantic_router.calibration import Calibrator
from semantic_router.route import Route
from semantic_router.routers.base import BaseRouter, run

Question = dict[str, object]
Answer = dict[str, object]


def choice(instructions: str, options: dict[str, str]) -> Question:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def noul(instructions: str, yes: str = "yes", no: str = "no") -> Question:
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {"true": yes, "false": no},
    }


def score(instructions: str, levels: Sequence[str]) -> Question:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


class _Retry(Exception):
    pass


class DecisionRouter(BaseRouter):
    probabilistic = True  # the model returns a probability per route, not a similarity

    def __init__(
        self,
        routes: Sequence[Route],
        api_key: str | None = None,
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai",
        instructions: str = "Which route does the user's message belong to?",
        timeout: float = 10.0,
        retries: int = 3,
        concurrency: int = 8,
        confidence_threshold: float | None = None,
        calibrator: Calibrator | None = None,
        session: aiohttp.ClientSession | None = None,
    ):
        super().__init__(routes, confidence_threshold, calibrator)
        self.api_key = api_key or os.environ.get("JEV_API_KEY", "")
        if not self.api_key:
            raise ValueError("pass api_key or set JEV_API_KEY")
        self.model, self.url, self.instructions = (
            model,
            f"{base_url.rstrip('/')}/v1/systemone",
            instructions,
        )
        self.timeout, self.retries = timeout, retries
        # requests in flight when scoring a batch: with one call per query, this caps a
        # batch's throughput, so raise it as far as your rate limit allows
        self.concurrency = concurrency
        self._session = session
        self._loop: asyncio.AbstractEventLoop | None = None

    def route_question(self) -> Question:
        """the choice question for the current route set; descriptions beat utterance samples"""
        return choice(
            self.instructions,
            {
                r.name: r.description or "; ".join(str(u) for u in r.utterances[:5])
                for r in self.routes
            },
        )

    async def aask(
        self, state: object, questions: dict[str, Question]
    ) -> dict[str, Answer]:
        """answer several typed questions about one state in a single model pass."""
        loop = asyncio.get_running_loop()
        if (
            self._session is None
            or self._session.closed
            or self._loop not in (None, loop)
        ):
            # one pooled session per loop so repeated calls reuse the tls connection
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        self._loop = loop
        body = {"state": state, "model": self.model, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.retries + 1):
            try:
                async with self._session.post(
                    self.url, json=body, headers=headers
                ) as r:
                    if r.status in (429, 529) or r.status >= 500:
                        raise _Retry(r.status)
                    if r.status != 200:
                        raise RuntimeError(f"systemone {r.status}: {await r.text()}")
                    return (await r.json())["answers"]
            except (_Retry, asyncio.TimeoutError, aiohttp.ClientError):
                if attempt == self.retries:
                    raise
                # exponential backoff with jitter; the api asks for exactly this on 429/529
                await asyncio.sleep(0.25 * 2**attempt * (1 + random.random()))
        raise AssertionError("unreachable")

    async def _scores(self, query: str) -> dict[str, float]:
        answer = (await self.aask(query, {"route": self.route_question()}))["route"]
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            raise RuntimeError(f"systemone answer has no probabilities: {answer}")
        return {r.name: float(probs.get(r.name, 0.0)) for r in self.routes}

    async def aclose(self) -> None:
        if self._session is None or self._session.closed or self._loop is None:
            return
        if self._loop is asyncio.get_running_loop():
            await self._session.close()
        elif self._loop.is_closed() or not self._loop.is_running():
            # the loop that owned it is gone or idle, so nothing can run a graceful close
            # there from here: drop it and let the sockets go with garbage collection
            pass
        else:
            # the session lives on another live loop (the shared sync loop); close it there
            done = asyncio.run_coroutine_threadsafe(self._session.close(), self._loop)
            await asyncio.wrap_future(done)
        self._session = None

    def close(self) -> None:
        loop = self._loop
        if self._session is None or self._session.closed or loop is None:
            return
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is running:
            # blocking here would stop the very loop that has to run the close
            raise RuntimeError(
                "this router's session belongs to the running event loop: "
                "use `await router.aclose()` here"
            )
        if running is None and not loop.is_running() and not loop.is_closed():
            # its loop is idle (a run_until_complete that returned), so close it there
            loop.run_until_complete(self._session.close())
            self._session = None
            return
        run(self.aclose())
