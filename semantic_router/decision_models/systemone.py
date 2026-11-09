"""decision models served over HTTP at /v1/systemone (TypeSafe's Jev, and servers that copy it)."""

from __future__ import annotations

import asyncio
import os
import random
import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar, Sequence
from urllib.parse import urlsplit

import aiohttp
from pydantic import Field, PrivateAttr

from semantic_router.decision_models.base import Answer, DecisionModel, Question


class _Retry(Exception):
    pass


def _origin(url: str) -> str:
    """scheme://host[:port] of a url, lowercased, with the default port dropped."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:
        return ""
    if port == {"https": 443, "http": 80}.get(scheme):
        port = None
    host = (parts.hostname or "").lower()
    return f"{scheme}://{host}" + (f":{port}" if port else "")


@lru_cache(maxsize=1)
def _nace_client() -> str | None:
    """`semantic-router/<version>`, or None for a checkout that was never installed."""
    try:
        v = version("semantic-router")
    except PackageNotFoundError:
        return None
    return f"semantic-router/{re.sub(r'[^A-Za-z0-9 ._@+-]', '-', v)}"[:64]


class SystemOneDecisionModel(DecisionModel):
    """any server that speaks /v1/systemone: Jev, Laya, Von, or one of your own.

    :param name: The model id sent with every request.
    :param base_url: The server; requests go to `{base_url}/v1/systemone`.
    :param api_key: Sent as a Bearer token. Read from `api_key_env` when not given;
        a server that needs no key gets no Authorization header.
    :param timeout: Seconds per request.
    :param retries: Retries on rate limits (429, 529), server errors and timeouts.
    :param concurrency: Requests in flight when answering many states at once. With
        one request per state, this caps a batch's throughput.
    :param session: An aiohttp session to use instead of the model's own (tests).
    """

    type: str = "systemone"
    base_url: str
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    timeout: float = 10.0
    retries: int = 3
    concurrency: int = 8
    session: Any = Field(default=None, exclude=True, repr=False)

    # where to find the key when none is passed, and whether one is required
    api_key_env: ClassVar[str | None] = None
    requires_key: ClassVar[bool] = False
    # when set, the key is read from `api_key_env` only if base_url has this origin, so
    # an environment key is never sent to another host. A key passed in is always used.
    env_key_origin: ClassVar[str | None] = None

    _session: aiohttp.ClientSession | None = PrivateAttr(default=None)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)
    # the key as read from the environment, so a later base_url change can't carry it away
    _env_key: str | None = PrivateAttr(default=None)

    def _check_key_destination(self) -> None:
        if (
            self._env_key is not None
            and self.api_key == self._env_key
            and self.env_key_origin is not None
            and _origin(self.base_url) != self.env_key_origin
        ):
            raise ValueError(
                f"{self.api_key_env} is only used with {self.env_key_origin}; "
                f"pass api_key to use {self.base_url}"
            )

    def model_post_init(self, __context: Any) -> None:
        env_ok = self.env_key_origin is None or (
            _origin(self.base_url) == self.env_key_origin
        )
        if self.api_key is None and self.api_key_env and env_ok:
            self.api_key = self._env_key = os.environ.get(self.api_key_env) or None
        if self.api_key is None and self.requires_key:
            if not env_ok:
                raise ValueError(
                    f"{self.api_key_env} is only used with {self.env_key_origin}; "
                    f"pass api_key to use {self.base_url}"
                )
            raise ValueError(f"pass api_key or set {self.api_key_env}")
        self._session = self.session

    def _extra_headers(self) -> dict[str, str]:
        """headers sent on every request besides Authorization; none by default."""
        return {}

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/systemone"

    async def aask(
        self, state: object, questions: dict[str, Question]
    ) -> dict[str, Answer]:
        """answer several typed questions about one state in a single request."""
        self._check_key_destination()  # before any session or request exists
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
        body = {"state": state, "model": self.name, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        headers.update(self._extra_headers())
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

    async def aask_many(
        self, states: Sequence[object], questions: dict[str, Question]
    ) -> list[dict[str, Answer]]:
        # one request per state, at most `concurrency` in flight
        sem = asyncio.Semaphore(self.concurrency)

        async def one(state: object) -> dict[str, Answer]:
            async with sem:
                return await self.aask(state, questions)

        return list(await asyncio.gather(*(one(s) for s in states)))

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
        # imported here: the routers package imports this one
        from semantic_router.routers.base import run

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
                "this model's session belongs to the running event loop: "
                "use `await aclose()` here"
            )
        if running is None and not loop.is_running() and not loop.is_closed():
            # its loop is idle (a run_until_complete that returned), so close it there
            loop.run_until_complete(self._session.close())
            self._session = None
            return
        run(self.aclose())


class TypeSafeDecisionModel(SystemOneDecisionModel):
    """TypeSafe's Jev over its API. Reads `JEV_API_KEY` when no key is passed.

    :param name: The Jev model to call.
    """

    type: str = "typesafe"
    name: str = "jev-latest"
    base_url: str = "https://api.typesafe.ai"

    api_key_env: ClassVar[str | None] = "JEV_API_KEY"
    requires_key: ClassVar[bool] = True


class NaceDecisionModel(SystemOneDecisionModel):
    """nace.ai's Drex over its API. Reads `NACE_API_KEY` when no key is passed.

    The environment key is used only with the default `base_url`; a different host
    needs an explicit `api_key`, so the environment key never leaves nace.ai.

    :param name: The Drex model to call: `drex-max` (default), `drex-max-v1.1` or
        `drex-v2.0`. The Max models also read images, but this model sends the
        text `state` only.
    """

    type: str = "nace"
    name: str = "drex-max"
    base_url: str = "https://console.nace.ai"

    api_key_env: ClassVar[str | None] = "NACE_API_KEY"
    env_key_origin: ClassVar[str | None] = "https://console.nace.ai"
    requires_key: ClassVar[bool] = True

    def _extra_headers(self) -> dict[str, str]:
        # names this integration and its version for nace.ai's usage attribution
        client = _nace_client()
        return {"X-Nace-Client": client} if client else {}
