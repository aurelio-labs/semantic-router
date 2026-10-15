import asyncio
import json

import pytest

from semantic_router.route import Route
from semantic_router.routers import DecisionRouter, choice, noul

ROUTES = [
    Route(name="billing", utterances=[], description="money already paid or owed"),
    Route(name="technical", utterances=["app crash", "error on save"]),
]


class FakeResponse:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def json(self):
        return self._body

    async def text(self):
        return json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeSession:
    """replays a scripted list of responses and records every request body"""

    closed = False

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json=None, headers=None):
        self.calls.append(json)
        return self.responses.pop(0)


def answer(probs):
    return FakeResponse(
        200, {"answers": {"route": {"type": "choice", "probabilities": probs}}}
    )


def test_scores_come_from_probabilities():
    r = DecisionRouter(
        ROUTES,
        api_key="k",
        session=FakeSession(answer({"billing": 0.9, "technical": 0.1})),
    )
    pick = r("why was i charged twice")
    assert pick.name == "billing" and pick.confidence == 0.9 and pick.calibrated


def test_route_question_prefers_descriptions_then_utterances():
    r = DecisionRouter(ROUTES, api_key="k", session=FakeSession())
    crit = r.route_question()["criteria"]
    assert crit["billing"] == "money already paid or owed"
    assert crit["technical"] == "app crash; error on save"


def test_retries_on_429_then_succeeds():
    s = FakeSession(FakeResponse(429, {}), answer({"billing": 0.2, "technical": 0.8}))
    r = DecisionRouter(ROUTES, api_key="k", session=s, retries=2)
    assert r("it crashed").name == "technical"
    assert len(s.calls) == 2


def test_gives_up_after_retries():
    s = FakeSession(*[FakeResponse(529, {})] * 3)
    r = DecisionRouter(ROUTES, api_key="k", session=s, retries=2)
    with pytest.raises(Exception):
        r("anything")
    assert len(s.calls) == 3


def test_non_retryable_status_raises_immediately():
    s = FakeSession(FakeResponse(422, {"error": "bad"}))
    r = DecisionRouter(ROUTES, api_key="k", session=s, retries=3)
    with pytest.raises(RuntimeError, match="422"):
        r("anything")
    assert len(s.calls) == 1


def test_aask_batches_many_questions_in_one_call():
    body = {
        "answers": {
            "route": {"type": "choice", "probabilities": {"billing": 1.0}},
            "guard": {"type": "noul", "noul": 0.02},
        }
    }
    s = FakeSession(FakeResponse(200, body))
    r = DecisionRouter(ROUTES, api_key="k", session=s)
    out = asyncio.run(
        r.aask("hi", {"route": r.route_question(), "guard": noul("hostile?")})
    )
    assert set(out) == {"route", "guard"} and len(s.calls) == 1
    assert set(s.calls[0]["questions"]) == {"route", "guard"}


def test_question_helpers_match_the_wire_shape():
    assert choice("pick", {"a": "x"}) == {
        "type": "choice",
        "instructions": "pick",
        "criteria": {"a": "x"},
    }
    assert noul("yes?", "y", "n")["criteria"] == {"true": "y", "false": "n"}


# --- against a real local http server, so session reuse across sync calls is exercised ---


@pytest.fixture
def server(unused_tcp_port):
    """a tiny systemone stand-in: first request 429s, the rest answer; counts hits"""
    from aiohttp import web

    hits = {"n": 0}

    async def handle(request):
        hits["n"] += 1
        if hits["n"] == 1:
            return web.Response(status=429)
        body = await request.json()
        names = list(body["questions"]["route"]["criteria"])
        return web.json_response(
            {"answers": {"route": {"type": "choice", "probabilities": {names[0]: 0.9}}}}
        )

    app = web.Application()
    app.router.add_post("/v1/systemone", handle)
    runner = web.AppRunner(app)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    loop.run_until_complete(site.start())
    import threading

    threading.Thread(target=loop.run_forever, daemon=True).start()
    yield f"http://127.0.0.1:{unused_tcp_port}", hits
    # shut the server down on its own loop so no handler task is left pending
    asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)


def test_sync_calls_reuse_one_session_across_calls(server):
    url, hits = server
    r = DecisionRouter(ROUTES, api_key="k", base_url=url, retries=1)
    # the bug this pins: a second sync call used to die with "event loop is closed"
    assert r("first").name == "billing"
    assert r("second").name == "billing"
    assert r("third").name == "billing"
    assert hits["n"] == 4  # one 429 + three answers, all over the same pooled session
    assert r._session is not None and not r._session.closed


def test_close_after_async_calls_on_an_idle_loop_returns(server):
    import concurrent.futures

    url, _ = server
    r = DecisionRouter(ROUTES, api_key="k", base_url=url, retries=1)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(r.acall("hello there"))  # the session now belongs to `loop`
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        # this used to hang forever: the close was scheduled on a loop nothing was running
        pool.submit(r.close).result(timeout=5)
    assert r._session is None
    loop.close()


def test_sync_close_on_the_owning_loop_says_to_await(server):
    url, _ = server
    r = DecisionRouter(ROUTES, api_key="k", base_url=url, retries=1)

    async def inside():
        await r.acall("hello there")
        with pytest.raises(RuntimeError, match="aclose"):
            r.close()  # blocking would stop the loop that has to run the close
        await r.aclose()

    asyncio.run(inside())
    assert r._session is None


def test_missing_api_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(ValueError, match="JEV_API_KEY"):
        DecisionRouter(ROUTES)


def test_concurrency_caps_requests_in_flight_for_a_batch():
    r = DecisionRouter(ROUTES, api_key="k", concurrency=3)
    live, peak = 0, 0

    async def fake_ask(state, questions):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return {"route": {"probabilities": {"billing": 0.9}}}

    r.aask = fake_ask  # type: ignore[method-assign]
    scored = asyncio.run(r._scores_many([f"q{i}" for i in range(12)]))
    assert len(scored) == 12 and peak == 3
