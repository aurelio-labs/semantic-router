import asyncio
import json
import re
from importlib.metadata import version

import pytest

from semantic_router.decision_models import (
    NaceDecisionModel,
    SystemOneDecisionModel,
    TypeSafeDecisionModel,
    choice,
    noul,
)
from semantic_router.route import Route
from semantic_router.routers import DecisionRouter

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
    """replays a scripted list of responses and records every request"""

    closed = False

    def __init__(self, *responses):
        self.responses, self.calls, self.headers = list(responses), [], []

    def post(self, url, json=None, headers=None):
        self.calls.append(json)
        self.headers.append(headers)
        return self.responses.pop(0)


def answer(probs):
    return FakeResponse(
        200, {"answers": {"route": {"type": "choice", "probabilities": probs}}}
    )


def jev(session, **kwargs):
    return TypeSafeDecisionModel(api_key="k", session=session, **kwargs)


def test_scores_come_from_probabilities():
    r = DecisionRouter(
        model=jev(FakeSession(answer({"billing": 0.9, "technical": 0.1}))),
        routes=ROUTES,
    )
    pick = r("why was i charged twice")
    assert pick.name == "billing" and pick.confidence == 0.9 and pick.calibrated


def test_request_carries_state_model_and_questions():
    s = FakeSession(answer({"billing": 1.0}))
    r = DecisionRouter(model=jev(s, name="jev-test"), routes=ROUTES)
    r("hello")
    assert s.calls[0]["state"] == "hello" and s.calls[0]["model"] == "jev-test"
    assert set(s.calls[0]["questions"]) == {"route"}
    assert s.headers[0] == {"Authorization": "Bearer k"}


def test_retries_on_429_then_succeeds():
    s = FakeSession(FakeResponse(429, {}), answer({"billing": 0.2, "technical": 0.8}))
    r = DecisionRouter(model=jev(s, retries=2), routes=ROUTES)
    assert r("it crashed").name == "technical"
    assert len(s.calls) == 2


def test_gives_up_after_retries():
    s = FakeSession(*[FakeResponse(529, {})] * 3)
    r = DecisionRouter(model=jev(s, retries=2), routes=ROUTES)
    with pytest.raises(Exception):
        r("anything")
    assert len(s.calls) == 3


def test_non_retryable_status_raises_immediately():
    s = FakeSession(FakeResponse(422, {"error": "bad"}))
    r = DecisionRouter(model=jev(s, retries=3), routes=ROUTES)
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
    r = DecisionRouter(model=jev(s), routes=ROUTES)
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


def test_url_and_defaults():
    m = TypeSafeDecisionModel(api_key="k")
    assert m.url == "https://api.typesafe.ai/v1/systemone" and m.name == "jev-latest"
    own = SystemOneDecisionModel(name="laya-1", base_url="http://127.0.0.1:8000/")
    assert own.url == "http://127.0.0.1:8000/v1/systemone"


def test_the_key_never_shows():
    m = TypeSafeDecisionModel(api_key="jev-SECRET")
    assert "jev-SECRET" not in repr(m)
    assert "jev-SECRET" not in json.dumps(m.model_dump(mode="json"))


def test_a_server_with_no_key_gets_no_auth_header():
    s = FakeSession(answer({"billing": 1.0}))
    own = SystemOneDecisionModel(name="local", base_url="http://x", session=s)
    asyncio.run(own.aask("hi", {"route": choice("pick", {"billing": "b", "x": "y"})}))
    assert s.headers[0] == {}


# --- nace.ai's Drex: the same protocol, with its own defaults and key ---


def drex(session, **kwargs):
    return NaceDecisionModel(api_key="k", session=session, **kwargs)


def test_nace_url_and_defaults():
    m = NaceDecisionModel(api_key="k")
    assert m.url == "https://console.nace.ai/v1/systemone"
    assert m.name == "drex-max" and m.type == "nace"


def test_nace_request_carries_the_model_the_key_and_text_only():
    s = FakeSession(answer({"billing": 0.9, "technical": 0.1}))
    r = DecisionRouter(model=drex(s), routes=ROUTES)
    assert r("why was i charged twice").name == "billing"
    assert s.calls[0]["model"] == "drex-max"
    assert set(s.calls[0]) == {"state", "model", "questions"}  # no images array
    assert s.headers[0] == {
        "Authorization": "Bearer k",
        "X-Nace-Client": f"semantic-router/{version('semantic-router')}",
    }


def test_nace_client_header_has_the_documented_format():
    s = FakeSession(answer({"billing": 0.9, "technical": 0.1}))
    DecisionRouter(model=drex(s), routes=ROUTES)("why was i charged twice")
    header = s.headers[0]["X-Nace-Client"]
    assert re.fullmatch(r"semantic-router/[A-Za-z0-9 ._@+-]+", header)
    assert len(header) <= 64


def test_other_systemone_models_do_not_send_the_nace_header():
    for model in (
        jev,
        lambda s: SystemOneDecisionModel(
            name="m", base_url="https://x.test", session=s
        ),
    ):
        s = FakeSession(answer({"billing": 0.9, "technical": 0.1}))
        DecisionRouter(model=model(s), routes=ROUTES)("why was i charged twice")
        assert "X-Nace-Client" not in s.headers[0]


def test_nace_reads_its_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "nace-env-SECRET")
    m = NaceDecisionModel()
    assert m.api_key == "nace-env-SECRET"
    assert "nace-env-SECRET" not in repr(m)
    assert "nace-env-SECRET" not in json.dumps(m.model_dump(mode="json"))


def test_nace_missing_api_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("NACE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NACE_API_KEY"):
        NaceDecisionModel()


def test_nace_env_key_is_not_read_for_a_foreign_host(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "nace-env-SECRET")
    with pytest.raises(ValueError, match="only used with https://console.nace.ai"):
        NaceDecisionModel(base_url="https://example.test")
    with pytest.raises(ValueError, match="only used with"):
        NaceDecisionModel(base_url="https://console.nace.ai.example.test")
    with pytest.raises(ValueError, match="only used with"):
        NaceDecisionModel(base_url="http://console.nace.ai")


@pytest.mark.parametrize(
    "url",
    [
        "https://console.nace.ai",
        "https://console.nace.ai/",
        "https://CONSOLE.nace.ai:443/x",
    ],
)
def test_nace_origin_uses_the_env_key(monkeypatch, url):
    monkeypatch.setenv("NACE_API_KEY", "nace-env-SECRET")
    assert NaceDecisionModel(base_url=url).api_key == "nace-env-SECRET"


def test_nace_explicit_key_with_a_custom_host_is_sent_there(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "nace-env-SECRET")
    s = FakeSession(answer({"billing": 0.9, "technical": 0.1}))
    m = NaceDecisionModel(base_url="https://proxy.test", api_key="mine", session=s)
    assert DecisionRouter(model=m, routes=ROUTES)("refund").name == "billing"
    assert s.headers[0] == {
        "Authorization": "Bearer mine",
        "X-Nace-Client": f"semantic-router/{version('semantic-router')}",
    }
    assert m.url == "https://proxy.test/v1/systemone"


def test_nace_env_key_is_refused_after_base_url_changes(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "nace-env-SECRET")
    s = FakeSession(answer({"billing": 0.9, "technical": 0.1}))
    m = NaceDecisionModel(session=s)
    m.base_url = "https://example.test"
    q = {"route": choice("pick", {"billing": "b", "x": "y"})}
    with pytest.raises(ValueError, match="only used with https://console.nace.ai"):
        asyncio.run(m.aask("hi", q))
    with pytest.raises(ValueError, match="only used with"):
        DecisionRouter(model=m, routes=ROUTES)("hi")  # the sync path
    assert s.calls == [] and s.headers == []


def test_nace_explicit_key_survives_a_base_url_change(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "nace-env-SECRET")
    s = FakeSession(answer({"billing": 0.9, "technical": 0.1}))
    m = NaceDecisionModel(api_key="mine", session=s)
    m.base_url = "https://proxy.test"
    assert DecisionRouter(model=m, routes=ROUTES)("refund").name == "billing"
    assert s.headers[0] == {
        "Authorization": "Bearer mine",
        "X-Nace-Client": f"semantic-router/{version('semantic-router')}",
    }


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


def local_jev(url):
    return TypeSafeDecisionModel(api_key="k", base_url=url, retries=1)


def test_sync_calls_reuse_one_session_across_calls(server):
    url, hits = server
    model = local_jev(url)
    r = DecisionRouter(model=model, routes=ROUTES)
    # the bug this pins: a second sync call used to die with "event loop is closed"
    assert r("first").name == "billing"
    assert r("second").name == "billing"
    assert r("third").name == "billing"
    assert hits["n"] == 4  # one 429 + three answers, all over the same pooled session
    assert model._session is not None and not model._session.closed


def test_close_after_async_calls_on_an_idle_loop_returns(server):
    import concurrent.futures

    url, _ = server
    model = local_jev(url)
    r = DecisionRouter(model=model, routes=ROUTES)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(r.acall("hello there"))  # the session now belongs to `loop`
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        # this used to hang forever: the close was scheduled on a loop nothing was running
        pool.submit(r.close).result(timeout=5)
    assert model._session is None
    loop.close()


def test_sync_close_on_the_owning_loop_says_to_await(server):
    url, _ = server
    model = local_jev(url)
    r = DecisionRouter(model=model, routes=ROUTES)

    async def inside():
        await r.acall("hello there")
        with pytest.raises(RuntimeError, match="aclose"):
            r.close()  # blocking would stop the loop that has to run the close
        await r.aclose()

    asyncio.run(inside())
    assert model._session is None


def test_missing_api_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(ValueError, match="JEV_API_KEY"):
        TypeSafeDecisionModel()


def test_concurrency_caps_requests_in_flight_for_a_batch(monkeypatch):
    live, peak = 0, 0

    async def fake_ask(self, state, questions):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return {"route": {"probabilities": {"billing": 0.9}}}

    # on the class: a pydantic model refuses new attributes on an instance
    monkeypatch.setattr(TypeSafeDecisionModel, "aask", fake_ask)
    r = DecisionRouter(
        model=TypeSafeDecisionModel(api_key="k", concurrency=3), routes=ROUTES
    )
    scored = asyncio.run(r._scores_many([f"q{i}" for i in range(12)]))
    assert len(scored) == 12 and peak == 3
