import asyncio
from typing import Any, ClassVar

import pytest

from semantic_router.decision_models import DecisionModel, noul
from semantic_router.route import Route
from semantic_router.routers import DecisionRouter

ROUTES = [
    Route(name="billing", utterances=[], description="money already paid or owed"),
    Route(name="technical", utterances=["app crash", "error on save"]),
]


class FakeModel(DecisionModel):
    """answers from a {query: probabilities} table and records every call"""

    name: str = "fake"
    table: dict[str, dict[str, float]] = {}
    calls: list[Any] = []
    closed: bool = False

    async def aask(self, state, questions):
        self.calls.append(("aask", state, questions))
        out = {"route": {"type": "choice", "probabilities": self.table[state]}}
        for key in set(questions) - {"route"}:
            out[key] = {"type": "noul", "noul": 0.1}
        return out

    async def aask_many(self, states, questions):
        self.calls.append(("aask_many", list(states), questions))
        return [
            {"route": {"type": "choice", "probabilities": self.table[s]}}
            for s in states
        ]

    async def aclose(self):
        self.closed = True

    def close(self):
        self.closed = True


class RawScores(FakeModel):
    probabilistic: ClassVar[bool] = False


TABLE = {
    "charged twice": {"billing": 0.9, "technical": 0.1},
    "it crashed": {"billing": 0.2, "technical": 0.8},
    "hello": {"billing": 0.5, "technical": 0.5},
}


def test_scores_come_from_the_models_probabilities():
    r = DecisionRouter(model=FakeModel(table=TABLE), routes=ROUTES)
    pick = r("charged twice")
    assert pick.name == "billing" and pick.confidence == 0.9 and pick.calibrated


def test_route_question_prefers_descriptions_then_utterances():
    r = DecisionRouter(model=FakeModel(), routes=ROUTES)
    q = r.route_question()
    assert q["type"] == "choice"
    assert q["criteria"]["billing"] == "money already paid or owed"
    assert q["criteria"]["technical"] == "app crash; error on save"


def test_a_batch_is_one_call_into_the_model():
    model = FakeModel(table=TABLE)
    r = DecisionRouter(model=model, routes=ROUTES)
    acc = r.evaluate(X=["charged twice", "it crashed"], y=["billing", "technical"])
    assert acc == 1.0
    assert [c[0] for c in model.calls] == ["aask_many"]
    assert model.calls[0][1] == ["charged twice", "it crashed"]


def test_aask_passes_every_question_through():
    model = FakeModel(table=TABLE)
    r = DecisionRouter(model=model, routes=ROUTES)
    out = asyncio.run(
        r.aask("hello", {"route": r.route_question(), "guard": noul("hostile?")})
    )
    assert set(out) == {"route", "guard"}
    assert set(model.calls[0][2]) == {"route", "guard"}


def test_a_route_the_model_leaves_out_scores_zero():
    model = FakeModel(table={"q": {"billing": 1.0, "unknown": 0.4}})
    scored = asyncio.run(DecisionRouter(model=model, routes=ROUTES)._scores_many(["q"]))
    assert [qr.scores for qr in scored] == [{"billing": 1.0, "technical": 0.0}]


def test_an_answer_without_probabilities_is_an_error():
    class Broken(FakeModel):
        async def aask_many(self, states, questions):
            return [{"route": {"type": "choice", "choice": "billing"}} for _ in states]

    with pytest.raises(RuntimeError, match="no probabilities"):
        DecisionRouter(model=Broken(), routes=ROUTES)("q")


def test_a_model_without_probabilities_needs_a_fit():
    r = DecisionRouter(model=RawScores(table=TABLE), routes=ROUTES)
    assert not r.calibrated and not r("charged twice").calibrated


def test_close_reaches_the_model():
    model = FakeModel()
    r = DecisionRouter(model=model, routes=ROUTES)
    r.close()
    assert model.closed
    model.closed = False
    asyncio.run(r.aclose())
    assert model.closed


def test_added_routes_join_the_next_question():
    r = DecisionRouter(model=FakeModel(), routes=ROUTES)
    r.add(Route(name="account", description="logins and passwords"))
    assert set(r.route_question()["criteria"]) == {"billing", "technical", "account"}
