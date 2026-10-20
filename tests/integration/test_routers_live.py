"""real api calls. skipped unless the keys are set; costs a fraction of a cent."""

import asyncio

import pytest

from semantic_router.decision_models import TypeSafeDecisionModel, noul
from semantic_router.encoders import OpenAIEncoder
from semantic_router.route import Route
from semantic_router.routers import Chain, DecisionRouter, IndexedRouter

ROUTES = [
    Route(
        name="billing",
        description="Money already paid or owed: refunds, charges, invoices, subscription changes, payment methods.",
        utterances=[
            "i want a refund",
            "why was i charged twice",
            "cancel my subscription",
            "update my card",
        ],
    ),
    Route(
        name="technical",
        description="The product is malfunctioning: errors, crashes, failures, things not working.",
        utterances=[
            "the app crashes on upload",
            "i get an error when saving",
            "login is broken",
            "page will not load",
        ],
    ),
    Route(
        name="support",
        description="A request for a person to help, call back, or take an action on the user's behalf.",
        utterances=[
            "can someone call me back",
            "i need to speak to a human",
            "please escalate this",
            "who can help me",
        ],
    ),
]
CASES = [
    ("my card was declined again", "billing"),
    ("it throws an exception on startup", "technical"),
    ("get me a person on the phone", "support"),
]


@pytest.mark.live("OPENAI_API_KEY")
def test_indexed_router_routes_and_calibrates():
    r = IndexedRouter(OpenAIEncoder(), ROUTES)
    for q, want in CASES:
        assert r(q).name == want
    X = [u for rt in ROUTES for u in rt.utterances] + [
        "what is the weather",
        "tell me a joke",
    ]
    y = [rt.name for rt in ROUTES for _ in rt.utterances] + [None, None]
    report = r.fit(X, y)
    assert r("i was billed twice").calibrated and report.accuracy > 0.8


@pytest.mark.live("JEV_API_KEY")
def test_decision_router_routes():
    r = DecisionRouter(model=TypeSafeDecisionModel(), routes=ROUTES)
    try:
        for q, want in CASES:
            pick = r(q)
            assert pick.name == want and pick.calibrated
    finally:
        r.close()


@pytest.mark.live("JEV_API_KEY")
def test_decision_router_batches_questions():
    r = DecisionRouter(model=TypeSafeDecisionModel(), routes=ROUTES)
    try:
        out = asyncio.run(
            r.aask(
                "HELP my payouts have failed for 3 days!!",
                {
                    "route": r.route_question(),
                    "urgent": noul(
                        "Is this time-sensitive?", "needs action now", "can wait"
                    ),
                },
            )
        )
        assert out["route"]["choice"] == "billing" and out["urgent"]["noul"] > 0.5
    finally:
        r.close()


@pytest.mark.live("OPENAI_API_KEY", "JEV_API_KEY")
def test_chain_semantic_then_decision():
    fast = IndexedRouter(OpenAIEncoder(), ROUTES, confidence_threshold=0.9)
    slow = DecisionRouter(model=TypeSafeDecisionModel(), routes=ROUTES)
    try:
        assert Chain(routers=[fast, slow])(
            "how do i export everything before i leave"
        ).name in {r.name for r in ROUTES}
    finally:
        slow.close()
