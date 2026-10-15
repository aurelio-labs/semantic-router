"""latency of every v2 path, measured against the real apis. prints a table and writes json.

usage: uv run python benchmarks/v2_latency.py [--n 20]
cost: ~n*8 jev calls at ~450 tokens and n*3 openai embeds; well under a cent at n=20.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import time
from pathlib import Path

from dotenv import load_dotenv

from semantic_router.encoders import OpenAIEncoder
from semantic_router.route import Route
from semantic_router.v2 import Chain, DecisionRouter, IndexedRouter, noul

load_dotenv(Path(__file__).parents[1] / ".env")

ROUTES = [
    Route(
        name="billing",
        description="Money already paid or owed: refunds, charges, invoices, subscription changes, payment methods.",
        utterances=[
            "i want a refund",
            "why was i charged twice",
            "cancel my subscription",
            "update my card",
            "invoice is wrong",
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
            "sync stopped working",
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
            "open a ticket for me",
        ],
    ),
]
QUERY = "why was i charged twice this month"
# held-out paraphrases for calibration: never the stored utterances, which self-match at ~1.0
HELD_OUT = [
    ("my card was declined again", "billing"),
    ("i was billed twice", "billing"),
    ("where is my refund", "billing"),
    ("stop my subscription", "billing"),
    ("the invoice amount is wrong", "billing"),
    ("it throws an exception on startup", "technical"),
    ("uploads keep failing", "technical"),
    ("i cannot log in", "technical"),
    ("the screen is blank", "technical"),
    ("saving gives an error", "technical"),
    ("get me a person on the phone", "support"),
    ("i want to talk to someone", "support"),
    ("can you call me", "support"),
    ("raise this with your team", "support"),
    ("i need help from a human", "support"),
]
NEGATIVES = [
    "what is the weather today",
    "tell me a joke",
    "who won the game",
    "recipe for pancakes",
    "translate this to french",
]
GUARD = noul("Is the user abusive or threatening?", "hostile or threatening", "civil")
URGENT = noul("Is this time-sensitive?", "needs action now", "can wait")


async def timed(n: int, fn) -> list[float]:
    out = []
    for _ in range(n):
        t = time.perf_counter()
        await fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def row(name: str, ms: list[float]) -> dict:
    s = sorted(ms)
    return {
        "name": name,
        "n": len(ms),
        "p50": st.median(s),
        "p95": s[int(0.95 * (len(s) - 1))],
        "min": s[0],
        "max": s[-1],
    }


async def main(n: int) -> None:
    enc = OpenAIEncoder()
    indexed = IndexedRouter(enc, ROUTES)
    decision = DecisionRouter(ROUTES)
    rows = []

    # local math only: pre-embed once so the index query is isolated from the network
    vec = (await enc.acall([QUERY]))[0]
    rows.append(
        row(
            "index query only (local, numpy)",
            await timed(n * 10, lambda: indexed._query(vec, None)),
        )
    )

    async def sync_in_thread():
        await asyncio.to_thread(indexed.__call__, QUERY)

    await decision.aask(
        QUERY, {"route": decision.route_question()}
    )  # warm the pooled connection
    rows.append(row("openai embed, 1 text", await timed(n, lambda: enc.acall([QUERY]))))
    rows.append(
        row("openai embed, 16 texts", await timed(n, lambda: enc.acall([QUERY] * 16)))
    )
    rows.append(
        row(
            "indexed router acall (embed + query)",
            await timed(n, lambda: indexed.acall(QUERY)),
        )
    )
    rows.append(
        row(
            "indexed router sync __call__ (asyncio.run overhead)",
            await timed(n, sync_in_thread),
        )
    )
    rows.append(
        row(
            "jev, 1 question",
            await timed(
                n, lambda: decision.aask(QUERY, {"route": decision.route_question()})
            ),
        )
    )
    rows.append(
        row(
            "jev, 3 questions in one call",
            await timed(
                n,
                lambda: decision.aask(
                    QUERY,
                    {
                        "route": decision.route_question(),
                        "guard": GUARD,
                        "urgent": URGENT,
                    },
                ),
            ),
        )
    )

    # the three shapes of the same turn: route + guard + urgent
    async def sequential():
        for q in (
            {"route": decision.route_question()},
            {"guard": GUARD},
            {"urgent": URGENT},
        ):
            await decision.aask(QUERY, q)

    async def fanout():
        await asyncio.gather(
            *(
                decision.aask(QUERY, q)
                for q in (
                    {"route": decision.route_question()},
                    {"guard": GUARD},
                    {"urgent": URGENT},
                )
            )
        )

    rows.append(row("3 concerns: sequential (3 calls)", await timed(n, sequential)))
    rows.append(row("3 concerns: fan-out (3 calls, gathered)", await timed(n, fanout)))
    rows.append(
        row(
            "3 concerns: batched (1 call)",
            await timed(
                n,
                lambda: decision.aask(
                    QUERY,
                    {
                        "route": decision.route_question(),
                        "guard": GUARD,
                        "urgent": URGENT,
                    },
                ),
            ),
        )
    )

    # calibrate first: a raw cosine never clears 0.85, a probability can, and that is the point
    fast = IndexedRouter(enc, ROUTES)
    X = [q for q, _ in HELD_OUT] + NEGATIVES
    y = [name for _, name in HELD_OUT] + [None] * len(NEGATIVES)
    report = await fast.afit(X, y)
    print(
        f"\ncalibrated indexed router: accuracy {report.accuracy:.2f}, ece {report.ece:.3f}, n={report.n}"
    )
    fast.confidence_threshold = 0.85
    chain = Chain(fast, decision)
    rows.append(
        row(
            "chain (calibrated): confident, stops at indexed",
            await timed(n, lambda: chain.acall(QUERY)),
        )
    )
    rows.append(
        row(
            "chain (calibrated): ambiguous, falls to decision",
            await timed(
                n, lambda: chain.acall("how do i export everything before i leave")
            ),
        )
    )

    await decision.aclose()
    print(f"\n{'path':58} {'p50':>8} {'p95':>8} {'min':>8} {'max':>8}")
    for r in rows:
        print(
            f"{r['name']:58} {r['p50']:8.1f} {r['p95']:8.1f} {r['min']:8.1f} {r['max']:8.1f}"
        )
    out = Path(__file__).parent / "results" / "v2_latency.json"
    out.write_text(json.dumps({"n": n, "rows": rows}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    asyncio.run(main(ap.parse_args().n))
