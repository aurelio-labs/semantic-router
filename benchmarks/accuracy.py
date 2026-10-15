"""accuracy, calibration and latency of the v1 routers on a fixed slice of CLINC150. writes json.

usage: uv run python benchmarks/accuracy.py [--seed 0] [--n 30] [--batch 50] [--latency all]
       --latency chain,merge re-times only those routers; the others keep their last timing.
       --latency none measures accuracy alone. the table shows the change since the last run.

data: CLINC150 (clinc/oos-eval, data_full.json, pinned to a commit). the seed picks 15 intents;
each becomes a route with 10 of its train utterances. val (plus out-of-scope val) fits every
calibrator, and test (plus out-of-scope test), held out from both, is what each number is
measured on. thresholds are not tuned: a router answers when its calibrated confidence >= 0.5.

cost: jev answers are cached on disk (benchmarks/.cache/), keyed on the whole request, so a
rerun with the same routes only spends jev calls on latency. the budget prints before any call
is made; --dry-run stops there. latency always hits the real apis, never the cache.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import random
import statistics as st
import subprocess
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from dotenv import load_dotenv

from semantic_router.calibration import ece
from semantic_router.encoders import FastEmbedEncoder, OpenAIEncoder, TfidfEncoder
from semantic_router.route import Route
from semantic_router.routers import (
    BaseRouter,
    Chain,
    DecisionRouter,
    IndexedRouter,
    Merge,
)

load_dotenv(Path(__file__).parents[1] / ".env")

HERE = Path(__file__).parent
CACHE = HERE / ".cache"
OUT = HERE / "results" / "accuracy.json"
DATA_URL = (
    "https://raw.githubusercontent.com/clinc/oos-eval/"
    "828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json"
)
DATA_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
N_INTENTS, N_UTTERANCES = 15, 10
N_OOS_FIT, N_OOS_TEST = 100, 150  # a quarter of each split is out of scope
BAR = 0.5  # every router answers when its calibrated P(correct) clears this
STOP_BAR = 0.85  # chain: the indexed stage answers on its own when this sure
STOP_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.01]
# hybrid: the weight on dense, the rest on sparse. 0.0 is sparse alone, and 1.0 must score
# exactly like dense alone
ALPHA_SWEEP = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
WARMUP = 2
ROUTERS = [
    "indexed_openai",
    "indexed_hybrid",
    "indexed_fastembed",
    "decision_jev",
    "chain",
    "merge",
    "chain_fastembed",
    "merge_fastembed",
]
OPENAI_MODEL = "text-embedding-3-small"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
# written from the intent name alone, before looking at val or test. an intent the seed picks
# that is not here gets no description, and DecisionRouter falls back to its utterances
DESCRIPTIONS = {
    "book_hotel": "Book a hotel room or other lodging for a place and dates.",
    "direct_deposit": "How to set up direct deposit of a paycheck into a bank account.",
    "ingredient_substitution": "Whether one cooking ingredient can be swapped for another.",
    "maybe": "The user answers with uncertainty: maybe, not sure, don't know.",
    "mpg": "The fuel economy (miles per gallon) of the user's car.",
    "nutrition_info": "Nutritional facts about a food: calories, fat, protein, how healthy it is.",
    "pto_used": "How many paid days off (PTO, vacation days) the user has already used.",
    "report_fraud": "Report a suspected fraudulent charge or activity on a bank or card account.",
    "restaurant_suggestion": "Recommend a restaurant to eat at.",
    "routing": "The routing number of the user's bank account.",
    "timer": "Set a countdown timer.",
    "tire_change": "When the tires on the user's car need replacing.",
    "transactions": "List recent transactions on the user's bank or card account.",
    "transfer": "Move money between the user's accounts or to another person.",
    "yes": "The user answers affirmatively: yes, correct, that's right.",
}

# calls that reached a real api, by provider
LIVE: dict[str, int] = defaultdict(int)
_VECS: dict[tuple[str, str], list[float]] = {}
_JEV: dict[str, dict] = {}


class Memo:
    """embeds each distinct text once per run: the accuracy pass scores the same queries many times."""

    name: str
    type: str

    def _miss(self, docs: list[str]) -> list[str]:
        return [d for d in dict.fromkeys(docs) if (self.name, d) not in _VECS]

    def _keep(self, docs: list[str], vecs: list[list[float]]) -> None:
        LIVE[self.type] += 1
        _VECS.update(zip([(self.name, d) for d in docs], vecs))

    def __call__(self, docs: list[str], *args, **kwargs) -> list[list[float]]:
        if miss := self._miss(docs):
            self._keep(miss, super().__call__(miss, *args, **kwargs))  # type: ignore[misc]
        return [_VECS[self.name, d] for d in docs]

    async def acall(self, docs: list[str], *args, **kwargs) -> list[list[float]]:
        if miss := self._miss(docs):
            self._keep(miss, await super().acall(miss, *args, **kwargs))  # type: ignore[misc]
        return [_VECS[self.name, d] for d in docs]


class MemoOpenAIEncoder(Memo, OpenAIEncoder):
    pass


class MemoFastEmbedEncoder(Memo, FastEmbedEncoder):
    pass


class LiveOpenAIEncoder(OpenAIEncoder):
    """the real encoder, counting requests."""

    def __call__(self, docs: list[str], truncate: bool = True) -> list[list[float]]:
        LIVE["openai"] += 1
        return super().__call__(docs, truncate)

    async def acall(self, docs: list[str], truncate: bool = True) -> list[list[float]]:
        LIVE["openai"] += 1
        return await super().acall(docs, truncate)


def jev_key(router: DecisionRouter, state: object, questions: dict) -> str:
    body = [router.url, router.model, state, questions]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class CachedDecisionRouter(DecisionRouter):
    """answers from the disk cache when the exact request was asked before."""

    async def aask(self, state: object, questions: dict) -> dict:
        key = jev_key(self, state, questions)
        if key not in _JEV:
            LIVE["jev"] += 1
            _JEV[key] = await super().aask(state, questions)
        return _JEV[key]


class LiveDecisionRouter(DecisionRouter):
    """the real router, counting requests."""

    async def aask(self, state: object, questions: dict) -> dict:
        LIVE["jev"] += 1
        return await super().aask(state, questions)


Labelled = list[tuple[str, str | None]]


def load_data(seed: int) -> tuple[list[Route], Labelled, Labelled, dict]:
    path = CACHE / "clinc_data_full.json"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise RuntimeError(f"{path} does not match the pinned CLINC150 file; delete it")
    data = json.loads(path.read_text())
    by: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for split in ("train", "val", "test"):
        for q, label in data[split]:
            by[split][label].append(q)
    rng = random.Random(seed)
    intents = sorted(rng.sample(sorted(by["train"]), N_INTENTS))
    routes = [
        Route(
            name=i,
            description=DESCRIPTIONS.get(i),
            utterances=rng.sample(by["train"][i], N_UTTERANCES),
        )
        for i in intents
    ]
    fit: Labelled = [(q, i) for i in intents for q in by["val"][i]]
    fit += [(q, None) for q in rng.sample([q for q, _ in data["oos_val"]], N_OOS_FIT)]
    test: Labelled = [(q, i) for i in intents for q in by["test"][i]]
    test += [
        (q, None) for q in rng.sample([q for q, _ in data["oos_test"]], N_OOS_TEST)
    ]
    # held out means held out: drop any test query that also appears as an utterance or in fit
    seen = {u for r in routes for u in r.utterances} | {q for q, _ in fit}
    dropped = sum(q in seen for q, _ in test)
    test = [(q, y) for q, y in test if q not in seen]
    meta = {
        "dataset": "clinc/oos-eval data_full.json",
        "url": DATA_URL,
        "sha256": DATA_SHA256,
        "intents": intents,
        "described": sorted(set(intents) & set(DESCRIPTIONS)),
        "utterances_per_route": N_UTTERANCES,
        "fit": {"in_scope": sum(y is not None for _, y in fit), "oos": N_OOS_FIT},
        "test": {
            "in_scope": sum(y is not None for _, y in test),
            "oos": sum(y is None for _, y in test),
            "dropped_as_seen": dropped,
        },
    }
    return routes, fit, test, meta


def top(probs: dict[str, float]) -> tuple[str, float]:
    name = max(probs, key=probs.__getitem__)
    return name, probs[name]


def e2e_at(tops: list[tuple[str, float]], y: Sequence[str | None], t: float) -> float:
    return float(np.mean([(n if c >= t else None) == g for (n, c), g in zip(tops, y)]))


async def measure_accuracy(router: BaseRouter, fit: Labelled, test: Labelled) -> dict:
    """fit on val, then score test: top-1, end-to-end with rejection, ece, and a threshold curve."""
    Xf, yf = [q for q, _ in fit], [y for _, y in fit]
    Xt, yt = [q for q, _ in test], [y for _, y in test]
    report = await router.afit(Xf, yf)
    router.confidence_threshold = BAR
    scored = await router._scores_many(Xt)
    tops, raw = [top(router._probs(s)) for s in scored], [top(s) for s in scored]
    inscope = np.array([y is not None for y in yt])
    picks = [n if c >= BAR else None for n, c in tops]
    correct = np.array([n == y for (n, _), y in zip(tops, yt)], dtype=float)
    # the threshold a person would pick from val, applied to test
    val_tops = [top(router._probs(s)) for s in await router._scores_many(Xf)]
    grid = [round(t, 2) for t in np.arange(0, 1.0001, 0.05)]
    tuned = max(grid, key=lambda t: (e2e_at(val_tops, yf, t), -abs(t - BAR)))
    e2e = await router.aevaluate(Xt, yt)
    if abs(e2e - np.mean([p == y for p, y in zip(picks, yt)])) > 1e-9:
        raise AssertionError("evaluate() disagrees with the picks this script scores")
    return {
        "top1": float(correct[inscope].mean()),
        "e2e": e2e,
        "in_scope_acc": float(np.mean([p == y for p, y in zip(picks, yt) if y])),
        "oos_recall": float(np.mean([p is None for p, y in zip(picks, yt) if not y])),
        "ece": ece(np.array([c for _, c in tops]), correct),
        # the same, on the raw scores: only a probabilistic router's mean much here
        "ece_raw": ece(np.clip([c for _, c in raw], 0, 1), correct),
        "nan_scores": sum(any(map(math.isnan, s.values())) for s in scored),
        "fit": {"accuracy": report.accuracy, "ece": report.ece, "n": report.n},
        "tuned": {"threshold": tuned, "e2e": e2e_at(tops, yt, tuned)},
        "curve": [[t, e2e_at(tops, yt, t)] for t in grid],
    }


async def chain_stops(chain: Chain, X: Sequence[str]) -> float:
    """the share of queries the first stage answers without asking the next."""
    first = chain.routers[0]
    stops = []
    for scores in await first._scores_many(X):
        name, c = top(first._probs(scores))
        # the bar Chain checks: the route's, else the router's, else the chain's; none means stop
        bar = first._bar(name)
        bar = chain.confidence_threshold if bar is None else bar
        stops.append(bar is None or c >= bar)
    return float(np.mean(stops))


def pct(ms: list[float], q: float) -> float:
    s = sorted(ms)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


async def measure_latency(
    router: BaseRouter, queries: list[str], batch: list[str], repeats: int
) -> dict:
    for q in queries[:WARMUP]:
        await router.acall(q)
    ms = []
    for q in queries[WARMUP:]:
        t = time.perf_counter()
        await router.acall(q)
        ms.append((time.perf_counter() - t) * 1000)
    secs = []
    for _ in range(repeats):
        t = time.perf_counter()
        await router.aevaluate(batch, [None] * len(batch))
        secs.append(time.perf_counter() - t)
    return {
        "n": len(ms),
        "p50": st.median(ms),
        "p95": pct(ms, 0.95),
        "mean": st.fmean(ms),
        "min": min(ms),
        "max": max(ms),
        "batch": len(batch),
        "batch_s": st.median(secs),
        "qps": len(batch) / st.median(secs),
    }


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=HERE,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def headline(r: dict) -> dict[str, float]:
    lat = r.get("latency", {})
    return {
        **{k: r[k] for k in ("top1", "e2e", "ece") if k in r},
        **{k: lat[k] for k in ("p50", "p95", "qps") if k in lat},
    }


def print_table(results: dict, previous: dict | None) -> None:
    cols = ["top1", "e2e", "ece", "p50", "p95", "qps"]
    print(f"\n{'router':18}" + "".join(f"{c:>16}" for c in cols))
    for name, r in results.items():
        if "error" in r:
            print(f"{name:18} error: {r['error']}")
            continue
        now, old = headline(r), headline((previous or {}).get(name, {}))
        cells = []
        for c in cols:
            fmt = "{:.3f}" if c in ("top1", "e2e", "ece") else "{:.0f}"
            v = fmt.format(now[c]) if c in now else "-"
            if c in now and c in old and fmt.format(now[c]) != fmt.format(old[c]):
                v += (
                    " ("
                    + ("+" if now[c] > old[c] else "")
                    + fmt.format(now[c] - old[c])
                    + ")"
                )
            cells.append(f"{v:>16}")
        print(f"{name:18}" + "".join(cells))
    if previous:
        print(
            "(in brackets: the change since the last results file; latency in ms, qps per sec)"
        )


async def main(args: argparse.Namespace) -> None:
    routes, fit, test, meta = load_data(args.seed)
    jev_path = CACHE / "jev.json"
    if jev_path.exists() and not args.fresh:
        _JEV.update(json.loads(jev_path.read_text()))
    rng = random.Random(args.seed + 1)
    lat_queries = [q for q, _ in rng.sample(test, args.n + WARMUP)]
    batch = [q for q, _ in rng.sample(test, args.batch)]

    decision = CachedDecisionRouter(routes)
    question = {"route": decision.route_question()}
    missing = {q for q, _ in fit + test if jev_key(decision, q, question) not in _JEV}
    timed = (
        ROUTERS
        if args.latency == "all"
        else []
        if args.latency == "none"
        else args.latency.split(",")
    )
    if unknown := set(timed) - set(ROUTERS):
        raise SystemExit(
            f"--latency: unknown routers {sorted(unknown)}; pick from {ROUTERS}"
        )
    per_router = args.n + WARMUP + args.batch * args.repeats
    # a ceiling: chains only ask jev on a fall-through
    jev_lat = per_router * sum(
        n.startswith(("decision", "chain", "merge")) for n in timed
    )
    # one request per query, and per batch for an indexed router; the composites fan out
    openai_lat = sum(
        (args.n + WARMUP)
        + (args.repeats if n.startswith("indexed") else args.repeats * args.batch)
        + 1
        for n in timed
        if n in ("indexed_openai", "indexed_hybrid", "chain", "merge")
    )
    print(
        f"seed {args.seed}: {len(routes)} routes, fit {len(fit)} ({N_OOS_FIT} oos), "
        f"test {len(test)} ({meta['test']['oos']} oos)\n"
        f"budget: jev {len(missing)} for accuracy ({len(fit) + len(test) - len(missing)} cached)"
        f" + up to {jev_lat} for latency; openai ~4 batched requests for accuracy"
        f" + ~{openai_lat} for latency"
    )
    if args.dry_run:
        return

    results: dict[str, dict] = {}
    fitted: dict[str, BaseRouter] = {}
    memo = MemoOpenAIEncoder(name=OPENAI_MODEL)
    memo_local = MemoFastEmbedEncoder(name=FASTEMBED_MODEL)
    candidates = {
        "indexed_openai": lambda: IndexedRouter(encoder=memo, routes=routes),
        "indexed_hybrid": lambda: IndexedRouter(
            encoder=memo, routes=routes, sparse_encoder=TfidfEncoder()
        ),
        "indexed_fastembed": lambda: IndexedRouter(encoder=memo_local, routes=routes),
        "decision_jev": lambda: decision,
    }
    try:
        for name, build in candidates.items():
            print(f"accuracy: {name}")
            try:
                router = build()
                results[name] = await measure_accuracy(router, fit, test)
                fitted[name] = router
            except Exception as e:  # one broken router should not hide the others
                results[name] = {"error": f"{type(e).__name__}: {e}"}

        if "indexed_hybrid" in fitted:
            sweep = []
            for alpha in ALPHA_SWEEP:
                print(f"accuracy: indexed_hybrid, alpha {alpha}")
                router = IndexedRouter(
                    encoder=memo,
                    routes=routes,
                    sparse_encoder=TfidfEncoder(),
                    alpha=alpha,
                )
                r = await measure_accuracy(router, fit, test)
                sweep.append(
                    {"alpha": alpha, **{k: r[k] for k in ("top1", "e2e", "ece")}}
                )
            results["indexed_hybrid"]["alpha_sweep"] = sweep
        # the composites, over the api index and over the local one
        for base, tag in (("indexed_openai", ""), ("indexed_fastembed", "_fastembed")):
            if not {base, "decision_jev"} <= set(fitted):
                continue
            idx, dec = fitted[base], fitted["decision_jev"]
            # the chain's first stage is the fitted indexed router with a higher bar to stop on
            sweep = []
            for bar in STOP_SWEEP:
                first = copy.copy(idx)
                first.confidence_threshold = bar
                chain = Chain(routers=[first, dec], confidence_threshold=BAR)
                print(f"accuracy: chain{tag}, stop at {bar}")
                r = await measure_accuracy(chain, fit, test)
                r["stops"] = await chain_stops(chain, [q for q, _ in test])
                keep = ("top1", "e2e", "ece", "stops")
                sweep.append({"stop_bar": bar, **{k: r[k] for k in keep}})
                if bar == STOP_BAR:
                    r["stops_by"] = {
                        "in_scope": await chain_stops(chain, [q for q, y in test if y]),
                        "oos": await chain_stops(chain, [q for q, y in test if not y]),
                    }
                    results[f"chain{tag}"], fitted[f"chain{tag}"] = r, chain
            results[f"chain{tag}"]["stop_sweep"] = sweep
            print(f"accuracy: merge{tag}")
            merge = Merge(routers=[idx, dec], confidence_threshold=BAR)
            results[f"merge{tag}"] = await measure_accuracy(merge, fit, test)
            fitted[f"merge{tag}"] = merge
    finally:
        CACHE.mkdir(parents=True, exist_ok=True)
        jev_path.write_text(json.dumps(_JEV))
        await decision.aclose()
    jev_accuracy, openai_accuracy = LIVE["jev"], LIVE["openai"]

    previous = json.loads(OUT.read_text()).get("routers", {}) if OUT.exists() else {}
    if timed:
        # fresh routers on the real apis, carrying the calibrators and bars fitted above
        def like(name: str, router: BaseRouter) -> BaseRouter:
            router.calibrator = fitted[name].calibrator
            router.confidence_threshold = fitted[name].confidence_threshold
            return router

        need = set(timed)
        for base, tag in (("indexed_openai", ""), ("indexed_fastembed", "_fastembed")):
            if need & {f"chain{tag}", f"merge{tag}"}:
                need |= {base, "decision_jev"}
        live = LiveOpenAIEncoder(name=OPENAI_MODEL)
        jev = LiveDecisionRouter(routes)
        builds = {
            "indexed_openai": lambda: IndexedRouter(encoder=live, routes=routes),
            "indexed_hybrid": lambda: IndexedRouter(
                encoder=live, routes=routes, sparse_encoder=TfidfEncoder()
            ),
            "indexed_fastembed": lambda: IndexedRouter(
                encoder=FastEmbedEncoder(name=FASTEMBED_MODEL), routes=routes
            ),
        }
        lat_routers: dict[str, BaseRouter] = {}
        build_s: dict[str, float] = {}
        for name, build in builds.items():
            if name in need and name in fitted:
                t = time.perf_counter()  # for fastembed this includes loading the model
                lat_routers[name] = like(name, build())
                build_s[name] = time.perf_counter() - t
        if "decision_jev" in need and "decision_jev" in fitted:
            lat_routers["decision_jev"] = like("decision_jev", jev)
        for base, tag in (("indexed_openai", ""), ("indexed_fastembed", "_fastembed")):
            if f"chain{tag}" in need and f"chain{tag}" in fitted:
                first = like(base, copy.copy(lat_routers[base]))
                first.confidence_threshold = STOP_BAR
                chain = Chain(routers=[first, jev])
                lat_routers[f"chain{tag}"] = like(f"chain{tag}", chain)
            if f"merge{tag}" in need and f"merge{tag}" in fitted:
                merge = Merge(routers=[lat_routers[base], jev])
                lat_routers[f"merge{tag}"] = like(f"merge{tag}", merge)
        try:
            for name in timed:
                if name not in lat_routers:
                    continue
                print(f"latency: {name}")
                before, load = LIVE["jev"], os.getloadavg()[0]
                lat = await measure_latency(
                    lat_routers[name], lat_queries, batch, args.repeats
                )
                if name.startswith("chain"):
                    # every jev call a chain makes is a query its index passed on
                    asked = len(lat_queries) + args.repeats * len(batch)
                    lat["fell_through"] = (LIVE["jev"] - before) / asked
                if name in build_s:
                    lat["build_s"] = build_s[name]
                # this machine's 1-minute load average: local encoders slow down when it is busy
                lat["load"] = load
                lat["git"] = git_sha()
                lat["date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                results[name]["latency"] = lat
        finally:
            await jev.aclose()
    for name, r in results.items():
        # a router not timed this run keeps its last timing, which says when and where it was taken
        old = previous.get(name, {}).get("latency")
        if "latency" not in r and "error" not in r and old:
            r["latency"] = {**old, "carried": True}

    print_table(results, previous)
    meta.update(
        {
            "seed": args.seed,
            "bar": BAR,
            "chain_stop_bar": STOP_BAR,
            "models": {
                "openai": OPENAI_MODEL,
                "fastembed": FASTEMBED_MODEL,
                "jev": decision.model,
            },
            "git": git_sha(),
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    budget = {
        "jev": {"accuracy": jev_accuracy, "latency": LIVE["jev"] - jev_accuracy},
        "openai": {
            "accuracy": openai_accuracy,
            "latency": LIVE["openai"] - openai_accuracy,
        },
    }
    print(f"\nlive calls: {json.dumps(budget)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"meta": meta, "budget": budget, "routers": results}, indent=1)
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=30, help="single-query latency samples")
    ap.add_argument(
        "--batch", type=int, default=50, help="queries per throughput batch"
    )
    ap.add_argument(
        "--repeats", type=int, default=2, help="throughput batches per router"
    )
    ap.add_argument(
        "--latency",
        default="all",
        help="routers to time, comma-separated, or all / none; the rest keep their last timing",
    )
    ap.add_argument("--fresh", action="store_true", help="ignore cached jev answers")
    ap.add_argument("--dry-run", action="store_true", help="print the budget and stop")
    asyncio.run(main(ap.parse_args()))
