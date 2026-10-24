"""turn a raw router score into P(route is correct). monotonic, so ranking never changes."""

from __future__ import annotations

from typing import Literal

import numpy as np

from semantic_router.utils.logger import logger

Method = Literal["platt", "isotonic"]


class Calibrator:
    """a fitted score -> probability map. build one with `Calibrator.fit`."""

    def __init__(self, method: Method, params: dict[str, list[float]]):
        self.method = method
        self.params = params

    @classmethod
    def fit(
        cls, scores: np.ndarray, correct: np.ndarray, method: Method = "platt"
    ) -> Calibrator:
        s, y = np.asarray(scores, dtype=float), np.asarray(correct, dtype=float)
        if s.size < 2 or y.min() == y.max():
            raise ValueError("calibration needs both correct and incorrect examples")
        return cls(method, _platt(s, y) if method == "platt" else _isotonic(s, y))

    def __call__(self, scores: np.ndarray | float) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if self.method == "platt":
            a, b = self.params["a"][0], self.params["b"][0]
            return 1.0 / (1.0 + np.exp(-(a * s + b)))
        return np.interp(s, self.params["x"], self.params["y"])

    def to_dict(self) -> dict[str, object]:
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Calibrator:
        return cls(data["method"], data["params"])  # type: ignore[arg-type]


def _platt(s: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    # platt's smoothed targets stop a thin class pinning the curve at 0 or 1
    n_pos, n_neg = y.sum(), y.size - y.sum()
    t = np.where(y > 0.5, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    a, b = 1.0, -float(s.mean())
    for _ in range(50):
        # newton on the convex log-loss of p = sigmoid(a*s + b)
        p = 1 / (1 + np.exp(-(a * s + b)))
        w = p * (1 - p) + 1e-9
        g = np.array([np.sum((p - t) * s), np.sum(p - t)])
        h = np.array([[np.sum(w * s * s), np.sum(w * s)], [np.sum(w * s), np.sum(w)]])
        step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    if a <= 0:
        # a falling curve would rank a weaker match above a stronger one. on these points
        # the score doesn't predict a right answer, so say the base rate for every score
        base = float(t.mean())
        logger.warning(
            "calibration: higher scores weren't more often right on this data, so every "
            f"confidence is the base rate ({base:.2f}). fit on more held-out queries."
        )
        a, b = 0.0, float(np.log(base / (1 - base)))
    return {"a": [float(a)], "b": [float(b)]}


def _isotonic(s: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    # A score has one probability regardless of the order of tied examples.
    # Keep their counts so pooling weights each sample equally, not each score.
    xs, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=y)
    # pool adjacent violators: merge neighbouring blocks until the means are non-decreasing
    blocks: list[list[float]] = []  # [sum_y, count, x_lo, x_hi]
    for x, total, count in zip(xs, sums, counts):
        blocks.append([float(total), float(count), float(x), float(x)])
        while (
            len(blocks) > 1
            and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]
        ):
            hi = blocks.pop()
            blocks[-1] = [
                blocks[-1][0] + hi[0],
                blocks[-1][1] + hi[1],
                blocks[-1][2],
                hi[3],
            ]
    # clip the block means like platt does, so the tails never claim certainty
    lo, hi_ = 1 / (s.size + 2), (s.size + 1) / (s.size + 2)
    px, py = [], []
    for sum_y, n, x_lo, x_hi in blocks:
        v = float(np.clip(sum_y / n, lo, hi_))
        px.append(x_lo)
        py.append(v)
        if x_hi != x_lo:
            px.append(x_hi)
            py.append(v)
    return {"x": px, "y": py}


def ece(probs: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """expected calibration error: bin-weighted gap between stated confidence and observed accuracy."""
    p, y = np.asarray(probs, dtype=float), np.asarray(correct, dtype=float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)
