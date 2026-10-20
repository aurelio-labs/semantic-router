"""GLiNER2 decision models, run locally: fastino/GLiNER2.5-Decide by default."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Any, Sequence

from pydantic import Field, PrivateAttr

from semantic_router.decision_models.base import Answer, DecisionModel, Question

# gliner2 writes every label, description and instruction into the model prompt and
# rejects any that contain its marker tokens, so they are blanked out first
_RESERVED = (
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "[P]",
    "[L]",
    "[C]",
    "[E]",
    "[R]",
)


def _safe(text: object) -> str:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    for token in _RESERVED:
        s = s.replace(token, " ")
    return re.sub(r"\s+", " ", s.replace("(", " ").replace(")", " ")).strip()


def _text(state: object) -> str:
    # a JSON state (labelled fields, a list of parts) is read as its compact JSON text
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def _margin(probs: Sequence[float]) -> float:
    top = sorted(probs, reverse=True)
    return top[0] - top[1] if len(top) > 1 else top[0]


class _Task:
    """one question as a gliner2 task: its labels, and the way back to our names."""

    def __init__(self, key: str, question: Question):
        self.key, self.type = key, question.get("type")
        self.instruction = _safe(question.get("instructions", "")) or None
        criteria = question.get("criteria")
        # gliner2 label -> our option name ("0", "1", ... for score levels), and the
        # description each label is shown with
        self.names: dict[str, str] = {}
        self.labels: dict[str, str | None] = {}
        self.legend: dict[str, str] = {}
        if self.type == "score":
            if isinstance(criteria, dict):
                # numeric-string keys, sorted; answers use 0-based positions
                criteria = [criteria[k] for k in sorted(criteria, key=float)]
            if not isinstance(criteria, list) or not criteria:
                raise ValueError(f"score question {key!r} needs criteria levels")
            for i, level in enumerate(criteria):
                # a level's own text is its label, so the model reads it
                self._add(_safe(level), str(i), None)
                self.legend[str(i)] = str(level)
            return
        if self.type == "noul":
            c = criteria if isinstance(criteria, dict) else {}
            # the yes and no texts are the labels: as bare true/false labels with the
            # texts as descriptions, GLiNER2.5-Decide answers near 0.5
            self._add(_safe(c.get("true", "yes")), "true", None)
            self._add(_safe(c.get("false", "no")), "false", None)
            return
        if self.type != "choice":
            raise ValueError(f"question {key!r} has unknown type {self.type!r}")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"choice question {key!r} needs criteria options")
        for name, desc in criteria.items():
            self._add(_safe(name), str(name), _safe(desc) if desc is not None else None)

    def _add(self, label: str, name: str, description: str | None) -> None:
        if not label or label in self.names:
            raise ValueError(
                f"option {name!r} in {self.key!r} is empty or matches another once "
                "brackets are removed, so gliner2 can't tell them apart"
            )
        self.names[label] = name
        self.labels[label] = description or None

    def add_to(self, schema: Any) -> None:
        build = schema.ordinal if self.type == "score" else schema.single
        build(self.key, self.labels, instruction=self.instruction)

    def answer(self, result: Any) -> Answer:
        p = {self.names[label]: float(v) for label, v in result.probabilities.items()}
        if self.type == "noul":
            yes = p["true"]
            return {"type": "noul", "noul": yes, "confidence": abs(2 * yes - 1)}
        if self.type == "choice":
            return {
                "type": "choice",
                "choice": self.names[result.label],
                "confidence": _margin(list(p.values())),
                "probabilities": p,
            }
        levels = sorted(p, key=int)
        return {
            "type": "score",
            "score": sum(int(k) * p[k] for k in levels),  # the expected level, 0-based
            "confidence": _margin(list(p.values())),
            "probabilities": {k: p[k] for k in levels},
            "legend": self.legend,
        }


class GLiNER2DecisionModel(DecisionModel):
    """A GLiNER2 decision model run in-process with the gliner2 library.

    Answers choice, noul and score questions locally, with no network calls and no
    key. Requires the `local` extra: `pip install "semantic-router[local]"`.

    :param name: The model on the Hugging Face Hub, or a local directory.
    :param device: `"cuda"`, `"mps"` or `"cpu"`. Left out, the first available.
    :param batch_size: States per forward pass when answering many at once.
    :param classifier: A loaded `gliner2.classification.Classifier` to use instead of
        loading `name`, e.g. one with a fine-tuned adapter.
    """

    type: str = "gliner2"
    name: str = "fastino/GLiNER2.5-Decide"
    device: str | None = None
    batch_size: int = 8
    classifier: Any = Field(default=None, exclude=True, repr=False)

    _clf: Any = PrivateAttr(default=None)
    _config: Any = PrivateAttr(default=None)
    _schema_cls: Any = PrivateAttr(default=None)
    # one forward pass at a time: torch on mps and cuda doesn't promise thread safety
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def model_post_init(self, __context: Any) -> None:
        try:
            from gliner2.classification import (
                ClassificationConfig,
                ClassificationSchema,
                Classifier,
            )
        except ImportError as e:
            raise ImportError(
                "GLiNER2DecisionModel needs gliner2. Install it with "
                '`pip install "semantic-router[local]"`'
            ) from e
        self._schema_cls = ClassificationSchema
        self._config = ClassificationConfig(batch_size=self.batch_size)
        if self.classifier is not None:
            self._clf = self.classifier
            return
        clf = Classifier.from_pretrained(self.name)
        self._clf = clf.to(device=self.device or _best_device()).eval()

    def _tasks(self, questions: dict[str, Question]) -> list[_Task]:
        if not questions:
            raise ValueError("ask at least one question")
        return [_Task(key, q) for key, q in questions.items()]

    def ask_many(
        self, states: Sequence[object], questions: dict[str, Question]
    ) -> list[dict[str, Answer]]:
        """answer the same questions about many states in batched forward passes."""
        tasks = self._tasks(questions)
        schema = self._schema_cls()
        for t in tasks:
            t.add_to(schema)
        texts = [_text(s) for s in states]
        with self._lock:
            results = self._clf.batch_classify(texts, schema, config=self._config)
        return [{t.key: t.answer(r[t.key]) for t in tasks} for r in results]

    def ask(self, state: object, questions: dict[str, Question]) -> dict[str, Answer]:
        """answer every question about one state in one forward pass."""
        return self.ask_many([state], questions)[0]

    async def aask(
        self, state: object, questions: dict[str, Question]
    ) -> dict[str, Answer]:
        # inference blocks, so it runs off the event loop
        return await asyncio.to_thread(self.ask, state, questions)

    async def aask_many(
        self, states: Sequence[object], questions: dict[str, Question]
    ) -> list[dict[str, Answer]]:
        return await asyncio.to_thread(self.ask_many, list(states), questions)


def _best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
