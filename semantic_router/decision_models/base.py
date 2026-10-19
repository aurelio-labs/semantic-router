"""the decision-model contract: answer typed questions about a state, with probabilities.

a decision model is to DecisionRouter what an encoder is to IndexedRouter. the router
turns its routes into one choice question; the model answers it.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar, Sequence

from pydantic import BaseModel, ConfigDict, Field

Question = dict[str, object]
Answer = dict[str, object]


def choice(instructions: str, options: dict[str, str]) -> Question:
    """which one of several named options applies. `options` maps a name to its meaning."""
    return {"type": "choice", "instructions": instructions, "criteria": options}


def noul(instructions: str, yes: str = "yes", no: str = "no") -> Question:
    """a yes/no question, answered with the probability of yes."""
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {"true": yes, "false": no},
    }


def score(instructions: str, levels: Sequence[str]) -> Question:
    """where the state sits on an ordered scale. `levels` go lowest first."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


class DecisionModel(BaseModel):
    """answers typed questions (choice, noul, score) about a state.

    questions and answers use the /v1/systemone shapes, whatever runs the model: a
    model over HTTP sends them as they are, a local model maps them onto its own API.
    """

    name: str
    type: str = Field(default="base")
    # true when every answer carries real probabilities, so a router's confidence is
    # one without a fit
    probabilistic: ClassVar[bool] = True

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    async def aask(
        self, state: object, questions: dict[str, Question]
    ) -> dict[str, Answer]:
        """answer every question about one state, keyed by the question names."""
        raise NotImplementedError("Subclasses must implement this method")

    async def aask_many(
        self, states: Sequence[object], questions: dict[str, Question]
    ) -> list[dict[str, Answer]]:
        """answer the same questions about many states, one result per state."""
        # one aask per state; a model that batches overrides this
        return list(await asyncio.gather(*(self.aask(s, questions) for s in states)))

    async def aclose(self) -> None:
        """release sessions or weights. a no-op unless the model holds any."""

    def close(self) -> None:
        """release sessions or weights from sync code."""
