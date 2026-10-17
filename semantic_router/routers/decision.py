"""a router that asks a decision model which route applies. one call answers many questions."""

from __future__ import annotations

from typing import Sequence

from semantic_router.calibration import Calibrator
from semantic_router.decision_models.base import (
    Answer,
    DecisionModel,
    Question,
    choice,
)
from semantic_router.route import Route
from semantic_router.routers.base import BaseRouter


class DecisionRouter(BaseRouter):
    """routes by asking a decision model one choice question: which route applies.

    :param model: The decision model that answers, e.g. `TypeSafeDecisionModel()`
        over HTTP.
    :param routes: The routes to choose between.
    :param instructions: The question asked about each query.
    :param confidence_threshold: The confidence a route needs, unless it sets its own.
    :param calibrator: A fitted score -> probability map; `fit()` makes one.
    """

    def __init__(
        self,
        model: DecisionModel,
        routes: Sequence[Route] = (),
        instructions: str = "Which route does the user's message belong to?",
        confidence_threshold: float | None = None,
        calibrator: Calibrator | None = None,
    ):
        super().__init__(routes, confidence_threshold, calibrator)
        self.model = model
        self.instructions = instructions
        # the model says whether its answers are probabilities, not the router
        self.probabilistic = model.probabilistic

    def route_question(self) -> Question:
        """the choice question for the current route set; descriptions beat utterance samples"""
        return choice(
            self.instructions,
            {
                r.name: r.description or "; ".join(str(u) for u in r.utterances[:5])
                for r in self.routes
            },
        )

    async def aask(
        self, state: object, questions: dict[str, Question]
    ) -> dict[str, Answer]:
        """answer several typed questions about one state in a single model pass."""
        return await self.model.aask(state, questions)

    def _route_scores(self, answer: Answer) -> dict[str, float]:
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            raise RuntimeError(f"decision model answer has no probabilities: {answer}")
        return {r.name: float(probs.get(r.name, 0.0)) for r in self.routes}

    async def _scores(self, query: str) -> dict[str, float]:
        return (await self._scores_many([query]))[0]

    async def _scores_many(self, queries: Sequence[str]) -> list[dict[str, float]]:
        # one call into the model for the whole batch: fit, evaluate, Chain and Merge
        question = {"route": self.route_question()}
        answers = await self.model.aask_many(list(queries), question)
        return [self._route_scores(a["route"]) for a in answers]

    async def aclose(self) -> None:
        await self.model.aclose()

    def close(self) -> None:
        self.model.close()
