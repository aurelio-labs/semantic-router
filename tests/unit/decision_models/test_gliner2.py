from types import SimpleNamespace

import pytest

from semantic_router.decision_models import choice, noul, score
from semantic_router.decision_models.gliner2 import _safe, _Task


class FakeSchema:
    """records the tasks a question adds, the way gliner2's ClassificationSchema takes them"""

    def __init__(self):
        self.tasks = []

    def single(self, key, labels, instruction=None):
        self.tasks.append(("single", key, dict(labels), instruction))
        return self

    def ordinal(self, key, labels, instruction=None):
        self.tasks.append(("ordinal", key, dict(labels), instruction))
        return self


def result(probabilities):
    top = max(probabilities, key=probabilities.__getitem__)
    return SimpleNamespace(probabilities=probabilities, label=top)


def added(question, key="q"):
    schema = FakeSchema()
    _Task(key, question).add_to(schema)
    return schema.tasks[0]


# --- questions become gliner2 tasks; no model needed ---


def test_a_choice_is_a_single_label_task_with_descriptions():
    q = choice("Which team?", {"billing": "money owed", "technical": "it's broken"})
    assert added(q) == (
        "single",
        "q",
        {"billing": "money owed", "technical": "it's broken"},
        "Which team?",
    )


def test_a_noul_labels_with_its_yes_and_no_texts():
    # as bare true/false labels with descriptions, GLiNER2.5-Decide answers near 0.5
    kind, _, labels, ins = added(noul("Urgent?", yes="needs action now", no="can wait"))
    assert kind == "single" and labels == {"needs action now": None, "can wait": None}
    assert ins == "Urgent?"


def test_a_score_is_an_ordinal_task_over_its_levels():
    kind, _, labels, _ = added(score("Upset?", levels=["calm", "annoyed", "angry"]))
    assert kind == "ordinal" and list(labels) == ["calm", "annoyed", "angry"]


def test_brackets_gliner2_refuses_are_blanked():
    assert _safe("refunds (not pricing) [P] here") == "refunds not pricing here"
    q = choice("Pick (one)", {"billing": "charges (and refunds)", "other": "x"})
    _, _, labels, ins = added(q)
    assert labels["billing"] == "charges and refunds" and ins == "Pick one"


def test_options_that_clash_once_cleaned_are_refused():
    with pytest.raises(ValueError, match="can't tell them apart"):
        _Task("q", choice("Pick", {"a (1)": "x", "a 1": "y"}))


def test_an_unknown_question_type_is_refused():
    with pytest.raises(ValueError, match="unknown type"):
        _Task("q", {"type": "rank", "instructions": "x", "criteria": ["a"]})


# --- gliner2 results become systemone answers ---


def test_a_choice_answer_maps_back_to_route_names():
    t = _Task(
        "route", choice("Which?", {"billing (b)": "money", "technical": "broken"})
    )
    a = t.answer(result({"billing b": 0.7, "technical": 0.3}))
    assert a["type"] == "choice" and a["choice"] == "billing (b)"
    assert a["probabilities"] == {"billing (b)": 0.7, "technical": 0.3}
    assert a["confidence"] == pytest.approx(0.4)


def test_a_noul_answer_is_the_probability_of_yes():
    t = _Task("urgent", noul("Urgent?", yes="now", no="later"))
    a = t.answer(result({"now": 0.9, "later": 0.1}))
    assert a == {"type": "noul", "noul": 0.9, "confidence": pytest.approx(0.8)}


def test_a_score_answer_is_the_expected_level():
    t = _Task("mood", score("Upset?", levels=["calm", "annoyed", "angry"]))
    a = t.answer(result({"calm": 0.0, "annoyed": 0.08, "angry": 0.92}))
    assert a["score"] == pytest.approx(1.92)
    assert a["probabilities"] == {"0": 0.0, "1": 0.08, "2": 0.92}
    assert a["legend"] == {"0": "calm", "1": "annoyed", "2": "angry"}


# --- the model, with a stand-in classifier (needs gliner2 for its schema class) ---


class FakeClassifier:
    """answers every task by putting 0.8 on its first label"""

    def __init__(self):
        self.batches = []

    def batch_classify(self, texts, schema, config=None):
        self.batches.append(list(texts))
        out = []
        for _ in texts:
            row = {}
            for spec in schema.task_specs:
                names = [label.name for label in spec.labels]
                rest = 0.2 / max(len(names) - 1, 1)
                probs = {n: (0.8 if i == 0 else rest) for i, n in enumerate(names)}
                row[spec.name] = result(probs)
            out.append(row)
        return out


def test_the_model_batches_states_and_answers_every_question():
    pytest.importorskip("gliner2")
    from semantic_router import DecisionRouter, Route
    from semantic_router.decision_models import GLiNER2DecisionModel

    clf = FakeClassifier()
    model = GLiNER2DecisionModel(classifier=clf)
    routes = [Route(name="billing", description="money"), Route(name="tech")]
    r = DecisionRouter(model=model, routes=routes)
    assert r.evaluate(X=["a", "b", "c"], y=["billing"] * 3) == 1.0
    assert clf.batches == [["a", "b", "c"]]  # one batched pass, not three
    out = model.ask(
        {"subject": "refund", "body": "charged twice"},
        {"route": r.route_question(), "urgent": noul("Urgent?")},
    )
    assert set(out) == {"route", "urgent"} and out["urgent"]["noul"] == 0.8
    # a JSON state reaches the model as its compact JSON text
    assert clf.batches[-1] == ['{"subject":"refund","body":"charged twice"}']


# --- the real model: runs where the local extra is installed, like test_local.py ---


def test_glinerdecide_routes_and_answers_on_this_machine():
    pytest.importorskip("torch")
    pytest.importorskip("gliner2")
    from semantic_router import DecisionRouter, Route
    from semantic_router.decision_models import GLiNER2DecisionModel

    routes = [
        Route(
            name="billing", description="Money already paid or owed: refunds, charges."
        ),
        Route(name="technical", description="Something in the product is broken."),
        Route(
            name="account", description="Logging in, passwords and account settings."
        ),
    ]
    r = DecisionRouter(model=GLiNER2DecisionModel(), routes=routes)
    assert r.calibrated
    assert r("why was I charged twice this month?").name == "billing"
    assert r("the app crashes when I open settings").name == "technical"
    assert r("I can't remember my password").name == "account"
    out = r.model.ask(
        "my card was charged twice and I need it fixed today or I'm cancelling",
        {
            "urgent": noul(
                "Is this time-sensitive?", yes="needs action now", no="can wait"
            )
        },
    )
    assert out["urgent"]["noul"] > 0.5
