from semantic_router.route import Route


def test_defaults_let_a_route_be_just_a_name_and_description():
    route = Route(name="billing", description="Refunds, charges and invoices.")
    assert route.utterances == []
    assert route.confidence_threshold is None
    assert route.metadata == {}


def test_to_dict():
    route = Route(name="test", utterances=["utterance"], confidence_threshold=0.9)
    assert route.to_dict() == {
        "name": "test",
        "utterances": ["utterance"],
        "description": None,
        "confidence_threshold": 0.9,
        "metadata": {},
    }


def test_from_dict_round_trips():
    route = Route(name="test", utterances=["utterance"], description="d")
    assert Route.from_dict(route.to_dict()) == route
