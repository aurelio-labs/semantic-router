from unittest.mock import Mock

from semantic_router.routers.base import threshold_random_search


def test_threshold_random_search_seed_is_reproducible():
    route_layer = Mock()
    route_layer.get_thresholds.return_value = {"support": 0.5, "billing": 0.7}

    first = threshold_random_search(route_layer, search_range=0.2, seed=42)
    second = threshold_random_search(route_layer, search_range=0.2, seed=42)

    assert first == second


def test_threshold_random_search_does_not_change_global_random_state():
    route_layer = Mock()
    route_layer.get_thresholds.return_value = {"support": 0.5}

    import random

    random.seed(1234)
    expected = random.random()
    random.seed(1234)

    threshold_random_search(route_layer, search_range=0.2, seed=42)

    assert random.random() == expected
