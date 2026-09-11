"""Unit tests for RedisIndex's pure logic -- escaping, ID derivation and metric
mapping -- that don't need a live Redis/RediSearch server.

Behaviour that requires an actual RediSearch index (add/query/delete/sync) is
covered by the parametrized suite in tests/integration/test_router_integration.py,
run against the redis-stack container in compose.yaml (`make services`).
"""

import uuid

import pytest

pytest.importorskip("redis", reason="redis not installed")

from semantic_router.index.redis import RedisIndex, _escape_tag
from semantic_router.schema import Metric

DIMS = 4


def make_index(**kwargs):
    """Create a RedisIndex without touching the network -- redis-py's client
    constructor is lazy, so this never connects."""
    return RedisIndex(dimensions=DIMS, **kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self):
        idx = make_index()
        assert idx.namespace is None
        assert idx.index_name == "semantic-router-index"
        assert idx.metric == Metric.COSINE
        assert idx.algorithm == "FLAT"
        assert not idx._index_initialized

    def test_clients_initialized(self):
        idx = make_index()
        assert idx.client is not None
        assert idx.aclient is not None

    def test_index_marker_set_eagerly(self):
        """`self.index` (BaseIndex's readiness marker) must be set in __init__,
        not deferred to the first `add()` -- otherwise a fresh handle onto an
        already-populated remote index looks uninitialized to
        `BaseIndex.get_utterances()`/`get_routes()`, breaking auto_sync='remote'."""
        idx = make_index()
        assert idx.index is idx

    def test_host_port_default_from_env(self, monkeypatch):
        monkeypatch.delenv("REDIS_HOST", raising=False)
        monkeypatch.delenv("REDIS_PORT", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        idx = make_index()
        assert idx.host == "localhost"
        assert idx.port == 6379

    def test_explicit_host_port_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "example.com")
        idx = make_index(host="myhost", port=1234)
        assert idx.host == "myhost"
        assert idx.port == 1234

    def test_connection_string_from_env(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://example.com:6380/2")
        idx = make_index()
        assert idx.connection_string == "redis://example.com:6380/2"

    def test_missing_redis_raises_helpful_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "redis" or name.startswith("redis."):
                raise ImportError("no redis")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="semantic-router\\[redis\\]"):
            make_index()


# ---------------------------------------------------------------------------
# _escape_tag
# ---------------------------------------------------------------------------


class TestEscapeTag:
    def test_plain_alphanumeric_untouched(self):
        assert _escape_tag("billing") == "billing"

    def test_escapes_space(self):
        assert _escape_tag("weird route name") == "weird\\ route\\ name"

    def test_escapes_hyphen(self):
        assert _escape_tag("my-route") == "my\\-route"

    def test_escapes_multiple_special_chars(self):
        assert _escape_tag("a.b@c!d") == "a\\.b\\@c\\!d"

    def test_non_str_input_coerced(self):
        assert _escape_tag(123) == "123"


# ---------------------------------------------------------------------------
# _redis_metric
# ---------------------------------------------------------------------------


class TestMetricMapping:
    @pytest.mark.parametrize(
        "metric,expected",
        [
            (Metric.COSINE, "COSINE"),
            (Metric.EUCLIDEAN, "L2"),
            (Metric.DOTPRODUCT, "IP"),
        ],
    )
    def test_supported_metrics(self, metric, expected):
        idx = make_index(metric=metric)
        assert idx._redis_metric() == expected

    def test_unsupported_metric_raises(self):
        idx = make_index(metric=Metric.MANHATTAN)
        with pytest.raises(ValueError, match="Unsupported Redis distance metric"):
            idx._redis_metric()


class TestDistanceToScore:
    def test_cosine_inverts_distance(self):
        idx = make_index(metric=Metric.COSINE)
        assert idx._distance_to_score(0.3) == pytest.approx(0.7)

    def test_dotproduct_inverts_distance(self):
        idx = make_index(metric=Metric.DOTPRODUCT)
        assert idx._distance_to_score(0.4) == pytest.approx(0.6)

    def test_euclidean_passes_through(self):
        idx = make_index(metric=Metric.EUCLIDEAN)
        assert idx._distance_to_score(1.23) == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------


class TestIds:
    def test_deterministic_without_namespace(self):
        idx = make_index()
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, "r:utt"))
        assert idx._id_for("r", "utt") == expected

    def test_deterministic_with_namespace(self):
        idx = make_index(namespace="org-abc")
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, "org-abc:r:utt"))
        assert idx._id_for("r", "utt") == expected

    def test_namespace_changes_id(self):
        assert make_index()._id_for("r", "utt") != make_index(
            namespace="org-x"
        )._id_for("r", "utt")

    def test_key_includes_prefix(self):
        idx = make_index(index_name="my-idx")
        key = idx._key_for("r", "utt")
        assert key == f"my-idx:{idx._id_for('r', 'utt')}"


# ---------------------------------------------------------------------------
# Filter query building
# ---------------------------------------------------------------------------


class TestFilterQuery:
    def test_no_namespace_no_extra_is_wildcard(self):
        idx = make_index()
        assert idx._filter_query() == "*"

    def test_namespace_only(self):
        idx = make_index(namespace="org-1")
        assert idx._filter_query() == "(@namespace:{org\\-1})"

    def test_route_filter_term_joins_with_pipe(self):
        idx = make_index()
        term = idx._route_filter_term(["billing", "support"])
        assert term == "@sr_route:{billing|support}"

    def test_namespace_and_route_filter_combine(self):
        idx = make_index(namespace="org-1")
        term = idx._route_filter_term(["billing"])
        assert idx._filter_query([term]) == "(@namespace:{org\\-1} @sr_route:{billing})"


# ---------------------------------------------------------------------------
# Mapping construction
# ---------------------------------------------------------------------------


class TestBuildMapping:
    def test_mapping_contains_expected_fields(self):
        idx = make_index()
        mapping = idx._build_mapping(
            "r", "utt", [0.1, 0.2, 0.3, 0.4], {"foo": "bar"}, {"name": "fn"}
        )
        assert mapping["sr_route"] == "r"
        assert isinstance(mapping["vector"], bytes)
        assert "namespace" not in mapping

        import json

        record = json.loads(mapping["sr_record"])
        assert record["sr_route"] == "r"
        assert record["sr_utterance"] == "utt"
        assert record["sr_function_schema"] == '{"name": "fn"}'
        assert record["foo"] == "bar"

    def test_mapping_includes_namespace_when_set(self):
        idx = make_index(namespace="org-1")
        mapping = idx._build_mapping("r", "utt", [0.1, 0.2, 0.3, 0.4], {}, None)
        assert mapping["namespace"] == "org-1"

    def test_no_function_schema_defaults_empty(self):
        idx = make_index()
        mapping = idx._build_mapping("r", "utt", [0.1, 0.2, 0.3, 0.4], {}, None)
        import json

        record = json.loads(mapping["sr_record"])
        assert record["sr_function_schema"] == "{}"
