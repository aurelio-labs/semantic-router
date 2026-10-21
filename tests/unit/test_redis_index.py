"""Unit tests for RedisIndex's pure logic -- filter tags, ID derivation and
metric mapping -- that don't need a live Redis/RediSearch server.

Behaviour that requires an actual RediSearch index (add/query/delete/sync) is
covered by the parametrized suite in tests/integration/test_router_integration.py
and by tests/integration/test_redis_integration.py, run against the redis-stack
container in compose.yaml (`make services`).
"""

import asyncio
import json
import uuid

import numpy as np
import pytest

pytest.importorskip("redis", reason="redis not installed")

from semantic_router.index.redis import RedisIndex, _missing_index, _tag
from semantic_router.schema import Metric

DIMS = 4


def make_index(**kwargs):
    """Create a RedisIndex without touching the network -- redis-py's client
    constructor is lazy, so this never connects. The cluster check (an INFO
    call) counts as done; TestClusterCheck covers it."""
    idx = RedisIndex(dimensions=DIMS, **kwargs)
    idx._standalone_checked = True
    return idx


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
# _tag
# ---------------------------------------------------------------------------


class TestTag:
    def test_is_hex(self):
        assert all(c in "0123456789abcdef" for c in _tag("weird route, name?"))

    def test_keeps_case_and_commas_apart(self):
        # a raw RediSearch TAG folds case and splits on commas
        assert _tag("Chat") != _tag("chat")
        assert _tag("a,b") != _tag("a")

    def test_non_str_input_coerced(self):
        assert _tag(123) == _tag("123")


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

    def test_euclidean_is_root_of_squared_distance(self):
        # Redis returns the squared L2 distance
        idx = make_index(metric=Metric.EUCLIDEAN)
        assert idx._distance_to_score(2.25) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------


class TestIds:
    def test_deterministic_without_namespace(self):
        idx = make_index()
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, json.dumps([None, "r", "utt"])))
        assert idx._id_for("r", "utt") == expected

    def test_deterministic_with_namespace(self):
        idx = make_index(namespace="org-abc")
        expected = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, json.dumps(["org-abc", "r", "utt"]))
        )
        assert idx._id_for("r", "utt") == expected

    def test_colons_dont_make_ids_collide(self):
        assert make_index()._id_for("a:b", "c") != make_index()._id_for("a", "b:c")
        assert make_index(namespace="a")._id_for("b:c", "d") != make_index(
            namespace="a:b"
        )._id_for("c", "d")

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
        assert idx._filter_query() == f"(@sr_namespace_tag:{{{_tag('org-1')}}})"

    def test_route_filter_term_joins_with_pipe(self):
        idx = make_index()
        term = idx._route_filter_term(["billing", "support"])
        assert term == f"@sr_route_tag:{{{_tag('billing')}|{_tag('support')}}}"

    def test_namespace_and_route_filter_combine(self):
        idx = make_index(namespace="org-1")
        term = idx._route_filter_term(["billing"])
        assert idx._filter_query([term]) == (
            f"(@sr_namespace_tag:{{{_tag('org-1')}}} @sr_route_tag:{{{_tag('billing')}}})"
        )


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
        assert mapping["sr_route_tag"] == _tag("r")
        assert isinstance(mapping["vector"], bytes)
        assert "namespace" not in mapping
        assert "sr_namespace_tag" not in mapping

        record = json.loads(mapping["sr_record"])
        assert record["sr_route"] == "r"
        assert record["sr_utterance"] == "utt"
        assert record["sr_function_schema"] == '{"name": "fn"}'
        assert record["foo"] == "bar"

    def test_mapping_includes_namespace_when_set(self):
        idx = make_index(namespace="org-1")
        mapping = idx._build_mapping("r", "utt", [0.1, 0.2, 0.3, 0.4], {}, None)
        assert mapping["namespace"] == "org-1"
        assert mapping["sr_namespace_tag"] == _tag("org-1")

    def test_no_function_schema_defaults_empty(self):
        idx = make_index()
        mapping = idx._build_mapping("r", "utt", [0.1, 0.2, 0.3, 0.4], {}, None)
        record = json.loads(mapping["sr_record"])
        assert record["sr_function_schema"] == "{}"


class TestKeyPrefix:
    def test_plain_name_is_the_prefix(self):
        assert make_index(index_name="routes")._key_prefix() == "routes:"

    def test_no_prefix_starts_another(self):
        # `routes:` must not be a prefix of `routes:v2`'s keys
        base = make_index(index_name="routes")._key_prefix()
        v2 = make_index(index_name="routes:v2")._key_prefix()
        assert not v2.startswith(base)


class TestWithoutRediSearch:
    def test_query_raises_instead_of_returning_empty(self, mocker):
        # plain Redis answers FT.INFO with "unknown command", which must not
        # read as "no index yet"
        from redis.exceptions import ResponseError

        idx = make_index()
        ft = mocker.patch.object(idx.client, "ft")
        ft.return_value.info.side_effect = ResponseError(
            "unknown command 'FT.INFO', with args beginning with: 'x' "
        )
        with pytest.raises(ResponseError, match="unknown command"):
            idx.query(vector=np.zeros(DIMS), top_k=1)

    def test_a_missing_index_still_reads_as_empty(self, mocker):
        from redis.exceptions import ResponseError

        idx = make_index()
        ft = mocker.patch.object(idx.client, "ft")
        ft.return_value.info.side_effect = ResponseError("Unknown index name")
        scores, routes = idx.query(vector=np.zeros(DIMS), top_k=1)
        assert routes == []


class TestMissingIndex:
    @pytest.mark.parametrize(
        "reply",
        [
            "Unknown index name",  # Redis Stack 7.x
            "Unknown Index name",  # older RediSearch 2.x
            "SEARCH_INDEX_NOT_FOUND Index not found: idx",  # Redis 8.x
            "idx: no such index",  # FT.SEARCH on Redis Stack 7.x
        ],
    )
    def test_missing_index_replies(self, reply):
        assert _missing_index(Exception(reply))

    @pytest.mark.parametrize(
        "reply",
        [
            "unknown command 'FT.INFO', with args beginning with: 'idx' ",
            "this user has no permissions to run the 'FT.INFO' command",
            "LOADING Redis is loading the dataset in memory",
            "unknown command 'FT.INFO', with args beginning with: 'no such index' ",
        ],
    )
    def test_other_errors_are_not_a_missing_index(self, reply):
        assert not _missing_index(Exception(reply))


SCANNING = {"indexing": 1, "num_docs": 0}
SCANNED = {"indexing": 0, "num_docs": 3}


class TestScanWait:
    """A new or reopened index is used only once its initial background scan
    is done (FT.INFO `indexing` is 0), and the wait is bounded."""

    def test_waits_for_the_scan_then_marks_the_handle_ready(self, mocker):
        idx = make_index(index_ready_timeout=5)
        info = mocker.patch.object(idx.client, "ft").return_value.info
        info.side_effect = [SCANNING, SCANNING, SCANNED]
        assert idx._ensure_index(create=False)
        assert info.call_count == 3
        assert idx._index_initialized

    def test_raises_when_the_scan_outlasts_the_timeout(self, mocker):
        idx = make_index(index_ready_timeout=0.1)
        info = mocker.patch.object(idx.client, "ft").return_value.info
        info.return_value = SCANNING
        with pytest.raises(TimeoutError, match="still indexing"):
            idx.query(vector=np.zeros(DIMS), top_k=1)
        assert not idx._index_initialized

    def test_async_wait_yields_to_the_event_loop(self, mocker):
        idx = make_index(index_ready_timeout=5)
        ft = mocker.patch.object(idx.aclient, "ft")
        ft.return_value.info = mocker.AsyncMock(side_effect=[SCANNING] * 4 + [SCANNED])
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        async def run() -> bool:
            task = asyncio.create_task(ticker())
            ready = await idx._aensure_index(create=False)
            task.cancel()
            return ready

        assert asyncio.run(run())
        assert ticks >= 4  # other tasks ran while it waited
        assert idx._index_initialized

    def test_async_raises_when_the_scan_outlasts_the_timeout(self, mocker):
        idx = make_index(index_ready_timeout=0.1)
        ft = mocker.patch.object(idx.aclient, "ft")
        ft.return_value.info = mocker.AsyncMock(return_value=SCANNING)
        with pytest.raises(TimeoutError, match="still indexing"):
            asyncio.run(idx.aquery(vector=np.zeros(DIMS), top_k=1))
        assert not idx._index_initialized


class TestIncompleteHits:
    def test_a_hit_without_its_score_raises(self, mocker):
        # Redis 8.2 with search workers can return a record without the
        # fields asked for; skipping it would return incomplete results
        from redis.commands.search.document import Document

        idx = make_index()
        ft = mocker.patch.object(idx.client, "ft").return_value
        ft.info.return_value = SCANNED
        ft.search.return_value.docs = [Document(id="k", sr_route="r")]
        with pytest.raises(RuntimeError, match="without its score"):
            idx.query(vector=np.zeros(DIMS), top_k=1)


class TestClusterCheck:
    """RedisIndex talks to one endpoint, so on an OSS cluster it raises a clear
    error before its first FT.INFO or config access, instead of MOVED errors
    mid-write (or, on Redis Stack 7.x, searches that see one shard)."""

    def _index(self, mocker, cluster_info):
        idx = RedisIndex(dimensions=DIMS)
        info = mocker.patch.object(idx.client, "info")
        if isinstance(cluster_info, Exception):
            info.side_effect = cluster_info
        else:
            info.return_value = cluster_info
        ft = mocker.patch.object(idx.client, "ft").return_value
        ft.info.return_value = SCANNED
        return idx, info, ft

    def test_a_cluster_raises_before_any_command(self, mocker):
        idx, info, ft = self._index(mocker, {"cluster_enabled": 1})
        with pytest.raises(ValueError, match="cluster mode"):
            idx.query(vector=np.zeros(DIMS), top_k=1)
        with pytest.raises(ValueError, match="cluster mode"):
            idx._read_hash()
        info.assert_called_with("cluster")
        ft.info.assert_not_called()

    def test_a_standalone_server_is_checked_once(self, mocker):
        idx, info, ft = self._index(mocker, {"cluster_enabled": 0})
        assert idx.is_ready()
        assert idx.is_ready()
        assert info.call_count == 1

    def test_an_acl_user_without_info_skips_the_check(self, mocker):
        from redis.exceptions import NoPermissionError

        idx, info, ft = self._index(
            mocker, NoPermissionError("this user has no permissions to run 'info'")
        )
        assert idx.is_ready()

    def test_async_a_cluster_raises(self, mocker):
        idx = RedisIndex(dimensions=DIMS)
        mocker.patch.object(
            idx.aclient, "info", mocker.AsyncMock(return_value={"cluster_enabled": 1})
        )
        with pytest.raises(ValueError, match="cluster mode"):
            asyncio.run(idx.aquery(vector=np.zeros(DIMS), top_k=1))
