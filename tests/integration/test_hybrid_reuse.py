"""A hybrid router with a fitted sparse encoder (TF-IDF) on an index that already holds
routes: reopening, adding and removing keep every stored sparse row valid, never delete
the index, and leave it as it was when encoding fails.

The dense encoder is constant, so on HybridLocalIndex only the sparse side decides a
route. pinecone-local stores records but scores dense values only, so on Pinecone these
tests check the writes, not the routing.
"""

import pytest

from semantic_router.encoders import DenseEncoder, TfidfEncoder
from semantic_router.index import BaseIndex, HybridLocalIndex, PineconeIndex
from semantic_router.route import Route
from semantic_router.routers import IndexedRouter

ZEBRA = Route(name="existing", utterances=["zebra"])
APPLE = Route(name="new", utterances=["apple"])
MANGO = Route(name="fruit", utterances=["mango"])


class ConstantEncoder(DenseEncoder):
    name: str = "constant"
    fail: bool = False

    def __call__(self, docs):
        if self.fail:
            raise RuntimeError("encoder down")
        return [[1.0, 0.0] for _ in docs]


@pytest.fixture(params=[HybridLocalIndex, PineconeIndex], ids=lambda c: c.__name__)
def index_cls(request):
    return request.param


def reopen(index: BaseIndex, make_index, **kwargs) -> BaseIndex:
    """another handle on the same stored index, as a second process would open it"""
    if isinstance(index, PineconeIndex):
        return make_index(PineconeIndex, name=index.index_name, **kwargs)
    return index


def build(routes, index, encoder=None) -> IndexedRouter:
    return IndexedRouter(
        encoder=encoder or ConstantEncoder(),
        routes=routes,
        index=index,
        sparse_encoder=TfidfEncoder(),
        alpha=0.5,
        confidence_threshold=0.8,
    )


def stored(index: BaseIndex) -> list:
    return sorted(
        (u.route, str(u.utterance), str(u.metadata)) for u in index.get_utterances()
    )


def routes_sparse(router: IndexedRouter) -> bool:
    # only HybridLocalIndex scores sparse vectors locally
    return isinstance(router.index, HybridLocalIndex)


def test_new_routes_on_a_filled_index_reencode_the_stored_rows(index_cls, make_index):
    index = make_index(index_cls)
    build([ZEBRA], index)

    router = build([APPLE], reopen(index, make_index))

    assert {r.name for r in router.routes} == {"existing", "new"}
    assert len(index.get_utterances()) == 2
    if routes_sparse(router):
        # "zebra" was stored under the old vocabulary; a stale row would miss it
        assert router("zebra").name == "existing"
        assert router("apple").name == "new"


@pytest.mark.parametrize("routes", [[ZEBRA, APPLE], [APPLE], []], ids=len)
def test_reopening_with_routes_it_holds_writes_nothing(
    index_cls, make_index, mocker, routes
):
    index = make_index(index_cls)
    build([ZEBRA, APPLE], index)
    add = mocker.spy(index_cls, "add")
    delete = mocker.spy(index_cls, "delete")
    delete_index = mocker.spy(index_cls, "delete_index")

    router = build(routes, reopen(index, make_index))

    assert add.call_count == delete.call_count == delete_index.call_count == 0
    assert len(index.get_utterances()) == 2
    if routes_sparse(router):
        # the refit on the same corpus gives the vocabulary the rows were written with
        assert router("zebra").name == "existing"
        assert router("apple").name == "new"


def test_a_failed_reopen_leaves_the_index_as_it_was(index_cls, make_index, mocker):
    index = make_index(index_cls)
    build([ZEBRA], index)
    before = stored(index)
    delete_index = mocker.spy(index_cls, "delete_index")

    with pytest.raises(RuntimeError, match="encoder down"):
        build([APPLE], reopen(index, make_index), encoder=ConstantEncoder(fail=True))

    assert delete_index.call_count == 0
    assert stored(index) == before
    router = build([], reopen(index, make_index))
    if routes_sparse(router):
        assert router("zebra").name == "existing"


def test_a_failed_add_leaves_the_index_and_the_router_as_they_were(
    index_cls, make_index
):
    router = build([ZEBRA, APPLE], make_index(index_cls))
    before = stored(router.index)

    router.encoder.fail = True
    with pytest.raises(RuntimeError, match="encoder down"):
        router.add(routes=MANGO)
    router.encoder.fail = False

    assert stored(router.index) == before
    assert {r.name for r in router.routes} == {"existing", "new"}
    if routes_sparse(router):
        # the sparse model went back to the one the stored rows were encoded with
        assert router("zebra").name == "existing"
        assert router("apple").name == "new"


def test_add_and_remove_keep_the_sparse_rows_valid(make_index):
    index = make_index(HybridLocalIndex)
    router = build([ZEBRA], index)

    router.add(routes=[APPLE, MANGO])
    router.remove(name="fruit")

    assert router("zebra").name == "existing"
    assert router("apple").name == "new"
    # a new process refits on what is left: its vocabulary must match the stored rows
    reopened = build([], index)
    assert reopened("zebra").name == "existing"
    assert reopened("apple").name == "new"
    assert reopened("mango").name is None


def test_add_and_remove_leave_other_namespaces_alone(make_index):
    a = make_index(PineconeIndex)
    b = make_index(PineconeIndex, name=a.index_name, namespace="other")
    router = build([ZEBRA], a)
    build([MANGO], b)

    router.add(routes=APPLE)
    router.remove(name="new")

    # fresh handles: an old one can keep reading a deleted index's data-plane host
    other = make_index(PineconeIndex, name=a.index_name, namespace="other")
    assert [u.route for u in other.get_utterances()] == ["fruit"]
    same = make_index(PineconeIndex, name=a.index_name)
    assert [u.route for u in same.get_utterances()] == ["existing"]


@pytest.mark.parametrize("how", ["add", "reopen"])
def test_new_fields_on_part_of_a_route_reach_all_its_rows(index_cls, make_index, how):
    index = make_index(index_cls)
    old = Route(name="fruit", utterances=["apple", "mango"], description="old")
    router = build([ZEBRA, old], index)
    changed = Route(name="fruit", utterances=["mango"], description="new")

    if how == "add":
        router.add(routes=changed)
    else:
        router = build([changed], reopen(index, make_index))

    fruit = {r.name: r for r in router.routes}["fruit"]
    assert fruit.description == "new"
    assert sorted(fruit.utterances) == ["apple", "mango"]
    rows = [u for u in index.get_utterances() if u.route == "fruit"]
    assert {u.metadata.get("sr_description") for u in rows} == {"new"}


def test_a_failed_delete_on_remove_leaves_the_router_aligned(make_index, mocker):
    index = make_index(HybridLocalIndex)
    router = build([ZEBRA, APPLE, MANGO], index)
    before = stored(index)
    mocker.patch.object(HybridLocalIndex, "delete", side_effect=RuntimeError("down"))

    with pytest.raises(RuntimeError, match="down"):
        router.remove(name="fruit")

    assert stored(index) == before
    # the sparse model still matches the rows the index holds
    assert router("zebra").name == "existing"
    assert router("apple").name == "new"
    assert router("mango").name == "fruit"


def test_retrying_a_remove_whose_write_failed_repairs_the_index(
    index_cls, make_index, mocker
):
    index = make_index(index_cls)
    router = build([ZEBRA, APPLE], index)
    add = mocker.patch.object(index_cls, "add", side_effect=RuntimeError("down"))

    with pytest.raises(RuntimeError, match="down"):
        router.remove(name="new")
    # the delete landed, so a retry finds the route gone but still rewrites the rest
    assert [u.route for u in index.get_utterances()] == ["existing"]
    add.side_effect = None
    add.reset_mock()
    mocker.stopall()
    rewrite = mocker.spy(index_cls, "add")
    router.remove(name="new")

    assert rewrite.call_count == 1
    assert [u.route for u in index.get_utterances()] == ["existing"]
    fresh = build([], reopen(index, make_index))
    assert [r.name for r in fresh.routes] == ["existing"]
    if routes_sparse(fresh):
        assert fresh("zebra").name == "existing"
        assert router("zebra").name == "existing"


@pytest.mark.parametrize("recovery", ["retry", "reopen"])
def test_recovering_from_a_partial_add_repairs_the_index(
    index_cls, make_index, mocker, recovery
):
    index = make_index(index_cls)
    router = build([ZEBRA], index)
    real_add = index_cls.add

    def partial(self, **records):
        # the new route's row lands, then the backend goes down before the rest
        keep = [i for i, r in enumerate(records["routes"]) if r == "new"]
        real_add(self, **{k: [v[i] for i in keep] for k, v in records.items()})
        raise RuntimeError("down")

    mocker.patch.object(index_cls, "add", partial)

    with pytest.raises(RuntimeError, match="down"):
        router.add(routes=APPLE)
    assert {u.route for u in index.get_utterances()} == {"existing", "new"}
    mocker.stopall()
    rewrite = mocker.spy(index_cls, "add")
    delete = mocker.spy(index_cls, "delete")
    if recovery == "retry":
        router.add(routes=APPLE)
    else:
        router = build([], reopen(index, make_index))

    # every row is written again, not only the new route's
    assert rewrite.call_count == 1
    assert sorted(rewrite.call_args.kwargs["routes"]) == ["existing", "new"]
    assert delete.call_count == 0
    fresh = build([], reopen(index, make_index))
    assert {r.name for r in fresh.routes} == {"existing", "new"}
    if routes_sparse(fresh):
        assert fresh("zebra").name == "existing"
        assert fresh("apple").name == "new"


def unfingerprinted(mocker):
    """rows as an index written before fit fingerprints existed holds them"""
    import semantic_router.routers.indexed as indexed

    plain = indexed._record_metadata
    mocker.patch.object(
        indexed, "_record_metadata", lambda route, sparse_fit=None: plain(route)
    )


def test_reopening_an_index_written_before_fingerprints_makes_no_writes(
    index_cls, make_index, mocker
):
    index = make_index(index_cls)
    unfingerprinted(mocker)
    build([ZEBRA, APPLE], index)
    mocker.stopall()
    add = mocker.spy(index_cls, "add")

    router = build([ZEBRA], reopen(index, make_index))

    assert add.call_count == 0
    assert {r.name for r in router.routes} == {"existing", "new"}


def test_growing_an_index_written_before_fingerprints_stamps_every_row(
    index_cls, make_index, mocker
):
    index = make_index(index_cls)
    unfingerprinted(mocker)
    build([ZEBRA], index)
    mocker.stopall()

    build([APPLE], reopen(index, make_index))
    add = mocker.spy(index_cls, "add")
    router = build([], reopen(index, make_index))

    marks = {u.metadata.get("sr_sparse_fit") for u in index.get_utterances()}
    assert len(marks) == 1 and None not in marks
    assert add.call_count == 0
    if routes_sparse(router):
        assert router("zebra").name == "existing"
        assert router("apple").name == "new"
