"""Exercise async listing over HTTP without a Pinecone service or API key."""

from uuid import uuid4

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from semantic_router.encoders import DenseEncoder, TfidfEncoder
from semantic_router.index.pinecone import PineconeIndex
from semantic_router.route import Route
from semantic_router.routers import IndexedRouter


def record_metadata(id):
    return {"sr_route": "route", "sr_utterance": id}


@pytest_asyncio.fixture
async def serve_index(mocker, monkeypatch):
    # Only the unused sync client is mocked; list requests use real aiohttp.
    mocker.patch.object(PineconeIndex, "_initialize_client")
    monkeypatch.delenv("PINECONE_API_BASE_URL", raising=False)
    runners = []

    async def start(handler, fetch_handler=None):
        async def fetch(request):
            return web.json_response(
                {
                    "vectors": {
                        id: {"id": id, "metadata": record_metadata(id)}
                        for id in request.query.getall("ids", [])
                    }
                }
            )

        app = web.Application()
        app.router.add_get("/vectors/list", handler)
        app.router.add_get("/vectors/fetch", fetch_handler or fetch)
        runner = web.AppRunner(app)
        await runner.setup()
        runners.append(runner)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        return PineconeIndex(
            api_key="test",
            index_name=f"test-{uuid4().hex[:12]}",
            dimensions=4,
            host=url,
            base_url=url,
            init_async_index=True,
        )

    yield start
    for runner in runners:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
@pytest.mark.parametrize("after_first_page", [False, True])
@pytest.mark.parametrize("include_metadata", [False, True])
async def test_failed_list_request_raises(
    serve_index, status, after_first_page, include_metadata
):
    requests = []

    async def unavailable(request):
        requests.append(dict(request.query))
        if after_first_page and "paginationToken" not in request.query:
            return web.json_response(
                {"vectors": [{"id": "route#1"}], "pagination": {"next": "page2"}}
            )
        # Error bodies need not be JSON.
        return web.Response(text="unavailable", status=status)

    index = await serve_index(unavailable)
    with pytest.raises(aiohttp.ClientResponseError) as exc:
        await index._async_get_all(include_metadata=include_metadata)
    assert exc.value.status == status
    assert exc.value.request_info.real_url.path == "/vectors/list"
    assert requests == (
        [{}, {"paginationToken": "page2"}] if after_first_page else [{}]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("after_first_page", [False, True])
@pytest.mark.parametrize("include_metadata", [False, True])
async def test_list_not_found_returns_empty(
    serve_index, after_first_page, include_metadata
):
    async def missing(request):
        if after_first_page and "paginationToken" not in request.query:
            return web.json_response(
                {"vectors": [{"id": "route#1"}], "pagination": {"next": "page2"}}
            )
        return web.Response(text="not found", status=404)

    index = await serve_index(missing)
    assert await index._async_get_all(include_metadata=include_metadata) == ([], [])


@pytest.mark.asyncio
@pytest.mark.parametrize("include_metadata", [False, True])
async def test_list_pages_preserve_filters_and_metadata(serve_index, include_metadata):
    pages, fetches = [], []

    async def listing(request):
        pages.append(dict(request.query))
        assert request.headers["Api-Key"] == "test"
        if "paginationToken" not in request.query:
            return web.json_response(
                {
                    "vectors": [{"id": "route#1"}, {"id": "route#2"}],
                    "pagination": {"next": "page/2+=token"},
                }
            )
        return web.json_response({"vectors": [{"id": "route#3"}]})

    async def fetch(request):
        ids = request.query.getall("ids")
        fetches.append(ids)
        return web.json_response(
            {
                "vectors": {
                    id: {"id": id, "metadata": record_metadata(id)}
                    for id in ids
                    if id != "route#2"  # deleted between list and fetch
                }
            }
        )

    index = await serve_index(listing, fetch)
    index.namespace = "test-namespace"
    ids, metadata = await index._async_get_all(
        prefix="route#", include_metadata=include_metadata
    )
    expected_ids = (
        ["route#1", "route#3"]
        if include_metadata
        else ["route#1", "route#2", "route#3"]
    )
    assert ids == expected_ids
    assert metadata == (
        [record_metadata(id) for id in expected_ids] if include_metadata else []
    )
    filters = {"prefix": "route#", "namespace": "test-namespace"}
    assert pages == [filters, {**filters, "paginationToken": "page/2+=token"}]
    assert fetches == (
        [["route#1", "route#2"], ["route#3"]] if include_metadata else []
    )


@pytest.mark.asyncio
async def test_successful_empty_list_returns_empty(serve_index):
    async def empty(request):
        return web.json_response({"vectors": []})

    index = await serve_index(empty)
    assert await index._async_get_all(include_metadata=True) == ([], [])


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["adelete", "aget_routes", "arefresh", "aadd"])
@pytest.mark.parametrize("after_first_page", [False, True])
async def test_failed_list_stops_downstream_operations(
    serve_index, mocker, operation, after_first_page
):
    async def unavailable(request):
        if after_first_page and "paginationToken" not in request.query:
            return web.json_response(
                {"vectors": [{"id": "route#1"}], "pagination": {"next": "page2"}}
            )
        return web.Response(text="unavailable", status=503)

    index = await serve_index(unavailable)
    delete = mocker.patch.object(PineconeIndex, "_async_delete")
    drop = mocker.patch.object(PineconeIndex, "delete_index")
    add = mocker.patch.object(PineconeIndex, "add")
    stored = [Route(name="route", utterances=["stored utterance"])]
    router = None
    if operation in ("arefresh", "aadd"):
        # Construct with a known sync snapshot, then exercise the real async read.
        mocker.patch.object(PineconeIndex, "get_routes", return_value=stored)
        mocker.patch.object(DenseEncoder, "__call__", return_value=[[1.0] * 4])
        router = IndexedRouter(
            encoder=DenseEncoder(name="unused"),
            index=index,
            sparse_encoder=TfidfEncoder() if operation == "aadd" else None,
        )
    with pytest.raises(aiohttp.ClientResponseError) as exc:
        if operation == "adelete":
            await index.adelete("route")
        elif operation == "aget_routes":
            await index.aget_routes()
        elif operation == "arefresh":
            await router.arefresh()
        else:
            await router.aadd(Route(name="new", utterances=["new utterance"]))
    assert exc.value.status == 503
    delete.assert_not_called()
    drop.assert_not_called()
    add.assert_not_called()
    if router is not None:
        assert router.routes == stored
