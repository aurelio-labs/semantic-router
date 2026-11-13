"""Unit tests for PineconeIndex async delete — uses a local HTTP server."""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web

from semantic_router.index.pinecone import PineconeIndex


async def _serve(status: int, body: str):
    async def handler(request):
        return web.Response(status=status, text=body)

    app = web.Application()
    app.router.add_post("/vectors/delete", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _index(host: str) -> PineconeIndex:
    idx = PineconeIndex.model_construct(
        host=host,
        base_url="https://api.pinecone.io",
        headers={},
        namespace="",
    )
    return idx


def _patched():
    return (
        patch.object(PineconeIndex, "ais_ready", AsyncMock(return_value=True)),
        patch.object(
            PineconeIndex,
            "_async_get_route_ids",
            AsyncMock(return_value=["r#1", "r#2"]),
        ),
    )


@pytest.mark.asyncio
async def test_adelete_raises_when_the_delete_request_fails():
    runner, host = await _serve(503, "service unavailable")
    try:
        idx = _index(host)
        p1, p2 = _patched()
        with p1, p2:
            with pytest.raises(Exception, match="Failed to delete vectors: 503"):
                await idx.adelete("r")
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_adelete_returns_ids_on_success():
    runner, host = await _serve(200, "{}")
    try:
        idx = _index(host)
        p1, p2 = _patched()
        with p1, p2:
            assert await idx.adelete("r") == ["r#1", "r#2"]
    finally:
        await runner.cleanup()
