"""A router on a remote index saves to a file that names the index and holds no routes;
loading it reads the routes back from the index, the single source of truth."""

import pytest
import yaml

from semantic_router import IndexedRouter, Route, load_router
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.index import PineconeIndex, PostgresIndex, QdrantIndex

pytest.importorskip("fastembed")

# a class AutoEncoder can rebuild from the file, and needs no API key
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ROUTES = [
    Route(
        name="politics",
        utterances=[
            "isn't politics the best thing ever",
            "don't you love the president",
        ],
        description="Opinions about politics.",
        confidence_threshold=0.1,
        metadata={"team": "trust-and-safety"},
    ),
    Route(name="chitchat", utterances=["how's the weather today?", "lovely weather"]),
]


@pytest.fixture(scope="module")
def encoder() -> FastEmbedEncoder:
    return FastEmbedEncoder(name=MODEL)


@pytest.mark.parametrize(
    "index_cls", [PineconeIndex, PostgresIndex, QdrantIndex], ids=lambda c: c.__name__
)
def test_a_remote_router_file_reads_its_routes_from_the_index(
    index_cls, make_index, encoder, tmp_path
):
    router = IndexedRouter(encoder=encoder, routes=ROUTES, index=make_index(index_cls))
    path = tmp_path / "router.yaml"
    router.to_yaml(path=path)

    data = yaml.safe_load(path.read_text())
    assert "routes" not in data
    assert data["index"]["index_name"] == router.index.index_name

    loaded = load_router(path=path)
    assert isinstance(loaded, IndexedRouter)
    by_name = {r.name: r.to_dict() for r in loaded.routes}
    assert by_name == {r.name: r.to_dict() for r in router.routes}
    assert by_name["politics"]["description"] == "Opinions about politics."
    assert by_name["politics"]["confidence_threshold"] == 0.1
    assert loaded("don't you love politics?").name == "politics"
