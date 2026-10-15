import json
from uuid import uuid4

import numpy as np
import pytest
import yaml

from semantic_router import (
    Chain,
    DecisionRouter,
    IndexedRouter,
    Merge,
    Route,
    load_router,
)
from semantic_router.calibration import Calibrator
from semantic_router.config import from_config, to_config
from semantic_router.encoders import DenseEncoder, OpenAIEncoder, TfidfEncoder
from semantic_router.index import LocalIndex, PostgresIndex, QdrantIndex
from semantic_router.schema import Metric

# each keyword owns an axis, so "refund" is near "charged" and far from "crash"
AXES = {"refund": 0, "charged": 0, "crash": 1, "error": 1}

# secrets reach the routers through the environment and through constructor args;
# none of them may end up in a file
SECRETS = {
    "OPENAI_API_KEY": "sk-env-SECRET",
    "JEV_API_KEY": "jev-env-SECRET",
    "POSTGRES_CONNECTION_STRING": "postgresql://u:pg-env-SECRET@localhost:5432/db",
}
PASSED_SECRETS = ["jev-arg-SECRET", "qd-arg-SECRET", "pg-arg-SECRET"]


def fake_embed(self, docs, truncate=True):
    out = []
    for d in docs:
        v = np.zeros(3)
        for w in str(d).lower().split():
            if w in AXES:
                v[AXES[w]] += 1
        if not v.any():
            v[:] = 1
        out.append(list(v / np.linalg.norm(v)))
    return out


async def fake_aembed(self, docs, truncate=True):
    return fake_embed(self, docs)


@pytest.fixture(autouse=True)
def offline(monkeypatch, mocker):
    # a real OpenAIEncoder, so the file names a type AutoEncoder can build, but no network
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    mocker.patch.object(OpenAIEncoder, "__call__", fake_embed)
    mocker.patch.object(OpenAIEncoder, "acall", fake_aembed)


ROUTES = [
    Route(
        name="billing",
        utterances=["refund please", "charged twice"],
        description="Money already paid or owed.",
        confidence_threshold=0.9,
        metadata={"team": "finance"},
    ),
    Route(name="technical", utterances=["app crash", "error on save"]),
]
CALIBRATOR = Calibrator(method="platt", params={"a": [14.2], "b": [-11.1]})


def indexed():
    return IndexedRouter(
        encoder=OpenAIEncoder(name="text-embedding-3-small"),
        routes=ROUTES,
        index=LocalIndex(),
        top_k=3,
        aggregation="max",
        confidence_threshold=0.5,
        calibrator=CALIBRATOR,
    )


def hybrid():
    return IndexedRouter(
        encoder=OpenAIEncoder(name="text-embedding-3-small"),
        sparse_encoder=TfidfEncoder(),
        routes=ROUTES,
        alpha=0.6,
    )


def decision():
    return DecisionRouter(
        routes=ROUTES,
        api_key="jev-arg-SECRET",
        base_url="https://example.test/",
        instructions="Which team should take this?",
        timeout=5.0,
        retries=1,
        confidence_threshold=0.8,
    )


def chain():
    return Chain(routers=[indexed(), decision()], confidence_threshold=0.7)


def merge():
    return Merge(routers=[indexed(), hybrid(), decision()], weights=[1.0, 1.0, 2.0])


def save(router, tmp_path, fmt):
    path = tmp_path / f"router.{fmt}"
    if fmt == "yaml":
        router.to_yaml(path=path)
    else:
        router.to_json(path=path)
    return path


@pytest.mark.parametrize("fmt", ["yaml", "json"])
@pytest.mark.parametrize("build", [indexed, hybrid, decision, chain, merge])
def test_round_trip(build, fmt, tmp_path):
    router = build()
    loaded = load_router(path=save(router, tmp_path, fmt))
    assert type(loaded) is type(router)
    assert to_config(loaded) == to_config(router)
    assert [r.to_dict() for r in loaded.routes] == [r.to_dict() for r in router.routes]


@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_file_says_which_router_it_is(fmt, tmp_path):
    text = save(chain(), tmp_path, fmt).read_text()
    data = yaml.safe_load(text) if fmt == "yaml" else json.loads(text)
    assert data["router_type"] == "chain"
    assert [r["router_type"] for r in data["routers"]] == ["indexed", "decision"]
    assert data["routers"][0]["encoder"] == {
        "type": "openai",
        "name": "text-embedding-3-small",
    }
    assert data["routers"][0]["index"] == {"type": "local"}
    assert data["routers"][1]["base_url"] == "https://example.test"


def test_loaded_indexed_router_routes_the_same(tmp_path):
    router = indexed()
    loaded = load_router(path=save(router, tmp_path, "yaml"))
    for q in ["i want a refund", "it keeps crashing with an error", "hello"]:
        assert loaded(q, limit=None) == router(q, limit=None)


def test_loaded_router_keeps_its_calibrator(tmp_path):
    loaded = load_router(path=save(indexed(), tmp_path, "json"))
    assert loaded.calibrator is not None
    assert loaded.calibrator.to_dict() == CALIBRATOR.to_dict()
    assert loaded("i want a refund").calibrated
    # no labelled queries came with the file, so a route change drops the calibrator
    loaded.add(Route(name="support", utterances=["call a human"]))
    assert loaded.calibrator is None


def test_fitted_chain_keeps_its_calibrator(tmp_path):
    router = chain()
    router.calibrator = CALIBRATOR
    loaded = load_router(path=save(router, tmp_path, "yaml"))
    assert loaded.calibrator is not None
    assert loaded.calibrator.to_dict() == CALIBRATOR.to_dict()


@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_no_secrets_in_the_file(fmt, tmp_path):
    # routes with no utterances, so the postgres router builds without a database
    described = [Route(name=r.name, description=r.name) for r in ROUTES]
    postgres = IndexedRouter(
        encoder=OpenAIEncoder(name="text-embedding-3-small"),
        routes=described,
        index=PostgresIndex(
            connection_string="postgresql://u:pg-arg-SECRET@localhost:5432/db",
            index_name=f"cfg-{uuid4().hex[:8]}",
        ),
    )
    qdrant = IndexedRouter(
        encoder=OpenAIEncoder(name="text-embedding-3-small"),
        routes=ROUTES,
        index=QdrantIndex(
            index_name=f"cfg-{uuid4().hex[:8]}", dimensions=3, api_key="qd-arg-SECRET"
        ),
    )
    text = save(
        Merge(routers=[postgres, qdrant, decision()]), tmp_path, fmt
    ).read_text()
    for secret in [*SECRETS.values(), *PASSED_SECRETS]:
        assert secret not in text
    assert "SECRET" not in text


def test_index_params_round_trip(tmp_path):
    name = f"cfg-{uuid4().hex[:8]}"
    router = IndexedRouter(
        encoder=OpenAIEncoder(name="text-embedding-3-small"),
        routes=[Route(name="billing", description="money")],
        index=PostgresIndex(
            index_name=name, namespace="prod", metric=Metric.DOTPRODUCT
        ),
    )
    loaded = load_router(path=save(router, tmp_path, "yaml"))
    assert isinstance(loaded, IndexedRouter)
    assert isinstance(loaded.index, PostgresIndex)
    assert loaded.index.index_name == name
    assert loaded.index.namespace == "prod"
    # the file holds "dotproduct"; the index needs the enum back
    assert loaded.index.metric is Metric.DOTPRODUCT
    # the connection string came from the environment, not the file
    assert loaded.index.connection_string == SECRETS["POSTGRES_CONNECTION_STRING"]


def test_a_0x_file_points_to_the_migration_guide(tmp_path):
    path = tmp_path / "router.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "encoder_type": "openai",
                "encoder_name": "text-embedding-3-small",
                "routes": [{"name": "billing", "utterances": ["refund please"]}],
            }
        )
    )
    with pytest.raises(ValueError, match="0.x") as e:
        load_router(path=path)
    assert "migration-to-v1" in str(e.value)


def test_a_key_in_the_index_config_is_refused():
    config = to_config(indexed()).model_dump(mode="json")
    config["index"] = {"type": "pinecone", "index_name": "x", "api_key": "pc-key"}
    with pytest.raises(ValueError, match="environment"):
        from_config(config=config)


def test_a_typo_in_the_file_is_an_error():
    config = to_config(decision()).model_dump(mode="json")
    config["treshold"] = 0.5
    with pytest.raises(ValueError, match="treshold"):
        from_config(config=config)


def test_an_encoder_a_file_cannot_rebuild_is_refused(tmp_path):
    class FakeEncoder(DenseEncoder):
        name: str = "fake"

        def __call__(self, docs):
            return fake_embed(self, docs)

    router = IndexedRouter(encoder=FakeEncoder(), routes=ROUTES)
    with pytest.raises(ValueError, match="AutoEncoder"):
        router.to_yaml(path=tmp_path / "router.yaml")


def test_unknown_extension(tmp_path):
    path = tmp_path / "router.toml"
    path.write_text("router_type = 'decision'")
    with pytest.raises(ValueError, match="use .yaml, .yml or .json"):
        load_router(path=path)


def test_auto_encoder_knows_the_local_type():
    # AutoEncoder had a LOCAL branch but EncoderType had no LOCAL, so "local" was refused
    from semantic_router.config import _ENCODERS
    from semantic_router.encoders import LocalEncoder
    from semantic_router.schema import EncoderType

    assert EncoderType("local") is EncoderType.LOCAL
    assert _ENCODERS[LocalEncoder] is EncoderType.LOCAL


def test_a_local_encoder_round_trips(tmp_path):
    pytest.importorskip("sentence_transformers")
    from semantic_router.encoders import LocalEncoder

    router = IndexedRouter(
        encoder=LocalEncoder(),
        routes=[Route(name="billing", utterances=["refund please"])],
    )
    path = tmp_path / "router.yaml"
    router.to_yaml(path=path)
    loaded = load_router(path=path)
    assert isinstance(loaded.encoder, LocalEncoder)
    assert loaded.encoder.name == router.encoder.name
