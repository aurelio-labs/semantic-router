"""save a router to a yaml or json file and load it back. `router_type` picks the shape.

keys and secrets are never written: a loaded router reads them from the environment
(`OPENAI_API_KEY`, `JEV_API_KEY`, `PINECONE_API_KEY`...), as it would if built in code.
a decision router's model is saved like an index: its type, and its non-default params.

an indexed router on a remote index (pinecone, postgres, a qdrant server) is saved without
its routes: the index holds them and stays the single source of truth, so the file only
names the index. an in-memory index keeps nothing between runs, so its file carries routes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_serializer,
    model_validator,
)

from semantic_router.calibration import Calibrator, Method
from semantic_router.decision_models import (
    DecisionModel,
    GLiNER2DecisionModel,
    NaceDecisionModel,
    SystemOneDecisionModel,
    TypeSafeDecisionModel,
)
from semantic_router.encoders import (
    AurelioSparseEncoder,
    AutoEncoder,
    AzureOpenAIEncoder,
    BedrockEncoder,
    BM25Encoder,
    CLIPEncoder,
    CohereEncoder,
    DenseEncoder,
    FastEmbedEncoder,
    GoogleEncoder,
    HuggingFaceEncoder,
    JinaEncoder,
    LiteLLMEncoder,
    LocalEncoder,
    MistralEncoder,
    NimEncoder,
    OllamaEncoder,
    OpenAIEncoder,
    SparseEncoder,
    TfidfEncoder,
    VitEncoder,
    VoyageEncoder,
)
from semantic_router.index import (
    BaseIndex,
    HybridLocalIndex,
    LocalIndex,
    MongoDbIndex,
    PineconeIndex,
    PostgresIndex,
    QdrantIndex,
)
from semantic_router.route import Route
from semantic_router.routers import (
    BaseRouter,
    Chain,
    DecisionRouter,
    IndexedRouter,
    Merge,
)
from semantic_router.routers.indexed import _AGG, Aggregation
from semantic_router.schema import EncoderType

MIGRATION_GUIDE = (
    "https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1"
)

# a file rebuilds encoders with AutoEncoder, so these are the classes it can save. matched
# on the exact class: a subclass may not be what AutoEncoder builds back
_ENCODERS: dict[type, EncoderType] = {
    OpenAIEncoder: EncoderType.OPENAI,
    AzureOpenAIEncoder: EncoderType.AZURE,
    CohereEncoder: EncoderType.COHERE,
    AurelioSparseEncoder: EncoderType.AURELIO,
    BM25Encoder: EncoderType.BM25,
    TfidfEncoder: EncoderType.TFIDF,
    FastEmbedEncoder: EncoderType.FASTEMBED,
    HuggingFaceEncoder: EncoderType.HUGGINGFACE,
    MistralEncoder: EncoderType.MISTRAL,
    VoyageEncoder: EncoderType.VOYAGE,
    JinaEncoder: EncoderType.JINA,
    NimEncoder: EncoderType.NIM,
    VitEncoder: EncoderType.VIT,
    CLIPEncoder: EncoderType.CLIP,
    GoogleEncoder: EncoderType.GOOGLE,
    BedrockEncoder: EncoderType.BEDROCK,
    LiteLLMEncoder: EncoderType.LITELLM,
    OllamaEncoder: EncoderType.OLLAMA,
    LocalEncoder: EncoderType.LOCAL,
}

# index type -> its class and the constructor params a file may hold. secrets (api keys,
# connection strings) are left out on purpose: the index reads them from the environment
_INDEXES: dict[str, tuple[type[BaseIndex], tuple[str, ...]]] = {
    "local": (LocalIndex, ()),
    "hybrid_local": (HybridLocalIndex, ()),
    "pinecone": (
        PineconeIndex,
        (
            "index_name",
            "namespace",
            "dimensions",
            "metric",
            "cloud",
            "region",
            "base_url",
        ),
    ),
    "postgres": (
        PostgresIndex,
        ("index_name", "index_prefix", "namespace", "dimensions", "metric"),
    ),
    "mongodb": (
        MongoDbIndex,
        (
            "index_name",
            "database_name",
            "vector_index_name",
            "namespace",
            "dimensions",
            "metric",
            "sync_timeout",
        ),
    ),
    "qdrant": (
        QdrantIndex,
        (
            "index_name",
            "namespace",
            "dimensions",
            "metric",
            "location",
            "url",
            "host",
            "port",
            "grpc_port",
            "prefer_grpc",
            "https",
            "prefix",
            "timeout",
            "path",
            "config",
        ),
    ),
}

# decision model type -> its class and the params a file may hold. keys are left out:
# the model reads them from the environment
_SYSTEMONE_FIELDS = ("name", "base_url", "timeout", "retries", "concurrency")
_DECISION_MODELS: dict[str, tuple[type[DecisionModel], tuple[str, ...]]] = {
    "gliner2": (GLiNER2DecisionModel, ("name", "device", "batch_size")),
    "typesafe": (TypeSafeDecisionModel, _SYSTEMONE_FIELDS),
    "nace": (NaceDecisionModel, _SYSTEMONE_FIELDS),
    "systemone": (SystemOneDecisionModel, _SYSTEMONE_FIELDS),
}


class _Strict(BaseModel):
    # a typo in a hand-written file is an error, not a silently ignored key
    model_config = ConfigDict(extra="forbid")


class EncoderConfig(_Strict):
    type: str
    name: str | None = None


class IndexConfig(BaseModel):
    # the params depend on the type, so they ride as extra fields and are checked on build
    model_config = ConfigDict(extra="allow")

    type: str = "local"


class CalibratorConfig(_Strict):
    method: Method
    params: dict[str, list[float]]


class IndexedConfig(_Strict):
    router_type: Literal["indexed"] = "indexed"
    encoder: EncoderConfig
    sparse_encoder: EncoderConfig | None = None
    alpha: float = 0.7
    top_k: int = 5
    aggregation: Aggregation = "mean"
    index: IndexConfig = IndexConfig()
    confidence_threshold: float | None = None
    calibrator: CalibratorConfig | None = None
    refresh_interval: float | None = None
    # None when the index holds the routes; a file then has no routes key at all
    routes: list[Route] | None = None

    @model_serializer(mode="wrap")
    def _leave_out_index_routes(self, handler):
        data = handler(self)
        if self.routes is None:
            data.pop("routes", None)
        return data


class DecisionModelConfig(BaseModel):
    # the params depend on the type, so they ride as extra fields and are checked on build
    model_config = ConfigDict(extra="allow")

    type: str
    name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _not_a_name(cls, data: Any) -> Any:
        if isinstance(data, str):
            # 1.0.0 wrote the model as a bare name, with the HTTP params beside it
            raise ValueError(
                f"model is {data!r}, a bare model name as semantic-router 1.0.0 wrote it. "
                "write a block instead, e.g. model: {type: typesafe, name: jev-latest} "
                "or model: {type: gliner2}"
            )
        return data


class DecisionConfig(_Strict):
    router_type: Literal["decision"] = "decision"
    model: DecisionModelConfig
    instructions: str = "Which route does the user's message belong to?"
    confidence_threshold: float | None = None
    calibrator: CalibratorConfig | None = None
    routes: list[Route]


class ChainConfig(_Strict):
    router_type: Literal["chain"] = "chain"
    confidence_threshold: float | None = None
    # a chain or merge can be fitted like any router, so its calibrator is kept too
    calibrator: CalibratorConfig | None = None
    routers: list[RouterConfig]


class MergeConfig(_Strict):
    router_type: Literal["merge"] = "merge"
    confidence_threshold: float | None = None
    calibrator: CalibratorConfig | None = None
    weights: list[float] | None = None
    routers: list[RouterConfig]


RouterConfig = Annotated[
    Union[IndexedConfig, DecisionConfig, ChainConfig, MergeConfig],
    Field(discriminator="router_type"),
]
ChainConfig.model_rebuild()
MergeConfig.model_rebuild()
_ADAPTER: TypeAdapter[RouterConfig] = TypeAdapter(RouterConfig)

Format = Literal["yaml", "json"]


def to_config(router: BaseRouter) -> RouterConfig:
    """describe a router as a config, the shape a file holds. keys and secrets are left out."""
    calibrator = (
        CalibratorConfig.model_validate(router.calibrator.to_dict())
        if router.calibrator is not None
        else None
    )
    if isinstance(router, IndexedRouter):
        return IndexedConfig(
            encoder=_encoder_config(router.encoder),
            sparse_encoder=(
                _encoder_config(router.sparse_encoder)
                if router.sparse_encoder is not None
                else None
            ),
            alpha=router.alpha,
            top_k=router.top_k,
            aggregation=_aggregation(router),
            index=_index_config(router.index),
            confidence_threshold=router.confidence_threshold,
            calibrator=calibrator,
            refresh_interval=router.refresh_interval,
            # read the routes only from an index that keeps nothing between runs
            routes=None if _persists(router.index) else _routes(router),
        )
    if isinstance(router, DecisionRouter):
        return DecisionConfig(
            model=_decision_model_config(router.model),
            instructions=router.instructions,
            confidence_threshold=router.confidence_threshold,
            calibrator=calibrator,
            routes=_routes(router),
        )
    if isinstance(router, Chain):
        return ChainConfig(
            routers=[to_config(r) for r in router.routers],
            confidence_threshold=router.confidence_threshold,
            calibrator=calibrator,
        )
    if isinstance(router, Merge):
        return MergeConfig(
            routers=[to_config(r) for r in router.routers],
            weights=router.weights,
            confidence_threshold=router.confidence_threshold,
            calibrator=calibrator,
        )
    raise ValueError(f"{type(router).__name__} can't be saved to a file")


def from_config(config: RouterConfig | dict[str, Any]) -> BaseRouter:
    """build the router a config describes. keys and secrets come from the environment."""
    cfg = _parse(config) if isinstance(config, dict) else config
    calibrator = (
        Calibrator(method=cfg.calibrator.method, params=cfg.calibrator.params)
        if cfg.calibrator is not None
        else None
    )
    if isinstance(cfg, IndexedConfig):
        encoder = _build_encoder(cfg.encoder)
        if not isinstance(encoder, DenseEncoder):
            raise ValueError(
                f"encoder {cfg.encoder.type!r} is sparse: put it under sparse_encoder"
            )
        sparse: SparseEncoder | None = None
        if cfg.sparse_encoder is not None:
            built = _build_encoder(cfg.sparse_encoder)
            if not isinstance(built, SparseEncoder):
                raise ValueError(
                    f"sparse_encoder {cfg.sparse_encoder.type!r} is dense: put it under encoder"
                )
            sparse = built
        return IndexedRouter(
            encoder=encoder,
            # routes in the file are upserted; with none, they're read from the index
            routes=cfg.routes or (),
            index=_build_index(cfg.index),
            sparse_encoder=sparse,
            alpha=cfg.alpha,
            top_k=cfg.top_k,
            aggregation=cfg.aggregation,
            confidence_threshold=cfg.confidence_threshold,
            calibrator=calibrator,
            refresh_interval=cfg.refresh_interval,
        )
    if isinstance(cfg, DecisionConfig):
        return DecisionRouter(
            model=_build_decision_model(cfg.model),
            routes=cfg.routes,
            instructions=cfg.instructions,
            confidence_threshold=cfg.confidence_threshold,
            calibrator=calibrator,
        )
    if isinstance(cfg, ChainConfig):
        chain = Chain(
            routers=[from_config(r) for r in cfg.routers],
            confidence_threshold=cfg.confidence_threshold,
        )
        chain.calibrator = calibrator
        return chain
    merge = Merge(
        routers=[from_config(r) for r in cfg.routers],
        weights=cfg.weights,
        confidence_threshold=cfg.confidence_threshold,
    )
    merge.calibrator = calibrator
    return merge


def save_router(
    router: BaseRouter, path: str | Path, format: Format | None = None
) -> None:
    """write a router to a file; the format follows the extension unless given."""
    p = Path(path)
    data = _ADAPTER.dump_python(to_config(router), mode="json")
    if (format or _format(p)) == "yaml":
        p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    else:
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_router(path: str | Path) -> BaseRouter:
    """build a router from a yaml or json file written by `to_yaml` or `to_json`.

    the file's `router_type` picks the class: IndexedRouter, DecisionRouter, Chain or Merge.
    """
    p = Path(path)
    fmt = _format(p)
    text = p.read_text()
    data = yaml.safe_load(text) if fmt == "yaml" else json.loads(text)
    return from_config(data)


def _format(path: Path) -> Format:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return "yaml"
    if suffix == ".json":
        return "json"
    raise ValueError(f"can't tell the format of {path}: use .yaml, .yml or .json")


def _parse(data: object) -> RouterConfig:
    if not isinstance(data, dict):
        raise ValueError("a router config is a mapping with a router_type field")
    if "router_type" not in data:
        # 0.x files hold encoder_type / encoder_name / routes, with no router_type
        raise ValueError(
            "this config has no router_type, so it was written by semantic-router 0.x, "
            "which v1 can't load. build the router in code and save it again with "
            f"to_yaml or to_json; the migration guide shows how: {MIGRATION_GUIDE}"
        )
    return _ADAPTER.validate_python(data)


def _persists(index: BaseIndex) -> bool:
    """whether the index keeps its routes between runs, so a file needn't carry them."""
    if isinstance(index, LocalIndex):  # HybridLocalIndex too
        return False
    if isinstance(index, QdrantIndex):
        # qdrant's default location is an in-memory store
        return index.location != ":memory:"
    return True


def _routes(router: BaseRouter) -> list[Route]:
    # copies, so a config never shares mutable routes with the router it came from
    return [Route.from_dict(r.to_dict()) for r in router.routes]


def _aggregation(router: IndexedRouter) -> Aggregation:
    for name, fn in _AGG.items():
        if router.aggregate is fn:
            return name  # type: ignore[return-value]
    raise ValueError("a router with a custom aggregate function can't be saved")


def _encoder_config(encoder: DenseEncoder | SparseEncoder) -> EncoderConfig:
    kind = _ENCODERS.get(type(encoder))
    if kind is None:
        supported = sorted(cls.__name__ for cls in _ENCODERS)
        raise ValueError(
            f"{type(encoder).__name__} can't be saved to a file. a file rebuilds its "
            f"encoder with AutoEncoder, which builds: {', '.join(supported)}"
        )
    return EncoderConfig(type=kind.value, name=encoder.name)


def _build_encoder(cfg: EncoderConfig) -> DenseEncoder | SparseEncoder:
    return AutoEncoder(type=cfg.type, name=cfg.name).model


def _index_config(index: BaseIndex) -> IndexConfig:
    for kind, (cls, fields) in _INDEXES.items():
        if type(index) is cls:
            # only what differs from the default, so the file stays short and readable
            params = index.model_dump(
                mode="json", include=set(fields), exclude_defaults=True
            )
            return IndexConfig(type=kind, **params)
    raise ValueError(
        f"{type(index).__name__} can't be saved to a file; "
        f"supported indexes: {', '.join(_INDEXES)}"
    )


def _decision_model_config(model: DecisionModel) -> DecisionModelConfig:
    for kind, (cls, fields) in _DECISION_MODELS.items():
        if type(model) is cls:
            # the name always, so a file says which model it runs; the rest when changed
            params = model.model_dump(
                mode="json", include=set(fields) - {"name"}, exclude_defaults=True
            )
            return DecisionModelConfig(type=kind, name=model.name, **params)
    raise ValueError(
        f"{type(model).__name__} can't be saved to a file; "
        f"supported decision models: {', '.join(_DECISION_MODELS)}"
    )


def _build_decision_model(cfg: DecisionModelConfig) -> DecisionModel:
    if cfg.type not in _DECISION_MODELS:
        raise ValueError(
            f"unknown decision model type {cfg.type!r}; "
            f"supported: {', '.join(_DECISION_MODELS)}"
        )
    cls, fields = _DECISION_MODELS[cfg.type]
    params = dict(cfg.model_extra or {})
    if cfg.name is not None:
        params["name"] = cfg.name
    if unknown := sorted(set(params) - set(fields)):
        raise ValueError(
            f"decision model type {cfg.type!r} doesn't take {unknown} from a file "
            f"(it takes {list(fields)}); keys come from the environment"
        )
    return cls(**params)


def _build_index(cfg: IndexConfig) -> BaseIndex:
    if cfg.type not in _INDEXES:
        raise ValueError(
            f"unknown index type {cfg.type!r}; supported: {', '.join(_INDEXES)}"
        )
    cls, fields = _INDEXES[cfg.type]
    params = cfg.model_extra or {}
    if unknown := sorted(set(params) - set(fields)):
        raise ValueError(
            f"index type {cfg.type!r} doesn't take {unknown} from a file "
            f"(it takes {list(fields)}); keys and secrets come from the environment"
        )
    # a file holds plain values; coerce them to the field types (e.g. the Metric enum)
    kwargs: dict[str, Any] = {
        k: TypeAdapter(cls.model_fields[k].annotation).validate_python(v)
        for k, v in params.items()
    }
    return cls(**kwargs)
