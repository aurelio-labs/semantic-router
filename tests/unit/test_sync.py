import asyncio
import importlib
import os
import time
from datetime import datetime
from functools import wraps
from platform import python_version
from typing import Optional

import pytest

from semantic_router.encoders import CohereEncoder, DenseEncoder, OpenAIEncoder
from semantic_router.index import (
    HybridLocalIndex,
    LocalIndex,
    PineconeIndex,
    PostgresIndex,
    QdrantIndex,
)
from semantic_router.route import Route
from semantic_router.routers import HybridRouter, SemanticRouter
from semantic_router.schema import Utterance
from semantic_router.utils.logger import logger

PINECONE_SLEEP = 6
RETRY_COUNT = 5


# retry decorator for PineconeIndex cases (which need delay)
def retry(max_retries: int = 5, delay: int = 8):
    """Retry decorator, currently used for PineconeIndex which often needs some time
    to be populated and have all correct data. Once full Pinecone mock is built we
    should remove this decorator.

    :param max_retries: Maximum number of retries.
    :param delay: Delay between retries in seconds.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            count = 0
            last_exception = None
            while count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {count} | Error in {func.__name__}: {e}")
                    last_exception = e
                    count += 1
                    time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


# retry decorator for PineconeIndex cases (which need delay)
def async_retry(max_retries: int = 5, delay: int = 8):
    """Retry decorator, currently used for PineconeIndex which often needs some time
    to be populated and have all correct data. Once full Pinecone mock is built we
    should remove this decorator.

    :param max_retries: Maximum number of retries.
    :param delay: Delay between retries in seconds.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            count = 0
            last_exception = None
            while count < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {count} | Error in {func.__name__}: {e}")
                    last_exception = e
                    count += 1
                    await asyncio.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


def mock_encoder_call(utterances):
    # Define a mapping of utterances to return values
    mock_responses = {
        "Hello": [0.1, 0.2, 0.3],
        "Hi": [0.4, 0.5, 0.6],
        "Goodbye": [0.7, 0.8, 0.9],
        "Bye": [1.0, 1.1, 1.2],
        "Au revoir": [1.3, 1.4, 1.5],
        "Asparagus": [-2.0, 1.0, 0.0],
    }
    return [mock_responses.get(u, [0.3, 0.1, 0.2]) for u in utterances]


TEST_ID = (
    f"{python_version().replace('.', '')}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
)


def init_index(
    index_cls,
    dimensions: Optional[int] = 3,
    namespace: Optional[str] = "",
    init_async_index: bool = False,
    index_name: Optional[str] = None,
):
    """We use this function to initialize indexes with different names to avoid
    issues during testing.
    """
    if index_cls is PineconeIndex:
        if index_name:
            if not dimensions and "OpenAIEncoder" in index_name:
                dimensions = 1536

            elif not dimensions and "CohereEncoder" in index_name:
                dimensions = 1024

        index_name = TEST_ID if not index_name else f"{TEST_ID}-{index_name.lower()}"
        index = index_cls(
            index_name=index_name,
            dimensions=dimensions,
            namespace=namespace,
            init_async_index=init_async_index,
        )
    else:
        index = index_cls()
    return index


def layer_json():
    return """{
    "encoder_type": "cohere",
    "encoder_name": "embed-english-v3.0",
    "routes": [
        {
            "name": "politics",
            "utterances": [
                "isn't politics the best thing ever",
                "why don't you tell me about your political opinions"
            ],
            "description": null,
            "function_schemas": null
        },
        {
            "name": "chitchat",
            "utterances": [
                "how's the weather today?",
                "how are things going?"
            ],
            "description": null,
            "function_schemas": null
        }
    ]
}"""


def layer_yaml():
    return """encoder_name: embed-english-v3.0
encoder_type: cohere
routes:
- description: null
  function_schemas: null
  name: politics
  utterances:
  - isn't politics the best thing ever
  - why don't you tell me about your political opinions
- description: null
  function_schemas: null
  name: chitchat
  utterances:
  - how's the weather today?
  - how are things going?
    """


# not all indexes support metadata, so we map the feature here
INCLUDE_METADATA_MAP = {
    PineconeIndex: True,
    HybridLocalIndex: False,
    LocalIndex: False,
    QdrantIndex: False,
    PostgresIndex: False,
}


def include_metadata(index_cls):
    return INCLUDE_METADATA_MAP.get(index_cls, False)


MERGE_FORCE_LOCAL_RESULT_WITH_METADATA = [
    Utterance(route="Route 1", utterance="Hello"),
    Utterance(route="Route 1", utterance="Hi"),
    Utterance(route="Route 2", utterance="Au revoir"),
    Utterance(route="Route 2", utterance="Bye"),
    Utterance(route="Route 2", utterance="Goodbye"),
    Utterance(route="Route 2", utterance="Hi"),
]

MERGE_FORCE_LOCAL_RESULT_WITHOUT_METADATA = [
    Utterance(route="Route 1", utterance="Hello"),
    Utterance(route="Route 1", utterance="Hi"),
    Utterance(route="Route 2", utterance="Au revoir"),
    Utterance(route="Route 2", utterance="Bye"),
    Utterance(route="Route 2", utterance="Goodbye"),
    Utterance(route="Route 2", utterance="Hi"),
]


@pytest.fixture
def base_encoder():
    return DenseEncoder(name="test-encoder", score_threshold=0.5)


@pytest.fixture
def cohere_encoder(mocker):
    mocker.patch.object(CohereEncoder, "__call__", side_effect=mock_encoder_call)

    # Mock async call
    async def async_mock_encoder_call(docs=None, utterances=None):
        # Handle either docs or utterances parameter
        texts = docs if docs is not None else utterances
        return mock_encoder_call(texts)

    mocker.patch.object(CohereEncoder, "acall", side_effect=async_mock_encoder_call)
    return CohereEncoder(name="test-cohere-encoder", cohere_api_key="test_api_key")


@pytest.fixture
def openai_encoder(mocker):
    mocker.patch.object(OpenAIEncoder, "__call__", side_effect=mock_encoder_call)

    # Mock async call
    async def async_mock_encoder_call(docs=None, utterances=None):
        # Handle either docs or utterances parameter
        texts = docs if docs is not None else utterances
        return mock_encoder_call(texts)

    mocker.patch.object(OpenAIEncoder, "acall", side_effect=async_mock_encoder_call)
    return OpenAIEncoder(name="text-embedding-3-small", openai_api_key="test_api_key")


@pytest.fixture
def routes():
    return [
        Route(name="Route 1", utterances=["Hello", "Hi"], metadata={"type": "default"}),
        Route(name="Route 2", utterances=["Goodbye", "Bye", "Au revoir"]),
        Route(name="Route 3", utterances=["Boo"]),
    ]


@pytest.fixture
def routes_2():
    return [
        Route(name="Route 1", utterances=["Hello"]),
        Route(name="Route 2", utterances=["Hi"]),
    ]


@pytest.fixture
def routes_3():
    return [
        Route(name="Route 1", utterances=["Hello"]),
        Route(name="Route 2", utterances=["Asparagus"]),
    ]


@pytest.fixture
def routes_4():
    return [
        Route(name="Route 1", utterances=["Goodbye"], metadata={"type": "default"}),
        Route(name="Route 2", utterances=["Asparagus"]),
    ]


@pytest.fixture
def dynamic_routes():
    return [
        Route(
            name="Route 1",
            utterances=["Hello", "Hi"],
            function_schemas=[{"name": "test"}],
        ),
        Route(
            name="Route 2",
            utterances=["Goodbye", "Bye", "Au revoir"],
            function_schemas=[{"name": "test"}],
        ),
    ]


@pytest.fixture
def test_data():
    return [
        ("What's your opinion on the current government?", "politics"),
        ("what's the weather like today?", "chitchat"),
        ("what is the Pythagorean theorem?", "mathematics"),
        ("what is photosynthesis?", "biology"),
        ("tell me an interesting fact", None),
    ]


def get_test_indexes():
    indexes = []

    # if importlib.util.find_spec("qdrant_client") is not None:
    #    indexes.append(QdrantIndex)
    if importlib.util.find_spec("pinecone") is not None:
        indexes.append(PineconeIndex)

    return indexes


def get_test_routers():
    routers = [SemanticRouter, HybridRouter]
    return routers


@pytest.mark.parametrize(
    "index_cls,router_cls",
    [(index, router) for index in get_test_indexes() for router in get_test_routers()],
)
class TestSemanticRouter:
    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_initialization(self, openai_encoder, routes, index_cls, router_cls):
        index = init_index(index_cls, index_name=router_cls.__name__)
        _ = router_cls(
            encoder=openai_encoder,
            routes=routes,
            top_k=10,
            index=index,
            auto_sync="local",
        )

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_second_initialization_sync(
        self, openai_encoder, routes, index_cls, router_cls
    ):
        index = init_index(index_cls, index_name=router_cls.__name__)
        route_layer = router_cls(
            encoder=openai_encoder, routes=routes, index=index, auto_sync="local"
        )

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_sync():
            assert route_layer.is_synced()

        check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_second_initialization_not_synced(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        index = init_index(index_cls, index_name=router_cls.__name__)
        _ = router_cls(
            encoder=openai_encoder, routes=routes, index=index, auto_sync="local"
        )
        route_layer = router_cls(encoder=openai_encoder, routes=routes_2, index=index)

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_sync():
            assert route_layer.is_synced() is False

        check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_utterance_diff(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        index = init_index(index_cls, index_name=router_cls.__name__)
        _ = router_cls(
            encoder=openai_encoder, routes=routes, index=index, auto_sync="local"
        )
        route_layer_2 = router_cls(encoder=openai_encoder, routes=routes_2, index=index)

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_utterance_diff():
            diff = route_layer_2.get_utterance_diff(include_metadata=True)
            assert '+ Route 1: Hello | None | {"type": "default"}' in diff
            assert '+ Route 1: Hi | None | {"type": "default"}' in diff
            assert "- Route 1: Hello | None | {}" in diff
            assert "+ Route 2: Au revoir | None | {}" in diff
            assert "- Route 2: Hi | None | {}" in diff
            assert "+ Route 2: Bye | None | {}" in diff
            assert "+ Route 2: Goodbye | None | {}" in diff
            assert "+ Route 3: Boo | None | {}" in diff

        check_utterance_diff()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_auto_sync_local(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        if index_cls is PineconeIndex:
            # TEST LOCAL
            pinecone_index = init_index(index_cls, index_name=router_cls.__name__)
            _ = router_cls(
                encoder=openai_encoder,
                routes=routes,
                index=pinecone_index,
            )
            time.sleep(PINECONE_SLEEP)  # allow for index to be populated
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes_2,
                index=pinecone_index,
                auto_sync="local",
            )
            time.sleep(PINECONE_SLEEP)

            @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            def check_sync():
                # TODO JB: this should use include_metadata=True
                assert route_layer.index.get_utterances(include_metadata=True) == [
                    Utterance(route="Route 1", utterance="Hello"),
                    Utterance(route="Route 2", utterance="Hi"),
                ], "The routes in the index should match the local routes"

            check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_auto_sync_remote(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        if index_cls is PineconeIndex:
            # TEST REMOTE
            pinecone_index = init_index(index_cls, index_name=router_cls.__name__)
            _ = router_cls(
                encoder=openai_encoder,
                routes=routes_2,
                index=pinecone_index,
                auto_sync="local",
            )
            time.sleep(PINECONE_SLEEP)  # allow for index to be populated
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes,
                index=pinecone_index,
                auto_sync="remote",
            )
            time.sleep(PINECONE_SLEEP)

            @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            def check_sync():
                assert route_layer.index.get_utterances(include_metadata=True) == [
                    Utterance(route="Route 1", utterance="Hello"),
                    Utterance(route="Route 2", utterance="Hi"),
                ], "The routes in the index should match the local routes"

            check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_auto_sync_merge_force_local(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        if index_cls is PineconeIndex:
            # TEST MERGE FORCE LOCAL
            pinecone_index = init_index(index_cls, index_name=router_cls.__name__)
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes,
                index=pinecone_index,
                auto_sync="local",
            )
            time.sleep(PINECONE_SLEEP * 2)  # allow for index to be populated
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes_2,
                index=pinecone_index,
                auto_sync="merge-force-local",
            )
            time.sleep(PINECONE_SLEEP * 2)

            @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            def check_sync():
                # confirm local and remote are synced
                assert route_layer.is_synced()
                # now confirm utterances are correct
                local_utterances = route_layer.index.get_utterances(
                    include_metadata=False
                )
                # we sort to ensure order is the same
                # TODO JB: there is a bug here where if we include_metadata=True it fails
                local_utterances.sort(key=lambda x: x.to_str(include_metadata=False))
                assert local_utterances == [
                    Utterance(route="Route 1", utterance="Hello"),
                    Utterance(route="Route 1", utterance="Hi"),
                    Utterance(route="Route 2", utterance="Au revoir"),
                    Utterance(route="Route 2", utterance="Bye"),
                    Utterance(route="Route 2", utterance="Goodbye"),
                    Utterance(route="Route 2", utterance="Hi"),
                    # Utterance(route="Route 3", utterance="Boo"),  # TODO should not be here
                ], "The routes in the index should match the local routes"

            check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_auto_sync_merge_force_remote(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        if index_cls is PineconeIndex:
            # TEST MERGE FORCE LOCAL
            pinecone_index = init_index(index_cls, index_name=router_cls.__name__)
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes,
                index=pinecone_index,
                auto_sync="local",
            )

            @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            def check_r1_utterances():
                # confirm local and remote are synced
                assert route_layer.is_synced()
                # now confirm utterances are correct
                r1_utterances = [
                    Utterance(
                        route="Route 1", utterance="Hello", metadata={"type": "default"}
                    ),
                    Utterance(
                        route="Route 1", utterance="Hi", metadata={"type": "default"}
                    ),
                    Utterance(route="Route 2", utterance="Au revoir"),
                    Utterance(route="Route 2", utterance="Bye"),
                    Utterance(route="Route 2", utterance="Goodbye"),
                    Utterance(route="Route 3", utterance="Boo"),
                ]
                local_utterances = route_layer.index.get_utterances(
                    include_metadata=True
                )
                # we sort to ensure order is the same
                local_utterances.sort(key=lambda x: x.to_str(include_metadata=True))
                assert local_utterances == r1_utterances

            check_r1_utterances()

            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes_2,
                index=pinecone_index,
                auto_sync="merge-force-remote",
            )

            @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            def check_r2_utterances():
                # confirm local and remote are synced
                assert route_layer.is_synced()
                local_utterances = route_layer.index.get_utterances(
                    include_metadata=True
                )
                # we sort to ensure order is the same
                local_utterances.sort(key=lambda x: x.to_str(include_metadata=True))
                assert local_utterances == [
                    Utterance(
                        route="Route 1", utterance="Hello", metadata={"type": "default"}
                    ),
                    Utterance(
                        route="Route 1", utterance="Hi", metadata={"type": "default"}
                    ),
                    Utterance(route="Route 2", utterance="Au revoir"),
                    Utterance(route="Route 2", utterance="Bye"),
                    Utterance(route="Route 2", utterance="Goodbye"),
                    Utterance(route="Route 2", utterance="Hi"),
                    Utterance(route="Route 3", utterance="Boo"),
                ], "The routes in the index should match the local routes"

            check_r2_utterances()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_sync(self, openai_encoder, index_cls, router_cls):
        route_layer = router_cls(
            encoder=openai_encoder,
            routes=[],
            index=init_index(index_cls, index_name=router_cls.__name__),
            auto_sync=None,
        )
        route_layer.sync("remote")

        @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        def check_sync():
            assert route_layer.is_synced()

        check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_auto_sync_merge(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        if index_cls is PineconeIndex:
            # TEST MERGE
            pinecone_index = init_index(index_cls, index_name=router_cls.__name__)
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes_2,
                index=pinecone_index,
                auto_sync="local",
            )
            time.sleep(PINECONE_SLEEP)  # allow for index to be populated
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes,
                index=pinecone_index,
                auto_sync="merge",
            )

            @retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            def check_sync():
                # confirm local and remote are synced
                assert route_layer.is_synced()
                # now confirm utterances are correct
                local_utterances = route_layer.index.get_utterances(
                    include_metadata=True
                )
                # we sort to ensure order is the same
                local_utterances.sort(key=lambda x: x.to_str(include_metadata=True))
                assert local_utterances == [
                    Utterance(
                        route="Route 1", utterance="Hello", metadata={"type": "default"}
                    ),
                    Utterance(
                        route="Route 1", utterance="Hi", metadata={"type": "default"}
                    ),
                    Utterance(route="Route 2", utterance="Au revoir"),
                    Utterance(route="Route 2", utterance="Bye"),
                    Utterance(route="Route 2", utterance="Goodbye"),
                    Utterance(route="Route 2", utterance="Hi"),
                    Utterance(route="Route 3", utterance="Boo"),
                ], "The routes in the index should match the local routes"

            check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_sync_lock_prevents_concurrent_sync(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        """Test that sync lock prevents concurrent synchronization operations"""
        index = init_index(index_cls, index_name=router_cls.__name__)
        route_layer = router_cls(
            encoder=openai_encoder,
            routes=routes_2,
            index=index,
            auto_sync="local",
        )
        # initialize an out of sync router
        route_layer = router_cls(
            encoder=openai_encoder,
            routes=routes,
            index=index,
            auto_sync=None,
        )
        # Acquire sync lock
        route_layer.index.lock(value=True)
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)

        # Attempt to sync while lock is held should raise exception
        with pytest.raises(Exception):
            route_layer.sync("local")

        # Release lock
        route_layer.index.lock(value=False)
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)

        # Should succeed after lock is released
        route_layer.sync("local")
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)
        assert route_layer.is_synced()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    def test_sync_lock_auto_releases(
        self, openai_encoder, routes, index_cls, router_cls
    ):
        """Test that sync lock is automatically released after sync operations"""
        index = init_index(index_cls, index_name=router_cls.__name__)
        route_layer = router_cls(
            encoder=openai_encoder,
            routes=routes,
            index=index,
            auto_sync="local",
        )
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)

        # Lock should be released, allowing another sync
        route_layer.sync("local")  # Should not raise exception
        if index_cls is PineconeIndex:
            time.sleep(PINECONE_SLEEP)
        assert route_layer.is_synced()

        # clear index if pinecone
        if index_cls is PineconeIndex:
            route_layer.index.client.delete_index(route_layer.index.index_name)


@pytest.mark.parametrize(
    "index_cls,router_cls",
    [(index, router) for index in get_test_indexes() for router in get_test_routers()],
)
class TestAsyncSemanticRouter:
    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_initialization(self, openai_encoder, routes, index_cls, router_cls):
        index = init_index(
            index_cls, init_async_index=True, index_name=router_cls.__name__
        )
        _ = router_cls(
            encoder=openai_encoder,
            routes=routes,
            top_k=10,
            index=index,
            auto_sync="local",
        )

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_second_initialization_sync(
        self, openai_encoder, routes, index_cls, router_cls
    ):
        index = init_index(
            index_cls, init_async_index=True, index_name=router_cls.__name__
        )
        route_layer = router_cls(
            encoder=openai_encoder, routes=routes, index=index, auto_sync="local"
        )
        if index_cls is PineconeIndex:
            await asyncio.sleep(PINECONE_SLEEP * 2)  # allow for index to be populated
        assert await route_layer.async_is_synced()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_second_initialization_not_synced(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        index = init_index(
            index_cls, init_async_index=True, index_name=router_cls.__name__
        )
        _ = router_cls(
            encoder=openai_encoder, routes=routes, index=index, auto_sync="local"
        )
        route_layer = router_cls(encoder=openai_encoder, routes=routes_2, index=index)
        if index_cls is PineconeIndex:
            await asyncio.sleep(PINECONE_SLEEP)  # allow for index to be populated
        assert await route_layer.async_is_synced() is False

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_utterance_diff(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        index = init_index(
            index_cls, init_async_index=True, index_name=router_cls.__name__
        )
        _ = router_cls(
            encoder=openai_encoder, routes=routes, index=index, auto_sync="local"
        )
        route_layer_2 = router_cls(encoder=openai_encoder, routes=routes_2, index=index)

        @async_retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        async def check_diff():
            diff = await route_layer_2.aget_utterance_diff(include_metadata=True)
            assert '+ Route 1: Hello | None | {"type": "default"}' in diff
            assert '+ Route 1: Hi | None | {"type": "default"}' in diff
            assert "- Route 1: Hello | None | {}" in diff
            assert "+ Route 2: Au revoir | None | {}" in diff
            assert "- Route 2: Hi | None | {}" in diff
            assert "+ Route 2: Bye | None | {}" in diff
            assert "+ Route 2: Goodbye | None | {}" in diff
            assert "+ Route 3: Boo | None | {}" in diff

        await check_diff()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_auto_sync_local(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        if index_cls is PineconeIndex:
            # TEST LOCAL
            pinecone_index = init_index(
                index_cls, init_async_index=True, index_name=router_cls.__name__
            )
            _ = router_cls(
                encoder=openai_encoder,
                routes=routes,
                index=pinecone_index,
            )
            await asyncio.sleep(PINECONE_SLEEP)  # allow for index to be populated
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes_2,
                index=pinecone_index,
                auto_sync="local",
            )

            @async_retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            async def check_sync():
                assert await route_layer.index.aget_utterances(
                    include_metadata=True
                ) == [
                    Utterance(route="Route 1", utterance="Hello"),
                    Utterance(route="Route 2", utterance="Hi"),
                ], "The routes in the index should match the local routes"

            await check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_auto_sync_remote(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        if index_cls is PineconeIndex:
            # TEST REMOTE
            pinecone_index = init_index(
                index_cls, init_async_index=True, index_name=router_cls.__name__
            )
            _ = router_cls(
                encoder=openai_encoder,
                routes=routes_2,
                index=pinecone_index,
                auto_sync="local",
            )
            await asyncio.sleep(PINECONE_SLEEP)  # allow for index to be populated
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes,
                index=pinecone_index,
                auto_sync="remote",
            )

            @async_retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            async def check_sync():
                assert await route_layer.index.aget_utterances(
                    include_metadata=True
                ) == [
                    Utterance(route="Route 1", utterance="Hello"),
                    Utterance(route="Route 2", utterance="Hi"),
                ], "The routes in the index should match the local routes"

            await check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_auto_sync_merge_force_local(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        if index_cls is PineconeIndex:
            # TEST MERGE FORCE LOCAL
            pinecone_index = init_index(
                index_cls, init_async_index=True, index_name=router_cls.__name__
            )
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes,
                index=pinecone_index,
                auto_sync="local",
            )
            await asyncio.sleep(PINECONE_SLEEP * 2)  # allow for index to be populated
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes_2,
                index=pinecone_index,
                auto_sync="merge-force-local",
            )
            await asyncio.sleep(PINECONE_SLEEP * 2)

            @async_retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            async def check_sync():
                # confirm local and remote are synced
                assert await route_layer.async_is_synced()
                # now confirm utterances are correct
                local_utterances = await route_layer.index.aget_utterances(
                    include_metadata=False
                )
                # we sort to ensure order is the same
                # TODO JB: there is a bug here where if we include_metadata=True it fails
                local_utterances.sort(key=lambda x: x.to_str(include_metadata=False))
                assert local_utterances == [
                    Utterance(route="Route 1", utterance="Hello"),
                    Utterance(route="Route 1", utterance="Hi"),
                    Utterance(route="Route 2", utterance="Au revoir"),
                    Utterance(route="Route 2", utterance="Bye"),
                    Utterance(route="Route 2", utterance="Goodbye"),
                    Utterance(route="Route 2", utterance="Hi"),
                ], "The routes in the index should match the local routes"

            await check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_auto_sync_merge_force_remote(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        if index_cls is PineconeIndex:
            # TEST MERGE FORCE LOCAL
            pinecone_index = init_index(
                index_cls, init_async_index=True, index_name=router_cls.__name__
            )
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes,
                index=pinecone_index,
                auto_sync="local",
            )

            @async_retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            async def populate_index():
                # confirm local and remote are synced
                assert await route_layer.async_is_synced()
                # now confirm utterances are correct
                local_utterances = await route_layer.index.aget_utterances(
                    include_metadata=True
                )
                # we sort to ensure order is the same
                local_utterances.sort(key=lambda x: x.to_str(include_metadata=True))
                assert local_utterances == [
                    Utterance(
                        route="Route 1", utterance="Hello", metadata={"type": "default"}
                    ),
                    Utterance(
                        route="Route 1", utterance="Hi", metadata={"type": "default"}
                    ),
                    Utterance(route="Route 2", utterance="Au revoir"),
                    Utterance(route="Route 2", utterance="Bye"),
                    Utterance(route="Route 2", utterance="Goodbye"),
                    Utterance(route="Route 3", utterance="Boo"),
                ], "The routes in the index should match the local routes"

            await populate_index()

            await asyncio.sleep(PINECONE_SLEEP)  # allow for index to be populated
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes_2,
                index=pinecone_index,
                auto_sync="merge-force-remote",
            )
            await asyncio.sleep(PINECONE_SLEEP)

            @async_retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            async def check_sync():
                # confirm local and remote are synced
                assert await route_layer.async_is_synced()
                # now confirm utterances are correct
                local_utterances = await route_layer.index.aget_utterances(
                    include_metadata=True
                )
                # we sort to ensure order is the same
                local_utterances.sort(key=lambda x: x.to_str(include_metadata=True))
                assert local_utterances == [
                    Utterance(
                        route="Route 1", utterance="Hello", metadata={"type": "default"}
                    ),
                    Utterance(
                        route="Route 1", utterance="Hi", metadata={"type": "default"}
                    ),
                    Utterance(route="Route 2", utterance="Au revoir"),
                    Utterance(route="Route 2", utterance="Bye"),
                    Utterance(route="Route 2", utterance="Goodbye"),
                    Utterance(route="Route 2", utterance="Hi"),
                    Utterance(route="Route 3", utterance="Boo"),
                ], "The routes in the index should match the local routes"

            await check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_sync(self, openai_encoder, index_cls, router_cls):
        route_layer = router_cls(
            encoder=openai_encoder,
            routes=[],
            index=init_index(
                index_cls, init_async_index=True, index_name=router_cls.__name__
            ),
            auto_sync=None,
        )
        await route_layer.async_sync("remote")

        @async_retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
        async def check_sync():
            # confirm local and remote are synced
            assert await route_layer.async_is_synced()

        await check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_auto_sync_merge(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        if index_cls is PineconeIndex:
            # TEST MERGE
            pinecone_index = init_index(
                index_cls, init_async_index=True, index_name=router_cls.__name__
            )
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes_2,
                index=pinecone_index,
                auto_sync="local",
            )
            await asyncio.sleep(PINECONE_SLEEP)  # allow for index to be populated
            route_layer = router_cls(
                encoder=openai_encoder,
                routes=routes,
                index=pinecone_index,
                auto_sync="merge",
            )
            await asyncio.sleep(PINECONE_SLEEP)

            @async_retry(max_retries=RETRY_COUNT, delay=PINECONE_SLEEP)
            async def check_sync():
                # confirm local and remote are synced
                assert await route_layer.async_is_synced()
                # now confirm utterances are correct
                local_utterances = await route_layer.index.aget_utterances(
                    include_metadata=True
                )
                # we sort to ensure order is the same
                local_utterances.sort(key=lambda x: x.to_str(include_metadata=True))
                assert local_utterances == [
                    Utterance(
                        route="Route 1",
                        utterance="Hello",
                        metadata={"type": "default"},
                    ),
                    Utterance(
                        route="Route 1",
                        utterance="Hi",
                        metadata={"type": "default"},
                    ),
                    Utterance(route="Route 2", utterance="Au revoir"),
                    Utterance(route="Route 2", utterance="Bye"),
                    Utterance(route="Route 2", utterance="Goodbye"),
                    Utterance(route="Route 2", utterance="Hi"),
                    Utterance(route="Route 3", utterance="Boo"),
                ], "The routes in the index should match the local routes"

            await check_sync()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_sync_lock_prevents_concurrent_sync(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        """Test that sync lock prevents concurrent synchronization operations"""
        index = init_index(
            index_cls, init_async_index=True, index_name=router_cls.__name__
        )
        route_layer = router_cls(
            encoder=openai_encoder,
            routes=routes_2,
            index=index,
            auto_sync="local",
        )
        # initialize an out of sync router
        route_layer = router_cls(
            encoder=openai_encoder,
            routes=routes,
            index=index,
            auto_sync=None,
        )

        # Acquire sync lock
        await route_layer.index.alock(value=True)
        if index_cls is PineconeIndex:
            await asyncio.sleep(PINECONE_SLEEP)

        # Attempt to sync while lock is held should raise exception
        with pytest.raises(Exception):
            await route_layer.async_sync("local")

        # Release lock
        await route_layer.index.alock(value=False)
        if index_cls is PineconeIndex:
            await asyncio.sleep(PINECONE_SLEEP)

        # Should succeed after lock is released
        await route_layer.async_sync("local")
        if index_cls is PineconeIndex:
            await asyncio.sleep(PINECONE_SLEEP)
        assert await route_layer.async_is_synced()

    @pytest.mark.skipif(
        os.environ.get("PINECONE_API_KEY") is None, reason="Pinecone API key required"
    )
    @pytest.mark.asyncio
    async def test_sync_lock_auto_releases(
        self, openai_encoder, routes, routes_2, index_cls, router_cls
    ):
        """Test that sync lock is automatically released after sync operations"""
        index = init_index(
            index_cls, init_async_index=True, index_name=router_cls.__name__
        )
        route_layer = router_cls(
            encoder=openai_encoder,
            routes=routes_2,
            index=index,
            auto_sync="local",
        )
        route_layer = router_cls(
            encoder=openai_encoder,
            routes=routes,
            index=index,
            auto_sync=None,
        )
        if index_cls is PineconeIndex:
            await asyncio.sleep(PINECONE_SLEEP)
        # Initial sync should acquire and release lock
        await route_layer.async_sync("local")
        if index_cls is PineconeIndex:
            await asyncio.sleep(PINECONE_SLEEP)

        # Lock should be released, allowing another sync
        await route_layer.async_sync("local")  # Should not raise exception
        if index_cls is PineconeIndex:
            await asyncio.sleep(PINECONE_SLEEP)
        assert await route_layer.async_is_synced()

        # clear index if pinecone
        if index_cls is PineconeIndex:
            route_layer.index.client.delete_index(route_layer.index.index_name)
