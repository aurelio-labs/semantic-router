import asyncio

import pytest

_ = pytest.importorskip("vertexai")

from google.api_core.exceptions import GoogleAPICallError  # noqa: E402
from vertexai.language_models import TextEmbedding  # noqa: E402
from vertexai.language_models._language_models import (  # noqa: E402
    TextEmbeddingStatistics,
)

from semantic_router.encoders import GoogleEncoder  # noqa: E402


@pytest.fixture
def google_encoder(mocker):
    mocker.patch("google.cloud.aiplatform.init")
    mocker.patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
    return GoogleEncoder(project_id="test_project_id")


class TestGoogleEncoder:
    def test_initialization_with_project_id(self, google_encoder):
        assert google_encoder.client is not None, "Client should be initialized"
        assert google_encoder.name == "text-embedding-005", (
            "Default name not set correctly"
        )

    def test_initialization_without_project_id(self, mocker, monkeypatch):
        monkeypatch.delenv("GOOGLE_PROJECT_ID", raising=False)
        mocker.patch("google.cloud.aiplatform.init")
        mocker.patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
        with pytest.raises(ValueError):
            GoogleEncoder()

    def test_call_method(self, google_encoder, mocker):
        mock_embeddings = [
            TextEmbedding(
                values=[0.1, 0.2, 0.3],
                statistics=TextEmbeddingStatistics(token_count=5, truncated=False),
            )
        ]
        mocker.patch.object(
            google_encoder.client, "get_embeddings", return_value=mock_embeddings
        )

        result = google_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(sublist, list) for sublist in result), (
            "Each item in result should be a list"
        )
        google_encoder.client.get_embeddings.assert_called_once()

    def test_returns_list_of_embeddings_for_valid_input(self, google_encoder, mocker):
        mock_embeddings = [
            TextEmbedding(
                values=[0.1, 0.2, 0.3],
                statistics=TextEmbeddingStatistics(token_count=5, truncated=False),
            )
        ]
        mocker.patch.object(
            google_encoder.client, "get_embeddings", return_value=mock_embeddings
        )

        result = google_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(sublist, list) for sublist in result), (
            "Each item in result should be a list"
        )
        google_encoder.client.get_embeddings.assert_called_once()

    def test_handles_multiple_inputs_correctly(self, google_encoder, mocker):
        mock_embeddings = [
            TextEmbedding(
                values=[0.1, 0.2, 0.3],
                statistics=TextEmbeddingStatistics(token_count=5, truncated=False),
            ),
            TextEmbedding(
                values=[0.4, 0.5, 0.6],
                statistics=TextEmbeddingStatistics(token_count=6, truncated=False),
            ),
        ]
        mocker.patch.object(
            google_encoder.client, "get_embeddings", return_value=mock_embeddings
        )

        result = google_encoder(["test1", "test2"])
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(sublist, list) for sublist in result), (
            "Each item in result should be a list"
        )
        google_encoder.client.get_embeddings.assert_called_once()

    def test_raises_value_error_if_project_id_is_none(self, mocker, monkeypatch):
        monkeypatch.delenv("GOOGLE_PROJECT_ID", raising=False)
        mocker.patch("google.cloud.aiplatform.init")
        mocker.patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
        with pytest.raises(ValueError):
            GoogleEncoder()

    def test_raises_value_error_if_google_client_fails_to_initialize(self, mocker):
        mocker.patch(
            "google.cloud.aiplatform.init",
            side_effect=Exception("Failed to initialize client"),
        )
        with pytest.raises(ValueError):
            GoogleEncoder(project_id="test_project_id")

    def test_raises_value_error_if_google_client_is_not_initialized(self, mocker):
        mocker.patch("google.cloud.aiplatform.init")
        mocker.patch(
            "vertexai.language_models.TextEmbeddingModel.from_pretrained",
            return_value=None,
        )
        encoder = GoogleEncoder(project_id="test_project_id")
        with pytest.raises(ValueError):
            encoder(["test"])

    def test_call_method_raises_error_on_api_failure(self, google_encoder, mocker):
        mocker.patch.object(
            google_encoder.client,
            "get_embeddings",
            side_effect=GoogleAPICallError("API call failed"),
        )
        with pytest.raises(ValueError):
            google_encoder(["test"])


class TestGoogleEncoderBatching:
    def test_call_splits_into_batches_of_250(self, google_encoder, mocker):
        def fake_get_embeddings(batch):
            return [
                TextEmbedding(
                    values=[float(doc)],
                    statistics=TextEmbeddingStatistics(token_count=1, truncated=False),
                )
                for doc in batch
            ]

        mocker.patch.object(
            google_encoder.client, "get_embeddings", side_effect=fake_get_embeddings
        )
        docs = [str(i) for i in range(600)]

        result = google_encoder(docs)

        calls = google_encoder.client.get_embeddings.call_args_list
        assert [len(c.args[0]) for c in calls] == [250, 250, 100]
        assert result == [[float(i)] for i in range(600)]

    def test_call_sends_one_text_per_request_for_gemini(self, mocker):
        mocker.patch("google.cloud.aiplatform.init")
        mocker.patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
        encoder = GoogleEncoder(
            name="gemini-embedding-001", project_id="test_project_id"
        )

        def fake_get_embeddings(batch):
            return [
                TextEmbedding(
                    values=[float(doc)],
                    statistics=TextEmbeddingStatistics(token_count=1, truncated=False),
                )
                for doc in batch
            ]

        mocker.patch.object(
            encoder.client, "get_embeddings", side_effect=fake_get_embeddings
        )

        result = encoder(["0", "1", "2"])

        calls = encoder.client.get_embeddings.call_args_list
        assert [len(c.args[0]) for c in calls] == [1, 1, 1]
        assert result == [[0.0], [1.0], [2.0]]


def _fake_embeddings(batch):
    return [
        TextEmbedding(
            values=[float(doc)],
            statistics=TextEmbeddingStatistics(token_count=1, truncated=False),
        )
        for doc in batch
    ]


class TestGoogleEncoderAcall:
    @pytest.mark.asyncio
    async def test_acall_returns_embeddings(self, google_encoder, mocker):
        mocker.patch.object(
            google_encoder.client, "get_embeddings", side_effect=_fake_embeddings
        )
        result = await google_encoder.acall(["1", "2"])
        assert result == [[1.0], [2.0]]

    @pytest.mark.asyncio
    async def test_acall_without_client_raises(self, google_encoder):
        google_encoder.client = None
        with pytest.raises(ValueError, match="client is not initialized"):
            await google_encoder.acall(["test"])

    @pytest.mark.asyncio
    async def test_acall_api_error_raises_value_error(self, google_encoder, mocker):
        mocker.patch.object(
            google_encoder.client,
            "get_embeddings",
            side_effect=GoogleAPICallError("boom"),
        )
        with pytest.raises(ValueError, match="API call failed"):
            await google_encoder.acall(["test"])

    @pytest.mark.asyncio
    async def test_acall_splits_into_batches_of_250(self, google_encoder, mocker):
        mocker.patch.object(
            google_encoder.client, "get_embeddings", side_effect=_fake_embeddings
        )
        docs = [str(i) for i in range(600)]

        result = await google_encoder.acall(docs)

        calls = google_encoder.client.get_embeddings.call_args_list
        assert [len(c.args[0]) for c in calls] == [250, 250, 100]
        assert result == [[float(i)] for i in range(600)]

    @pytest.mark.asyncio
    async def test_acall_sends_one_text_per_request_for_gemini(self, mocker):
        mocker.patch("google.cloud.aiplatform.init")
        mocker.patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
        encoder = GoogleEncoder(
            name="gemini-embedding-001", project_id="test_project_id"
        )
        mocker.patch.object(
            encoder.client, "get_embeddings", side_effect=_fake_embeddings
        )

        result = await encoder.acall(["0", "1", "2"])

        calls = encoder.client.get_embeddings.call_args_list
        assert [len(c.args[0]) for c in calls] == [1, 1, 1]
        assert result == [[0.0], [1.0], [2.0]]

    def test_acall_never_uses_the_loop_bound_async_client(self, google_encoder, mocker):
        # the SDK's async client binds a gRPC channel to the first event loop, so a
        # second asyncio.run() would fail; acall must stay on the sync client
        mocker.patch.object(
            google_encoder.client, "get_embeddings", side_effect=_fake_embeddings
        )
        async_embed = mocker.patch.object(
            google_encoder.client, "get_embeddings_async", create=True
        )

        assert asyncio.run(google_encoder.acall(["1"])) == [[1.0]]
        assert asyncio.run(google_encoder.acall(["2"])) == [[2.0]]
        async_embed.assert_not_called()
