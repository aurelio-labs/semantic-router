import pytest
from voyageai import VoyageError

from semantic_router.encoders import VoyageAIEncoder


@pytest.fixture
def voyageai_encoder(mocker):
    mocker.patch('voyageai.Client')
    return VoyageAIEncoder(voyage_api_key='test_api_key')


class TestVoyageAIEncoder:
    def test_voyageai_encoder_init_success(self, mocker):
        side_effect = ["fake-model-name", "fake-api-key"]
        mocker.patch("os.getenv", side_effect=side_effect)
        encoder = VoyageAIEncoder()
        assert encoder.client is not None

    def test_voyageai_encoder_init_no_api_key(self, mocker):
        mocker.patch("os.getenv", return_value=None)
        with pytest.raises(ValueError) as _:
            VoyageAIEncoder()

    def test_voyageai_encoder_call_uninitialized_client(self, voyageai_encoder):
        voyageai_encoder.client = None
        with pytest.raises(ValueError) as e:
            voyageai_encoder(["test document"])
        assert "VoyageAI client is not initialized." in str(e.value)

    def test_voyageai_encoder_init_exception(self, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("voyageai.Client", side_effect=Exception("Initialization error"))
        with pytest.raises(ValueError) as e:
            VoyageAIEncoder()
        assert (
            "VOYAGE API client failed to initialize. Error: Initialization error"
            in str(e.value)
        )

    def test_voyageai_encoder_call_success(self, voyageai_encoder, mocker):
        mock_response = mocker.Mock()
        mock_response.embeddings = [[0.1, 0.2]]

        mocker.patch("os.getenv", return_value="fake-api-key", autospec=True)
        mocker.patch("time.sleep", return_value=None)

        responses = [VoyageError("VoyageAI error"), mock_response]
        mocker.patch.object(
            voyageai_encoder.client, "embed", side_effect=responses
        )
        embeddings = voyageai_encoder(["test document"])
        assert embeddings == [[0.1, 0.2]]

    def test_voyageai_encoder_call_with_retries(self, voyageai_encoder, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)
        mocker.patch.object(
            voyageai_encoder.client,
            "embed",
            side_effect=VoyageError("Test error")
        )
        with pytest.raises(ValueError) as e:
            voyageai_encoder(["test document"])
        assert "VoyageAI API call failed. Error: " in str(e.value)

    def test_voyageai_encoder_call_failure_non_voyage_error(self, voyageai_encoder, mocker):
        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)
        mocker.patch.object(
            voyageai_encoder.client.embeddings,
            "embed",
            side_effect=Exception("Non-VoyageError")
        )
        with pytest.raises(ValueError) as e:
            voyageai_encoder(["test document"])
        assert "VoyageAI API call failed. Error: Non-VoyageError" in str(e.value)

    def test_voyageai_encoder_call_successful_retry(self, voyageai_encoder, mocker):
        mock_response = mocker.Mock()
        mock_response.embeddings = [[0.1, 0.2]]

        mocker.patch("os.getenv", return_value="fake-api-key")
        mocker.patch("time.sleep", return_value=None)

        responses = [VoyageError("VoyageAI error"), mock_response]
        mocker.patch.object(
            voyageai_encoder.client, "embed", side_effect=responses
        )
        embeddings = voyageai_encoder(["test document"])
        assert embeddings == [[0.1, 0.2]]
