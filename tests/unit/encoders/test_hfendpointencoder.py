import pytest
from semantic_router.encoders.huggingface import HFEndpointEncoder
from unittest import mock


@pytest.fixture
def encoder(requests_mock):
    # Mock the HTTP request made during HFEndpointEncoder initialization
    requests_mock.post(
        "https://api-inference.huggingface.co/models/bert-base-uncased",
        json=[0.1, 0.2, 0.3],
        status_code=200,
    )
    # Now, when HFEndpointEncoder is initialized, it will use the mocked response above
    return HFEndpointEncoder(
        huggingface_url="https://api-inference.huggingface.co/models/bert-base-uncased",
        huggingface_api_key="test-api-key",
        score_threshold=0.8,
    )


class TestHFEndpointEncoder:
    def test_initialization(self, encoder):
        assert (
            encoder.huggingface_url
            == "https://api-inference.huggingface.co/models/bert-base-uncased"
        )
        assert encoder.huggingface_api_key == "test-api-key"
        assert encoder.score_threshold == 0.8

    def test_initialization_failure_no_api_key(self):
        with pytest.raises(ValueError) as exc_info:
            HFEndpointEncoder(
                huggingface_url="https://api-inference.huggingface.co/models/bert-base-uncased"
            )
        assert "HuggingFace API key cannot be 'None'" in str(exc_info.value)

    def test_initialization_failure_no_url(self):
        with pytest.raises(ValueError) as exc_info:
            HFEndpointEncoder(huggingface_api_key="test-api-key")
        assert "HuggingFace endpoint url cannot be 'None'" in str(exc_info.value)

    def test_query_success(self, encoder, requests_mock):
        requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            json=[0.1, 0.2, 0.3],
            status_code=200,
        )
        response = encoder.query({"inputs": "Hello World!", "parameters": {}})
        assert response == [0.1, 0.2, 0.3]

    def test_query_failure(self, encoder, requests_mock):
        requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            text="Error",
            status_code=400,
        )
        with pytest.raises(ValueError) as exc_info:
            encoder.query({"inputs": "Hello World!", "parameters": {}})
        assert "Query failed with status 400: Error" in str(exc_info.value)

    def test_encode_documents_success(self, encoder, requests_mock):
        requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            json=[0.1, 0.2, 0.3],
            status_code=200,
        )
        embeddings = encoder(["Hello World!"])
        assert embeddings == [[0.1, 0.2, 0.3]]

    def test_initialization_failure_query_exception(self, requests_mock, mocker):
        # Mock the query method to raise an exception
        mocker.patch(
            "semantic_router.encoders.huggingface.HFEndpointEncoder.query",
            side_effect=Exception("Initialization error"),
        )

        with pytest.raises(ValueError) as exc_info:
            HFEndpointEncoder(
                huggingface_url="https://api-inference.huggingface.co/models/bert-base-uncased",
                huggingface_api_key="test-api-key",
            )
        assert (
            "HuggingFace endpoint client failed to initialize. Error: Initialization error"
            in str(exc_info.value)
        )

    def test_no_embeddings_returned(self, encoder, requests_mock):
        # Mock the response to return an empty list, simulating no embeddings
        requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            json=[],
            status_code=200,
        )
        with pytest.raises(ValueError) as exc_info:
            encoder(["Hello World!"])
        assert "No embeddings returned from the query." in str(exc_info.value)

    def test_no_embeddings_for_batch(self, encoder, requests_mock):
        # Mock the response to simulate a server error
        requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            text="Error",
            status_code=500,
        )
        with pytest.raises(ValueError) as exc_info:
            encoder(["Hello World!"])
        assert (
            "No embeddings returned for batch. Error: Query failed with status 500: Error"
            in str(exc_info.value)
        )

    def test_embeddings_extend(self, encoder, requests_mock):
        # Mock the response to return a list of embeddings
        requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            json=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            status_code=200,
        )
        embeddings = encoder(["Hello World!", "Test"])
        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_query_with_estimated_time(self, encoder, requests_mock):
        # Mock the response to simulate a 503 status with an estimated_time
        requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            [
                {
                    "json": {"estimated_time": 2},
                    "status_code": 503,
                },
                {
                    "json": [0.1, 0.2, 0.3],
                    "status_code": 200,
                },
            ],
        )

        with mock.patch("time.sleep", return_value=None) as mock_sleep:
            response = encoder.query({"inputs": "Hello World!", "parameters": {}})
            assert response == [0.1, 0.2, 0.3]
            mock_sleep.assert_called_once_with(2)
