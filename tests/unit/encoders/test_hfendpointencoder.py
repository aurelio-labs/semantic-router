import pytest
import requests

from semantic_router.encoders.huggingface import HFEndpointEncoder


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
    )


class TestHFEndpointEncoder:
    def test_initialization(self, encoder):
        assert (
            encoder.huggingface_url
            == "https://api-inference.huggingface.co/models/bert-base-uncased"
        )
        assert encoder.huggingface_api_key == "test-api-key"

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

    def test_query_failure(self, encoder, requests_mock, mocker):
        mocker.patch("semantic_router.encoders.huggingface.time.sleep")
        requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            text="Error",
            status_code=400,
        )
        with pytest.raises(ValueError) as exc_info:
            encoder.query({"inputs": "Hello World!", "parameters": {}})
        assert "Query failed with status 400: Error" in str(exc_info.value)

    def test_query_retries_max_retries_times(self, encoder, requests_mock, mocker):
        mocker.patch("semantic_router.encoders.huggingface.time.sleep")
        adapter = requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            text="Error",
            status_code=500,
        )
        with pytest.raises(ValueError):
            encoder.query({"inputs": "Hello World!", "parameters": {}}, max_retries=3)
        assert adapter.call_count == 3

    def test_query_connection_error_is_reported(self, encoder, requests_mock, mocker):
        mocker.patch("semantic_router.encoders.huggingface.time.sleep")
        requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            exc=requests.exceptions.ConnectionError("Name or service not known"),
        )
        with pytest.raises(ValueError, match="Name or service not known") as exc_info:
            encoder.query({"inputs": "Hello World!", "parameters": {}})
        assert isinstance(exc_info.value.__cause__, requests.exceptions.ConnectionError)

    def test_encode_documents_success(self, encoder, requests_mock):
        requests_mock.post(
            "https://api-inference.huggingface.co/models/bert-base-uncased",
            json=[0.1, 0.2, 0.3],
            status_code=200,
        )
        embeddings = encoder(["Hello World!"])
        assert embeddings == [[0.1, 0.2, 0.3]]
