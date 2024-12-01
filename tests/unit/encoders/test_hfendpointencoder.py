import pytest

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
