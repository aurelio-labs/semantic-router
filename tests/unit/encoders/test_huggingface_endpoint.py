from unittest.mock import MagicMock, patch

import pytest
import requests

from semantic_router.encoders.huggingface import HFEndpointEncoder


def _resp(status, body):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    r.text = str(body)
    if status >= 400:
        err = requests.exceptions.HTTPError(f"{status}")
        err.response = r
        r.raise_for_status.side_effect = err
    return r


@pytest.fixture
def encoder():
    with patch.object(HFEndpointEncoder, "query", return_value=[[0.0]]):
        return HFEndpointEncoder(
            huggingface_url="https://example.com", huggingface_api_key="key"
        )


class TestHuggingFaceEndpointQuery:
    def test_query_success(self, encoder):
        with patch("requests.post", return_value=_resp(200, [[0.1, 0.2]])):
            assert encoder.query({"inputs": "x"}) == [[0.1, 0.2]]

    def test_query_503_with_estimated_time_exhausted_raises(self, encoder):
        resp = _resp(503, {"error": "loading", "estimated_time": 0.0})
        with patch("requests.post", return_value=resp), patch("time.sleep"):
            with pytest.raises(ValueError):
                encoder.query({"inputs": "x"}, max_retries=2)

    def test_query_503_without_estimated_time_raises(self, encoder):
        resp = _resp(503, {"error": "unavailable"})
        with patch("requests.post", return_value=resp), patch("time.sleep"):
            with pytest.raises(ValueError):
                encoder.query({"inputs": "x"}, max_retries=2)

    def test_query_503_then_success(self, encoder):
        resps = [_resp(503, {"estimated_time": 0.0}), _resp(200, [[1.0]])]
        with patch("requests.post", side_effect=resps), patch("time.sleep"):
            assert encoder.query({"inputs": "x"}) == [[1.0]]
