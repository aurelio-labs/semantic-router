from typing import List

import httpx

CHAT_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "test"},
            "finish_reason": "stop",
        }
    ],
}


def capture_chat_requests(mocker) -> List[httpx.Request]:
    """Answer every HTTP request the OpenAI client sends with a canned chat
    completion and record it, so a test can check where a call went without a
    network or an API key."""
    requests: List[httpx.Request] = []

    def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=CHAT_COMPLETION, request=request)

    mocker.patch.object(httpx.Client, "send", send)
    return requests
