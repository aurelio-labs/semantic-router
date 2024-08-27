import pytest

from semantic_router.llms.unify import UnifyLLM
from semantic_router.schema import Message
from unify.exceptions import UnifyError
# from dotenv import load_dotenv

# load_dotenv()

@pytest.fixture
def unify_llm(mocker):
    mocker.patch("unify.clients.Unify")
    mocker.patch("json.loads", return_value=["llama-3-8b-chat@together-ai"])
    return UnifyLLM(unify_api_key="test_api_key")


class TestUnifyLLM:
    def test_unify_llm_init_parameter_success(self, unify_llm):
        assert unify_llm.name == "llama-3-8b-chat@together-ai"
        assert unify_llm.temperature == 0.01
        assert unify_llm.max_tokens == 200
        assert unify_llm.stream is False
    	
    def test_unify_llm_init_with_api_key(self, unify_llm):
        assert unify_llm.client is not None, "Client should be initialized"
        assert unify_llm.name == "llama-3-8b-chat@together-ai", "Default name not set correctly"

    def test_unify_llm_init_without_api_key(self, mocker):
        mocker.patch("os.environ.get", return_value=None)
        with pytest.raises(KeyError) as _:
            UnifyLLM()

    def test_unify_llm_call_uninitialized_client(self, unify_llm):
        unify_llm.client = None
        with pytest.raises(UnifyError) as e:
            llm_input = [Message(role="user", content="test")]
            unify_llm(llm_input)
        assert "Unify client is not initialized." in str(e.value)


# @pytest.fixture
# def unify_llm():
#     return UnifyLLM()


# class TestUnifyLLM:
#     def test_unify_llm_init_success(self, unify_llm):
#         assert unify_llm.name == "gpt-4o@openai"
#         assert unify_llm.temperature == 0.01
#         assert unify_llm.max_tokens == 200
#         assert unify_llm.stream is False

#     def test_unify_llm_call_success(self, unify_llm, mocker):
#         mock_response = mocker.MagicMock()
#         mock_response.json.return_value = {"message": {"content": "test response"}}
#         mocker.patch("requests.post", return_value=mock_response)

#         output = unify_llm([Message(role="user", content="test")])
#         assert output == "test response"

#     def test_unify_llm_error_handling(self, unify_llm, mocker):
#         mocker.patch("requests.post", side_effect=Exception("LLM error"))
#         with pytest.raises(Exception) as exc_info:
#             unify_llm([Message(role="user", content="test")])
#         assert "LLM error" in str(exc_info.value)
