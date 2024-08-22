from typing import List, Optional

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.defaults import EncoderDefault

# from unify.utils import _validate_api_key
from unify.exceptions import UnifyError
from unify.clients import Unify, AsyncUnify


class UnifyLLM(BaseLLM):

    client: Optional[Unify] = None
    async_client: Optional[AsyncUnify] = None

    def __init__(
        self,
        name: Optional[str] = None,
        unify_api_key: Optional[str] = None,
    ):

        if name is None:
            name = f"{EncoderDefault.UNIFY.value["language_model"]}@\
            {EncoderDefault.UNIFY.value["language_provider"]}"

        super().__init__(name=name)

        self.client = Unify(endpoint=name, api_key=unify_api_key)

    def __call__(self, messages: List[Message]) -> str:
        if self.client is None:
            raise UnifyError("Unify client is not initialized.")
        try:
            output = self.client.generate(
                messages=[m.to_openai() for m in messages]
                )

            if not output:
                raise Exception("No output generated")
            return output

        except Exception as e:
            raise UnifyError(f"Unify API call failed. Error: {e}") from e

