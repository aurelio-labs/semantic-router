import os
from typing import Dict, List, Optional

import unify

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message

from unify.utils import _validate_api_key
from unify.exceptions import UnifyError
from unify.clients import Unify, AsyncUnify

class UnifyLLM(BaseLLM):

    client: Optional[Unify] = None
	
    def __init__(
        self,
        name: Optional[str] = None,
        unify_api_key: Optional[str] = None,
    ):
	
        if name is None:
            name = EncoderDefault.UNIFy.value["end_point"]
        
        super().__init__(name=name)
	
        try:
            self.client = Unify(name, api_key=unify_api_key)
        except Exception as e:
            raise ValueError(
                f"Unify API client failed to initialize. Error: {e}"
            ) from e

        def __call__(self, messages: List[Message]) -> str:
            if self.client is None:
                raise ValueError("Unify client is not initialized.")
            try:
                output = self.client.generate(messages=[m.to_openai() for m in messages])
				
                if not output:
                    raise Exception("No output generated")
                return output

            except Exception as e:
                raise UnifyError(f"Unify API call failed. Error: {e}") from e
