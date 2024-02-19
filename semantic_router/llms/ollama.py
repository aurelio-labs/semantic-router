import os
from typing import List, Optional
import requests
import json

from semantic_router.llms import BaseLLM
from semantic_router.schema import Message
from semantic_router.utils.logger import logger


class OllamaLLM(BaseLLM):
    max_tokens: Optional[int] = 200


    def _call_(self, messages: List[Message]) -> str:
        
        try:

            payload = {
                "model": self.name,
                "messages": [m.to_openai() for m in messages],
                "options":{
                    "temperature":0.0,
                    "num_predict":self.max_tokens
                },
                "format":"json",
                "stream":False
            }

            response = requests.post("http://localhost:11434/api/chat", json=payload)
         
            output = response.json()["message"]["content"]

            return output
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise Exception(f"LLM error: {e}") from e