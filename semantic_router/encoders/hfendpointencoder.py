from typing import List, Optional

import requests

from semantic_router.encoders import BaseEncoder

class HFEndpointEncoder(BaseEncoder):
    huggingface_url : Optional[str] = None
    huggingface_api_key: Optional[str] = None
    score_threshold : float = 0.8

    def __init__(self, name: Optional[str] = "hugging_face_custom_endpoint", huggingface_url: Optional[str] = None, huggingface_api_key: Optional[str] = None, score_threshold: float = 0.8):

        super().__init__(name = name, huggingface_url=huggingface_url, huggingface_api_key=huggingface_api_key, score_threshold=score_threshold)
        huggingface_url = huggingface_url
        huggingface_api_key = huggingface_api_key
        score_threshold = score_threshold

        if huggingface_url is None:
            raise ValueError("HuggingFace endpoint url cannot be 'None'.")

        if huggingface_api_key is None:
            raise ValueError("HuggingFace API key cannot be 'None'.")
        
        try:
            self.query({"inputs": "Hello World!", "parameters": {} })            
            pass

        except Exception as e:
            raise ValueError(
                f"HuggingFace endpoint client failed to initialize. Error: {e}"
            ) from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        embeddings = []
        for d in docs:
            try:
                output = self.query({"inputs": d, "parameters": {} })
                embeddings.append(output[0])
            except Exception as e:
                raise ValueError(f"No embeddings returned. Error!")
        return embeddings
    

    def query(self, payload):
        API_URL = self.huggingface_url
        headers = {
            "Accept" : "application/json",
            "Authorization": f"Bearer {self.huggingface_api_key}",
            "Content-Type": "application/json" 
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
