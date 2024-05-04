import json
from typing import List, Optional, Any

import boto3

from semantic_router.encoders import BaseEncoder
from semantic_router.utils.defaults import EncoderDefault


class BedrockEncoder(BaseEncoder):
    client: Any = None
    type: str = "bedrock"
    input_type: Optional[str] = "search_query"
    session: Optional[Any] = None
    region: Optional[str] = None

    def __init__(
        self,
        name: Optional[str] = None,
        session: Optional[Any] = None,
        region: Optional[str] = None,
        score_threshold: float = 0.3,
        input_type: Optional[str] = "search_query",
    ):
        if name is None:
            name = EncoderDefault.BEDROCK.value["embedding_model"]
        super().__init__(name=name, score_threshold=score_threshold)
        self.input_type = input_type
        self.session = session or boto3.Session()
        if self.session.get_credentials() is None:
            raise ValueError("Could not get AWS session")
        self.region = region or self.session.region_name
        if self.region is None:
            raise ValueError("No AWS region provided")
        try:
            self.client = self.session.client(
                service_name="bedrock-runtime", region_name=str(self.region)
            )
        except Exception as e:
            raise ValueError(f"Bedrock client failed to initialise. Error: {e}") from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        if self.client is None:
            raise ValueError("Bedrock client is not initialised.")
        try:
            embeddings = []
            if "amazon" in self.name:
                for doc in docs:
                    doc = json.dumps(
                        {
                            "inputText": doc,
                        }
                    )
                    response = self.client.invoke_model(
                        body=doc,
                        modelId=self.name,
                        accept="*/*",
                        contentType="application/json",
                    )

                    response_body = json.loads(response.get("body").read())

                    embedding = response_body.get("embedding")
                    embeddings.append(embedding)
            elif "cohere" in self.name:
                MAX_WORDS = 400
                for doc in docs:
                    words = doc.split()
                    if len(words) > MAX_WORDS:
                        chunks = [
                            " ".join(words[i : i + MAX_WORDS])
                            for i in range(0, len(words), MAX_WORDS)
                        ]
                    else:
                        chunks = [doc]

                    for chunk in chunks:
                        chunk = json.dumps(
                            {"texts": [chunk], "input_type": self.input_type}
                        )

                        response = self.client.invoke_model(
                            body=chunk,
                            modelId=self.name,
                            accept="*/*",
                            contentType="application/json",
                        )

                        response_body = json.loads(response.get("body").read())

                        chunk_embeddings = response_body.get("embeddings")
                        embeddings.extend(chunk_embeddings)
            else:
                raise ValueError("Unknown model name")
            return embeddings
        except Exception as e:
            raise ValueError(f"Bedrock call failed. Error: {e}") from e
