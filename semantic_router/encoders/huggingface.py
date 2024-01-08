from typing import Any
import torch
from pydantic import PrivateAttr
from semantic_router.encoders import BaseEncoder


class HuggingFaceEncoder(BaseEncoder):
    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    type: str = "huggingface"
    score_threshold: float = 0.5
    tokenizer_kwargs: dict = {}
    model_kwargs: dict = {}
    device: str | None = None
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._tokenizer, self._model = self._initialize_hf_model()

    def _initialize_hf_model(self):
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Please install transformers to use HuggingFaceEncoder. "
                "You can install it with: "
                "`pip install semantic-router[transformers]`"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.name,
            **self.tokenizer_kwargs,
        )

        model = AutoModel.from_pretrained(self.name, **self.model_kwargs)

        if self.device:
            model.to(self.device)

        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            self.device = device

        return tokenizer, model

    def __call__(
        self,
        docs: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        pooling_strategy: str = "mean",
    ) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]

            encoded_input = self._tokenizer(
                batch_docs, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self._model(**encoded_input)

            if pooling_strategy == "mean":
                embeddings = self._mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )
            elif pooling_strategy == "max":
                embeddings = self._max_pooling(
                    model_output, encoded_input["attention_mask"]
                )

            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            embeddings = embeddings.tolist()
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]
