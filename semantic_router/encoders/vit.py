from typing import Any, List, Optional

from pydantic.v1 import PrivateAttr

from semantic_router.encoders import BaseEncoder


class VitEncoder(BaseEncoder):
    name: str = "google/vit-base-patch16-224"
    type: str = "huggingface"
    score_threshold: float = 0.5
    extractor_kwargs: dict = {}
    model_kwargs: dict = {}
    device: Optional[str] = None
    _extractor: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _torch: Any = PrivateAttr()
    _T: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._extractor, self._model = self._initialize_hf_model()

    def _initialize_hf_model(self):
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError:
            raise ImportError(
                "Please install transformers to use HuggingFaceEncoder. "
                "You can install it with: "
                "`pip install semantic-router[local]`"
            )

        try:
            import torch
            import torchvision.transforms as T
        except ImportError:
            raise ImportError(
                "Please install Pytorch to use HuggingFaceEncoder. "
                "You can install it with: "
                "`pip install semantic-router[local]`"
            )

        self._torch = torch
        self._T = T

        extractor = AutoFeatureExtractor.from_pretrained(
            self.name,
            **self.extractor_kwargs,
        )

        model = AutoModel.from_pretrained(self.name, **self.model_kwargs)

        if self.device:
            model.to(self.device)

        else:
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
            model.to(device)
            self.device = device

        return extractor, model

    def __call__(
        self,
        imgs: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i : i + batch_size]
            batch_imgs_transform = self._torch.stack(
                [self._transformation_chain(img) for img in batch_imgs]
            )
            new_batch = {"pixel_values": batch_imgs_transform.to(self._model.device)}
            with self._torch.no_grad():
                embeddings = (
                    self._model(**new_batch).last_hidden_state[:, 0].cpu().tolist()
                )
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _transformation_chain(self, img):
        return self._T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                self._T.Resize(int((256 / 224) * self._extractor.size["height"])),
                self._T.CenterCrop(self._extractor.size["height"]),
                self._T.ToTensor(),
                self._T.Normalize(
                    mean=self._extractor.image_mean, std=self._extractor.image_std
                ),
            ]
        )(img)
