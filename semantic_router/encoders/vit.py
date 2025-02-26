from typing import Any, Dict, List, Optional

from pydantic import PrivateAttr

from semantic_router.encoders import DenseEncoder


class VitEncoder(DenseEncoder):
    """Encoder for Vision Transformer models.

    This class provides functionality to encode images using a Vision Transformer
    model via Hugging Face. It supports various image processing and model initialization
    options.
    """

    name: str = "google/vit-base-patch16-224"
    type: str = "huggingface"
    processor_kwargs: Dict = {}
    model_kwargs: Dict = {}
    device: Optional[str] = None
    _processor: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _torch: Any = PrivateAttr()
    _T: Any = PrivateAttr()
    _Image: Any = PrivateAttr()

    def __init__(self, **data):
        """Initialize the VitEncoder.

        :param **data: Additional keyword arguments for the encoder.
        :type **data: dict
        """
        if data.get("score_threshold") is None:
            data["score_threshold"] = 0.5
        super().__init__(**data)
        self._processor, self._model = self._initialize_hf_model()

    def _initialize_hf_model(self):
        """Initialize the Hugging Face model.

        :return: The processor and model.
        :rtype: tuple
        """
        try:
            from transformers import ViTImageProcessor, ViTModel
        except ImportError:
            raise ImportError(
                "Please install transformers to use VitEncoder. "
                "You can install it with: "
                "`pip install semantic-router[vision]`"
            )

        try:
            import torch
            import torchvision.transforms as T
        except ImportError:
            raise ImportError(
                "Please install Pytorch to use VitEncoder. "
                "You can install it with: "
                "`pip install semantic-router[vision]`"
            )

        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Please install PIL to use VitEncoder. "
                "You can install it with: "
                "`pip install semantic-router[vision]`"
            )

        self._torch = torch
        self._Image = Image
        self._T = T

        processor = ViTImageProcessor.from_pretrained(
            self.name, **self.processor_kwargs
        )

        model = ViTModel.from_pretrained(self.name, **self.model_kwargs)

        self.device = self._get_device()
        model.to(self.device)

        return processor, model

    def _get_device(self) -> str:
        """Get the device to use for the model.

        :return: The device to use for the model.
        :rtype: str
        """
        if self.device:
            device = self.device
        elif self._torch.cuda.is_available():
            device = "cuda"
        elif self._torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return device

    def _process_images(self, images: List[Any]):
        """Process the images for the model.

        :param images: The images to process.
        :type images: List[Any]
        :return: The processed images.
        :rtype: Any
        """
        rgb_images = [self._ensure_rgb(img) for img in images]
        processed_images = self._processor(images=rgb_images, return_tensors="pt")
        processed_images = processed_images.to(self.device)
        return processed_images

    def _ensure_rgb(self, img: Any):
        """Ensure the image is in RGB format.

        :param img: The image to ensure is in RGB format.
        :type img: Any
        :return: The image in RGB format.
        :rtype: Any
        """
        rgbimg = self._Image.new("RGB", img.size)
        rgbimg.paste(img)
        return rgbimg

    def __call__(
        self,
        imgs: List[Any],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Encode a list of images into embeddings using the Vision Transformer model.

        :param imgs: The images to encode.
        :type imgs: List[Any]
        :param batch_size: The batch size for encoding.
        :type batch_size: int
        :return: The embeddings for the images.
        :rtype: List[List[float]]
        """
        all_embeddings = []
        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i : i + batch_size]
            batch_imgs_transform = self._process_images(batch_imgs)
            with self._torch.no_grad():
                embeddings = (
                    self._model(**batch_imgs_transform)
                    .last_hidden_state[:, 0]
                    .cpu()
                    .tolist()
                )
            all_embeddings.extend(embeddings)
        return all_embeddings
