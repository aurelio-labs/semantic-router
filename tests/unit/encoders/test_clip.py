import numpy as np
import pytest
from PIL import Image

from semantic_router.encoders import CLIPEncoder

clip_encoder = CLIPEncoder()


@pytest.fixture()
def dummy_pil_image():
    return Image.fromarray(np.random.rand(1024, 512, 3).astype(np.uint8))


@pytest.fixture()
def dummy_black_and_white_img():
    return Image.fromarray(np.random.rand(224, 224, 2).astype(np.uint8))


@pytest.fixture()
def misshaped_pil_image():
    return Image.fromarray(np.random.rand(64, 64, 3).astype(np.uint8))


class TestVitEncoder:
    def test_clip_encoder__import_errors_transformers(self, mocker):
        mocker.patch.dict("sys.modules", {"transformers": None})
        with pytest.raises(ImportError):
            CLIPEncoder()

    def test_clip_encoder__import_errors_torch(self, mocker):
        mocker.patch.dict("sys.modules", {"torch": None})
        with pytest.raises(ImportError):
            CLIPEncoder()

    def test_clip_encoder_initialization(self):
        assert clip_encoder.name == "openai/clip-vit-base-patch32"
        assert clip_encoder.type == "huggingface"
        assert clip_encoder.score_threshold == 0.2
        assert clip_encoder.device == "cpu"

    def test_clip_encoder_call_text(self):
        embeddings = clip_encoder(["hello", "world"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 512

    def test_clip_encoder_call_image(self, dummy_pil_image):
        encoded_images = clip_encoder([dummy_pil_image] * 3)

        assert len(encoded_images) == 3
        assert set(map(len, encoded_images)) == {512}

    def test_clip_encoder_call_misshaped(self, dummy_pil_image, misshaped_pil_image):
        encoded_images = clip_encoder([dummy_pil_image, misshaped_pil_image])

        assert len(encoded_images) == 2
        assert set(map(len, encoded_images)) == {512}

    def test_clip_device(self):
        device = clip_encoder._model.device.type
        assert device == "cpu"

    def test_clip_encoder_ensure_rgb(self, dummy_black_and_white_img):
        rgb_image = clip_encoder._ensure_rgb(dummy_black_and_white_img)

        assert rgb_image.mode == "RGB"
        assert np.array(rgb_image).shape == (224, 224, 3)