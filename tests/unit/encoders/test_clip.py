import os

import numpy as np
import pytest

_ = pytest.importorskip("torch")

from unittest.mock import patch  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from semantic_router.encoders import CLIPEncoder  # noqa: E402

test_model_name = "aurelio-ai/sr-test-clip"
embed_dim = 64

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


@pytest.fixture()
def dummy_pil_image():
    return Image.fromarray(np.random.rand(512, 224, 3).astype(np.uint8))


@pytest.fixture()
def dummy_black_and_white_img():
    return Image.fromarray(np.random.rand(224, 224, 2).astype(np.uint8))


@pytest.fixture()
def misshaped_pil_image():
    return Image.fromarray(np.random.rand(64, 64, 3).astype(np.uint8))


class TestClipEncoder:
    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_clip_encoder__import_errors_transformers(self):
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError) as error:
                CLIPEncoder()

        assert "install transformers" in str(error.value)

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_clip_encoder__import_errors_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError) as error:
                CLIPEncoder()

        assert "install Pytorch" in str(error.value)

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_clip_encoder_initialization(self):
        clip_encoder = CLIPEncoder(name=test_model_name)
        assert clip_encoder.name == test_model_name
        assert clip_encoder.type == "huggingface"
        assert clip_encoder.score_threshold == 0.2
        assert clip_encoder.device == device

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_clip_encoder_call_text(self):
        clip_encoder = CLIPEncoder(name=test_model_name)
        embeddings = clip_encoder(["hello", "world"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == embed_dim

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_clip_encoder_call_image(self, dummy_pil_image):
        clip_encoder = CLIPEncoder(name=test_model_name)
        encoded_images = clip_encoder([dummy_pil_image] * 3)

        assert len(encoded_images) == 3
        assert set(map(len, encoded_images)) == {embed_dim}

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_clip_encoder_call_misshaped(self, dummy_pil_image, misshaped_pil_image):
        clip_encoder = CLIPEncoder(name=test_model_name)
        encoded_images = clip_encoder([dummy_pil_image, misshaped_pil_image])

        assert len(encoded_images) == 2
        assert set(map(len, encoded_images)) == {embed_dim}

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_clip_device(self):
        clip_encoder = CLIPEncoder(name=test_model_name)
        device = clip_encoder._model.device.type
        assert device == device

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_clip_encoder_ensure_rgb(self, dummy_black_and_white_img):
        clip_encoder = CLIPEncoder(name=test_model_name)
        rgb_image = clip_encoder._ensure_rgb(dummy_black_and_white_img)

        assert rgb_image.mode == "RGB"
        assert np.array(rgb_image).shape == (224, 224, 3)
