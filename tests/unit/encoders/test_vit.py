import os
from unittest.mock import patch

import numpy as np
import pytest

_ = pytest.importorskip("torch")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from semantic_router.encoders import VitEncoder  # noqa: E402

test_model_name = "aurelio-ai/sr-test-vit"
embed_dim = 32

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


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
    def test_vit_encoder__import_errors_transformers(self):
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError) as error:
                VitEncoder()

        assert "Please install transformers to use VitEncoder" in str(error.value)

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_vit_encoder_initialization(self):
        vit_encoder = VitEncoder(name=test_model_name)
        assert vit_encoder.name == test_model_name
        assert vit_encoder.type == "huggingface"
        assert vit_encoder.score_threshold == 0.5
        assert vit_encoder.device == device

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_vit_encoder_call(self, dummy_pil_image):
        vit_encoder = VitEncoder(name=test_model_name)
        encoded_images = vit_encoder([dummy_pil_image] * 3)

        assert len(encoded_images) == 3
        assert set(map(len, encoded_images)) == {embed_dim}

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_vit_encoder_call_misshaped(self, dummy_pil_image, misshaped_pil_image):
        vit_encoder = VitEncoder(name=test_model_name)
        encoded_images = vit_encoder([dummy_pil_image, misshaped_pil_image])

        assert len(encoded_images) == 2
        assert set(map(len, encoded_images)) == {embed_dim}

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_vit_encoder_process_images_device(self, dummy_pil_image):
        vit_encoder = VitEncoder(name=test_model_name)
        imgs = vit_encoder._process_images([dummy_pil_image] * 3)["pixel_values"]

        assert imgs.device.type == device

    @pytest.mark.skipif(
        os.environ.get("RUN_HF_TESTS") is None, reason="Set RUN_HF_TESTS=1 to run"
    )
    def test_vit_encoder_ensure_rgb(self, dummy_black_and_white_img):
        vit_encoder = VitEncoder(name=test_model_name)
        rgb_image = vit_encoder._ensure_rgb(dummy_black_and_white_img)

        assert rgb_image.mode == "RGB"
        assert np.array(rgb_image).shape == (224, 224, 3)
