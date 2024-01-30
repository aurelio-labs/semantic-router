import pytest

from semantic_router.encoders import VitEncoder


@pytest.fixture
def mock_vit_encoder(mocker):
    mocker.patch("transformers.AutoModel", autospec=True)
    mocker.patch("transformers.AutoFeatureExtractor", autospec=True)

    encoder = VitEncoder(name="some_model_name")
    mocker.patch.object(encoder, "_initialize_hf_model")

    return encoder


class TestVitEncoder:
    def test_vit_encoder__import_errors_transformers(self, mocker):
        mocker.patch.dict("sys.modules", {"transformers": None})
        with pytest.raises(ImportError):
            VitEncoder()

    def test_vit_encoder__import_errors_torch(self, mocker):
        mocker.patch.dict("sys.modules", {"torch": None})
        with pytest.raises(ImportError):
            VitEncoder()

    def test_vit_encoder__import_errors_torchvision(self, mocker):
        mocker.patch.dict("sys.modules", {"torchvision": None})
        with pytest.raises(ImportError):
            VitEncoder()

    @pytest.mark.skip(reason="TODO: Fix torch mocking")
    def test_vit_encoder_initialization(self, mocker, monkeypatch):
        mock_model = mocker.patch(
            "transformers.AutoModel.from_pretrained", autospec=True
        )
        mock_extractor = mocker.patch(
            "transformers.AutoFeatureExtractor.from_pretrained", autospec=True
        )
        monkeypatch.setattr(
            "torch.cuda.is_available", mocker.MagicMock(return_value=False)
        )

        mock_model.return_value = mocker.MagicMock()
        mock_extractor.return_value = mocker.MagicMock(size={"height": 224})

        encoder = VitEncoder()

        assert encoder.name == "google/vit-base-patch16-224"
        assert encoder.type == "huggingface"
        assert encoder.score_threshold == 0.5
        assert encoder.device == "cpu"

        mock_model.assert_called_once_with(encoder.name, **encoder.model_kwargs)
        mock_extractor.assert_called_once_with(encoder.name, **encoder.extractor_kwargs)
