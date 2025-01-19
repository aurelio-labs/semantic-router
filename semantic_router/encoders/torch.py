from pydantic import PrivateAttr

from semantic_router.encoders import DenseEncoder


class TorchAbstractDenseEncoder(DenseEncoder):
    _torch: any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._torch = self._initialize_torch()

    def _initialize_torch(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                f"Please install PyTorch to use {self.__class__.__name__}. "
                "You can install it with: `pip install semantic-router[local]`"
            )
        
        return torch
    
    def _get_device(self) -> str:
        if self.device:
            device = self.device
        elif self._torch.cuda.is_available():
            device = "cuda"
        elif self._torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return device
