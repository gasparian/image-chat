from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image


class BaseModel(ABC):
    def __init__(self, device: str = "cpu", use_bfloat16: bool = False):
        self.device = device
        self.use_bfloat16 = use_bfloat16
        self._model: Optional[object] = None

    @abstractmethod
    def load(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @abstractmethod
    def get_info(self) -> dict:
        pass
