import torch
from typing import Dict, Optional
from detection_mcp_server.detectors.base import BaseModel
from detection_mcp_server.detectors.grounding_dino import GroundingDINODetector
from detection_mcp_server.detectors.sam2 import SAM2Segmenter
from detection_mcp_server.detectors.florence2 import Florence2Detector
from agent.logging_config import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    _instance: Optional['ModelRegistry'] = None

    def __new__(cls, device: str = "cpu", use_bfloat16: bool = False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: str = "cpu", use_bfloat16: bool = False):
        if not self._initialized:
            self._device = device
            self._use_bfloat16 = use_bfloat16
            self._models: Dict[str, BaseModel] = {}
            self._initialize_models()
            self._initialized = True

    def _initialize_models(self) -> None:
        self._models["grounding-dino-tiny"] = GroundingDINODetector(
            model_id="IDEA-Research/grounding-dino-tiny",
            device=self._device,
            use_bfloat16=self._use_bfloat16,
        )

        self._models["sam2.1-hiera-base-plus"] = SAM2Segmenter(
            model_id="facebook/sam2.1-hiera-base-plus",
            device=self._device if self._device != "mps" else "cpu",
            use_bfloat16=self._use_bfloat16 if self._device != "mps" else False,
        )

        self._models["florence-2-large"] = Florence2Detector(
            model_id="microsoft/florence-2-base",
            device=self._device,
            use_bfloat16=self._use_bfloat16,
        )

    def load_all_models(self) -> None:
        logger.info("Loading models", model_count=len(self._models))
        for model_name, model in self._models.items():
            if not model.is_loaded:
                logger.info("Loading model", model_name=model_name)
                try:
                    model.load()
                    logger.info("Model loaded successfully", model_name=model_name)
                except Exception as e:
                    logger.error("Model loading failed", model_name=model_name, error=str(e))
                    raise
        logger.info("All models loaded successfully", device=self._device)

    def get_model(self, model_name: str) -> Optional[BaseModel]:
        model = self._models.get(model_name)
        if model and not model.is_loaded:
            model.load()
        return model

    def list_models(self) -> list[dict]:
        return [model.get_info() for model in self._models.values()]

    def get_available_model_names(self) -> list[str]:
        return list(self._models.keys())

    def unload_model(self, model_name: str) -> bool:
        model = self._models.get(model_name)
        if model is None or not model.is_loaded:
            return False

        if hasattr(model, '_model') and model._model is not None:
            del model._model
            model._model = None

        if hasattr(model, '_pipe') and model._pipe is not None:
            del model._pipe
            model._pipe = None

        if hasattr(model, '_processor') and model._processor is not None:
            del model._processor
            model._processor = None

        self._clear_device_cache()
        return True

    def unload_all_models(self) -> int:
        count = 0
        for model_name in self._models.keys():
            if self.unload_model(model_name):
                count += 1
        return count

    def _clear_device_cache(self) -> None:
        if self._device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def device(self) -> str:
        return self._device

    @classmethod
    def reset(cls) -> None:
        if cls._instance is not None:
            cls._instance.unload_all_models()
            cls._instance = None


def get_registry(device: str = "cpu", use_bfloat16: bool = False) -> ModelRegistry:
    return ModelRegistry(device=device, use_bfloat16=use_bfloat16)
