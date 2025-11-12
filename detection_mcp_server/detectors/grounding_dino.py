import torch
from PIL import Image
from transformers import pipeline
from typing import Optional

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from detection_mcp_server.detectors.base import BaseModel
from agent.logging_config import get_logger

logger = get_logger(__name__)


class GroundingDINODetector(BaseModel):
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cpu",
        use_bfloat16: bool = True
    ):
        super().__init__(device, use_bfloat16)
        self.model_id = model_id
        self._pipe: Optional[object] = None

    def load(self) -> None:
        if self._pipe is not None:
            return

        logger.info("Loading Grounding DINO model", model_id=self.model_id)

        dtype = torch.bfloat16 if self.use_bfloat16 else None
        kwargs = {"torch_dtype": dtype} if dtype else {}

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            use_fast=True
        )

        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id,
            **kwargs
        ).to(self.device)

        self._pipe = pipeline(
            "zero-shot-object-detection",
            model=model,
            image_processor=processor.image_processor,
            tokenizer=processor.tokenizer,
            device=self.device
        )
        self._model = self._pipe
        dtype_str = "bfloat16" if self.use_bfloat16 else "float32"
        logger.info("Grounding DINO model loaded", device=self.device, dtype=dtype_str)

    def detect(
        self,
        image: Image.Image,
        phrases: list[str],
        threshold: float = 0.25
    ) -> list[dict]:
        if self._pipe is None:
            self.load()

        phrases = [p.strip() for p in phrases if p.strip()]
        if not phrases:
            return []

        all_results = []

        for phrase in phrases:
            formatted_phrase = phrase.lower().strip()
            if not formatted_phrase.endswith("."):
                formatted_phrase += "."

            try:
                results = self._pipe(
                    image,
                    candidate_labels=[formatted_phrase],
                    threshold=threshold
                )

                for result in results:
                    box = result["box"]
                    all_results.append({
                        "bbox": [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
                        "score": float(result["score"]),
                        "label": phrase
                    })
            except Exception as e:
                logger.warning("Detection failed for phrase", phrase=phrase, error=str(e))
                continue

        return all_results

    def get_info(self) -> dict:
        model_name = "grounding-dino-tiny" if "tiny" in self.model_id else "grounding-dino-base"
        return {
            "name": model_name,
            "description": f"Grounding DINO model for zero-shot object detection ({self.model_id})",
            "model_id": self.model_id,
            "loaded": self.is_loaded,
            "device": self.device,
            "supports_captioning": False,
            "dtype": "bfloat16" if self.use_bfloat16 else "float32"
        }
