import torch
from PIL import Image
from typing import Optional
from transformers import AutoProcessor, AutoModelForCausalLM

from detection_mcp_server.detectors.base import BaseModel
from agent.logging_config import get_logger

logger = get_logger(__name__)


class Florence2Detector(BaseModel):
    def __init__(
        self,
        model_id: str = "microsoft/florence-2-base",
        device: str = "cpu",
        use_bfloat16: bool = True
    ):
        super().__init__(device, use_bfloat16)
        self.model_id = model_id
        self._processor: Optional[object] = None

    def load(self) -> None:
        if self._model is not None:
            return

        logger.info("Loading Florence-2 model", model_id=self.model_id)

        dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(self.device)

        dtype_str = "bfloat16" if self.use_bfloat16 else "float32"
        logger.info("Florence-2 model loaded", device=self.device, dtype=dtype_str)

    def caption(
        self,
        image: Image.Image,
        task: str = "<CAPTION>"
    ) -> str:
        """Generate caption for an image using Florence-2.

        Args:
            image: PIL Image to caption
            task: Task prompt (e.g., "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>")

        Returns:
            Generated caption as string
        """
        if self._model is None:
            self.load()

        if image is None:
            raise ValueError("Image cannot be None")

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(
            text=task,
            images=image,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
                early_stopping=True,
                use_cache=False
            )

        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        parsed_answer = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )

        caption_result = parsed_answer.get(task, "")

        return caption_result

    def get_info(self) -> dict:
        model_name = "florence-2-base" if "base" in self.model_id else "florence-2-large"
        return {
            "name": model_name,
            "description": f"Florence-2 vision-language model for captioning ({self.model_id})",
            "model_id": self.model_id,
            "loaded": self.is_loaded,
            "device": self.device,
            "supports_captioning": True,
            "dtype": "bfloat16" if self.use_bfloat16 else "float32"
        }
