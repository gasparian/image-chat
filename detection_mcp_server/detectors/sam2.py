import torch
import numpy as np
from PIL import Image
from typing import Optional, List
from skimage import measure
from skimage.measure import approximate_polygon
from transformers import Sam2Processor, Sam2Model

from detection_mcp_server.detectors.base import BaseModel
from agent.logging_config import get_logger

logger = get_logger(__name__)


class SAM2Segmenter(BaseModel):
    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-small",
        device: str = "cpu",
        use_bfloat16: bool = True,
    ):
        super().__init__(device, use_bfloat16)
        self.model_id = model_id
        self._processor: Optional[object] = None

    def load(self) -> None:
        if self._model is not None:
            return

        logger.info("Loading SAM 2.1 model", model_id=self.model_id)

        if not hasattr(torch.compiler, 'is_compiling'):
            torch.compiler.is_compiling = lambda: False

        self._processor = Sam2Processor.from_pretrained(self.model_id)

        if self.use_bfloat16:
            self._model = Sam2Model.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16,
                ignore_mismatched_sizes=True
            ).to(self.device)
        else:
            self._model = Sam2Model.from_pretrained(
                self.model_id,
                ignore_mismatched_sizes=True
            ).to(self.device)

        logger.info("SAM 2.1 model loaded", device=self.device, use_bfloat16=self.use_bfloat16)

    def segment_from_boxes(
        self,
        image: Image.Image,
        bboxes: List[List[float]]
    ) -> list[dict]:
        if self._model is None:
            self.load()

        if not bboxes:
            return []

        input_boxes = [[bbox for bbox in bboxes]]

        inputs = self._processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt"
        )

        target_dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32

        if self.device == "mps":
            inputs = {
                k: v.to(self.device, dtype=torch.float32) if v.dtype in [torch.float64, torch.bfloat16] else v.to(self.device)
                for k, v in inputs.items()
            }
        else:
            inputs = {
                k: v.to(self.device, dtype=target_dtype) if v.dtype in [torch.float32, torch.float64] else v.to(self.device)
                for k, v in inputs.items()
            }

        with torch.no_grad():
            outputs = self._model(**inputs)

        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu()
        )[0]

        iou_scores = outputs.iou_scores.cpu().float().squeeze().numpy()
        if len(iou_scores.shape) == 0:
            iou_scores = np.array([iou_scores.item()])
        elif len(iou_scores.shape) == 1:
            iou_scores = np.array([iou_scores.max()])
        else:
            iou_scores = iou_scores.max(axis=1)

        results = []

        for i, (mask, bbox) in enumerate(zip(masks, bboxes)):
            if len(mask.shape) == 3:
                mask = mask[0]

            mask_np = mask.cpu().numpy().astype(bool)

            polygon = self._mask_to_polygon(mask_np)

            if polygon is not None:
                score = float(iou_scores[i]) if i < len(iou_scores) else 1.0
                results.append({
                    "bbox": bbox,
                    "polygon": polygon,
                    "score": score
                })

        return results

    def _mask_to_polygon(self, mask: np.ndarray, tolerance: float = 2.0) -> Optional[List[List[float]]]:
        contours = measure.find_contours(mask, 0.5)
        if not contours:
            return None
        largest_contour = max(contours, key=len)
        simplified = approximate_polygon(largest_contour, tolerance=tolerance)
        polygon = [[float(point[1]), float(point[0])] for point in simplified]
        return polygon

    def get_info(self) -> dict:
        if "tiny" in self.model_id:
            model_name = "sam2.1-hiera-tiny"
        elif "small" in self.model_id:
            model_name = "sam2.1-hiera-small"
        elif "base-plus" in self.model_id:
            model_name = "sam2.1-hiera-base-plus"
        elif "large" in self.model_id:
            model_name = "sam2.1-hiera-large"
        else:
            model_name = "sam2.1-hiera-small"

        dtype = "bfloat16" if self.use_bfloat16 else "float32"

        return {
            "name": model_name,
            "description": f"SAM 2.1 (Segment Anything Model) for instance segmentation - 6x faster and more accurate ({self.model_id})",
            "model_id": self.model_id,
            "loaded": self.is_loaded,
            "device": self.device,
            "supports_segmentation": True,
            "requires_bbox_input": True,
            "dtype": dtype
        }
