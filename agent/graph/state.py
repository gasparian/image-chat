from typing import TypedDict, Tuple, Any
from PIL import Image
from openai import OpenAI

from agent.mcp_client import MCPClient
from agent.structures import (
    DetectionResult, ChatAgentOutput, ChatMessage, CaptionData,
)

class ImageMetadata(TypedDict, total=False):
    width: int
    height: int
    channels: int
    mode: str
    format: str | None
    exif: dict[str, Any]
    aspect_ratio: float
    total_pixels: int


class ChatAgentState(TypedDict):
    user_query: str
    image: Image.Image | None
    llm_client: OpenAI
    mcp_client: MCPClient
    conversation_history: list[ChatMessage]
    intent: str | None
    caption_data: CaptionData | None
    previous_detections: list[DetectionResult] | None
    previous_class_vocab: list[str] | None
    phrases: list[str] | None
    label_vocab: list[str] | None
    grounding_dino_detections: list[dict] | None
    detections: list[dict] | None
    segmentation_masks: list[dict] | None
    final_output: ChatAgentOutput | None
    scale_factor: float
    original_image_size: Tuple[int, int] | None
    image_metadata: ImageMetadata | None
    human_feedback: str | None