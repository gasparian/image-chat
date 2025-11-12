from pydantic import BaseModel, Field, ConfigDict


class DetectionRequest(BaseModel):
    """Request for object detection."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_base64": "iVBORw0KGgoAAAANS...",
                "phrases": ["car", "person", "dog"],
                "threshold": 0.25
            }
        }
    )

    image_base64: str = Field(..., description="Base64-encoded image")
    phrases: list[str] = Field(..., description="Detection phrases/prompts")
    threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Detection confidence threshold")


class BoundingBox(BaseModel):
    """Bounding box in xyxy format."""
    model_config = ConfigDict(extra='forbid')

    xmin: float = Field(..., description="Minimum x coordinate")
    ymin: float = Field(..., description="Minimum y coordinate")
    xmax: float = Field(..., description="Maximum x coordinate")
    ymax: float = Field(..., description="Maximum y coordinate")


class Detection(BaseModel):
    """Single object detection result."""
    model_config = ConfigDict(extra='forbid')

    bbox: list[float] = Field(..., description="Bounding box [xmin, ymin, xmax, ymax]", min_length=4, max_length=4)
    score: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    label: str = Field(..., min_length=1, description="Detected object label")


class DetectionResponse(BaseModel):
    """Response containing detection results."""
    model_config = ConfigDict(extra='forbid')

    detections: list[Detection] = Field(..., description="List of detections")
    model: str = Field(..., description="Model used for detection")
    num_detections: int = Field(..., ge=0, description="Number of detections")


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(extra='forbid')

    status: str = Field(..., description="Health status")
    models_available: int = Field(..., ge=0, description="Number of available models")


class SegmentationRequest(BaseModel):
    """Request for image segmentation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_base64": "iVBORw0KGgoAAAANS...",
                "bboxes": [[100, 100, 200, 200], [300, 150, 450, 300]]
            }
        }
    )

    image_base64: str = Field(..., description="Base64-encoded image")
    bboxes: list[list[float]] = Field(
        ...,
        description="List of bounding boxes [xmin, ymin, xmax, ymax] to segment"
    )


class Polygon(BaseModel):
    """Polygon representation as list of [x, y] coordinates."""
    model_config = ConfigDict(extra='forbid')

    points: list[list[float]] = Field(
        ...,
        description="List of [x, y] coordinates defining the polygon"
    )


class SegmentationMask(BaseModel):
    """Single segmentation mask result."""
    model_config = ConfigDict(extra='forbid')

    bbox: list[float] = Field(..., min_length=4, max_length=4, description="Input bounding box [xmin, ymin, xmax, ymax]")
    polygon: Polygon = Field(..., description="Segmentation mask as polygon")
    score: float = Field(..., ge=0.0, le=1.0, description="Segmentation confidence score")


class SegmentationResponse(BaseModel):
    """Response containing segmentation results."""
    model_config = ConfigDict(extra='forbid')

    masks: list[SegmentationMask] = Field(..., description="List of segmentation masks")
    model: str = Field(..., description="Model used for segmentation")
    num_masks: int = Field(..., ge=0, description="Number of masks")


class CaptionRequest(BaseModel):
    """Request for image captioning."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_base64": "iVBORw0KGgoAAAANS...",
                "task": "<CAPTION>"
            }
        }
    )

    image_base64: str = Field(..., description="Base64-encoded image")
    task: str = Field(default="<CAPTION>", description="Caption task type")


class CaptionResponse(BaseModel):
    """Response containing caption result."""
    model_config = ConfigDict(extra='forbid')

    caption: str = Field(..., min_length=1, description="Generated caption")
    model: str = Field(..., description="Model used for captioning")
