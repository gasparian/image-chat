from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class DetectionResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    bbox: list[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    class_id: int = Field(..., ge=0, description="Class ID index")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    label: str = Field(..., min_length=1, description="Class label name")
    polygon_mask: Optional[list[list[float]]] = Field(None, description="Polygon mask as list of [x, y] coordinates")


class Provenance(BaseModel):
    model_config = ConfigDict(extra='allow')
    steps: list[str] = Field(default_factory=list, description="Pipeline steps executed")


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    detections: list[DetectionResult] = Field(..., description="List of detected objects")
    class_vocab: list[str] = Field(..., description="Vocabulary of class labels")
    provenance: Provenance = Field(..., description="Execution provenance")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., min_length=1, description="Message content")


class CaptionData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    caption: str = Field(..., min_length=1, description="Generated caption")
    detailed_caption: Optional[str] = Field(None, description="More detailed caption")
    task: str = Field(..., description="Caption task type")
    model: str = Field(..., description="Model used for captioning")


class QuestionAnswerOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer: str = Field(..., min_length=1, description="Answer to the question")
    caption: str = Field(..., description="Image caption used")
    provenance: Provenance = Field(..., description="Execution provenance")


class ChatAgentOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal["detection", "question_answer", "conversation"] = Field(..., description="Output type")
    detections: Optional[list[DetectionResult]] = Field(None, description="Detection results (detection type only)")
    class_vocab: Optional[list[str]] = Field(None, description="Class vocabulary (detection type only)")
    answer: Optional[str] = Field(None, description="Answer text or error message for all types")
    caption: Optional[str] = Field(None, description="Image caption (question_answer type only)")
    provenance: Provenance = Field(..., description="Execution provenance")
