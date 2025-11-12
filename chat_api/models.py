from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal

from agent.structures import DetectionResult, Provenance


class NewChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: Optional[str] = Field(None, description="Optional custom session ID")


class NewChatResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="Response message")


class ChatMessageRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(..., min_length=1, description="User's message/query")


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
    session_id: str = Field(..., description="Session ID")
    message_type: Literal["detection", "question_answer", "conversation"] = Field(..., description="Type of response")
    detections: Optional[list[DetectionResult]] = Field(None, description="Detection results (detection type)")
    class_vocab: Optional[list[str]] = Field(None, description="Class vocabulary (detection type)")
    visualization_path: Optional[str] = Field(None, description="Path to visualization image")
    answer: Optional[str] = Field(None, description="Answer text (question_answer and conversation types)")
    caption: Optional[str] = Field(None, description="Image caption (question_answer type)")
    provenance: Provenance = Field(..., description="Execution provenance")


class UploadImageResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="Response message")
    image_size: tuple[int, int] = Field(..., description="Image dimensions (width, height)")


class SessionStatusResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(..., description="Session ID")
    has_image: bool = Field(..., description="Whether session has an image")
    image_path: Optional[str] = Field(None, description="Path to uploaded image")
    message_count: int = Field(..., ge=0, description="Number of messages in conversation")
    has_detections: bool = Field(..., description="Whether session has detection results")
    has_caption: bool = Field(..., description="Whether session has image caption")


class StreamEvent(BaseModel):
    model_config = ConfigDict(extra='allow')
    event: Literal["status", "result", "error", "done"] = Field(..., description="Event type")
    data: dict = Field(..., description="Event data")
