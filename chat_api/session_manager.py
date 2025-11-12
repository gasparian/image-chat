import uuid
from typing import Dict, Optional, Union
from pathlib import Path
from PIL import Image
from openai import OpenAI

from agent.config import CFG
from agent.structures import ChatMessage, CaptionData, DetectionResult, ChatAgentOutput


class ChatSessionData:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.image: Optional[Image.Image] = None
        self.image_path: Optional[str] = None
        self.conversation_history: list[ChatMessage] = []
        self.last_detections: Optional[list[DetectionResult]] = None
        self.last_class_vocab: Optional[list[str]] = None
        self.caption_data: Optional[CaptionData] = None
        self.is_cancelled: bool = False  # Flag for cancellation
        self.llm_client = OpenAI(
            base_url=CFG.ollama_base_url,
            api_key="ollama"
        )

    def has_image(self) -> bool:
        return self.image is not None

    def set_image(self, image: Image.Image, image_path: str):
        self.image = image
        self.image_path = image_path
        self.conversation_history = []
        self.last_detections = None
        self.last_class_vocab = None
        self.caption_data = None

    def add_message(self, role: str, content: str):
        self.conversation_history.append(ChatMessage(role=role, content=content))

    def update_state(self, result: Union[dict, ChatAgentOutput]):
        result_type = result.type if isinstance(result, ChatAgentOutput) else result["type"]

        if result_type == "detection":
            if isinstance(result, ChatAgentOutput):
                self.last_detections = result.detections
                self.last_class_vocab = result.class_vocab
            else:
                self.last_detections = result.get("detections")
                self.last_class_vocab = result.get("class_vocab")
        else:
            caption = result.caption if isinstance(result, ChatAgentOutput) else result.get("caption")
            if caption:
                self.caption_data = CaptionData(
                    caption=caption,
                    task="<MORE_DETAILED_CAPTION>",
                    model="florence-2-base"
                )

    def cancel(self):
        self.is_cancelled = True

    def reset_cancellation(self):
        self.is_cancelled = False

    def reset(self):
        self.conversation_history = []
        self.last_detections = None
        self.last_class_vocab = None
        self.caption_data = None
        self.is_cancelled = False


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, ChatSessionData] = {}

    def create_session(self, session_id: Optional[str] = None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id in self._sessions:
            return session_id

        self._sessions[session_id] = ChatSessionData(session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[ChatSessionData]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
