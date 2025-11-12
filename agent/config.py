from pydantic import BaseModel


class Cfg(BaseModel):
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434/v1"
    max_queries: int = 4
    gd_box_thr: float = 0.25
    global_resize_long_side: int = 512
    min_area: int = 72 * 72
    nms_iou_threshold: float = 0.5
    nms_score_threshold: float = 0.1
    containment_threshold: float = 0.85
    max_conversation_turns: int = 10


CFG = Cfg()
