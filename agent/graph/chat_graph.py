from PIL import Image
from openai import OpenAI
from langgraph.graph import StateGraph, END

from agent.structures import (
    DetectionResult, ChatAgentOutput, ChatMessage, CaptionData
)
from agent.mcp_client import MCPClient
from agent.logging_config import get_logger
from agent.callbacks import init_tracker

from agent.graph.state import ChatAgentState
from agent.graph.nodes.preprocessing import preprocess_image_node
from agent.graph.nodes.intent_routing import route_intent_node, should_route_after_intent
from agent.graph.nodes.qna import answer_question_node
from agent.graph.nodes.conversation import conversation_node

logger = get_logger(__name__)


def create_chat_graph():
    workflow = StateGraph(ChatAgentState)

    workflow.add_node("preprocess_image", preprocess_image_node)
    workflow.add_node("route_intent", route_intent_node)
    workflow.add_node("answer_question", answer_question_node)
    workflow.add_node("conversation", conversation_node)

    workflow.set_entry_point("preprocess_image")
    workflow.add_edge("preprocess_image", "route_intent")

    workflow.add_conditional_edges(
        "route_intent",
        should_route_after_intent,
        {
            "question_flow": "answer_question",
            "conversation_flow": "conversation"
        }
    )

    workflow.add_edge("answer_question", END)
    workflow.add_edge("conversation", END)

    app = workflow.compile()

    return app


def run_chat_graph(
    llm_client: OpenAI,
    user_query: str,
    image: Image.Image | None = None,
    conversation_history: list[ChatMessage] | None = None,
    previous_detections: list[DetectionResult] | None = None,
    previous_class_vocab: list[str] | None = None,
    caption_data: CaptionData | None = None,
    mcp_client: MCPClient | None = None
) -> ChatAgentOutput:
    init_tracker()

    app = create_chat_graph()
    config = {"configurable": {"thread_id": "chat_session"}}

    if conversation_history is None:
        conversation_history = []

    if mcp_client is None:
        mcp_client = MCPClient()

    initial_state: ChatAgentState = {
        "user_query": user_query,
        "image": image,
        "llm_client": llm_client,
        "mcp_client": mcp_client,
        "conversation_history": conversation_history,
        "intent": None,
        "caption_data": caption_data,
        "previous_detections": previous_detections,
        "previous_class_vocab": previous_class_vocab,
        "phrases": None,
        "label_vocab": None,
        "grounding_dino_detections": None,
        "detections": None,
        "segmentation_masks": None,
        "final_output": None,
        "scale_factor": 1.0,
        "original_image_size": image.size if image else None,
        "image_metadata": None,
        "human_feedback": None,
    }

    result = None
    for event in app.stream(initial_state, config, stream_mode="values"):
        result = event

    return result["final_output"]
