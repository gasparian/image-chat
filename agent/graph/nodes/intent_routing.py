import json
import time
from enum import Enum

from pydantic import BaseModel, Field
from agent.callbacks import track_step, init_tracker
from agent.logging_config import get_logger
from agent.graph.state import ChatAgentState
from agent.config import CFG
from agent.callbacks import log_tokens

logger = get_logger(__name__)

class IntentType(str, Enum):
    DETECTION = "detection"
    QUESTION = "question"
    CONVERSATION = "conversation"


class IntentClassification(BaseModel):
    intent: IntentType = Field(
        description="The classified intent type: 'detection' or 'question'"
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )


SYSTEM = """You are an intent classifier for a vision AI assistant.
Your job is to classify user queries into three categories:
1. DETECTION: User wants to detect/find/locate objects in an image
2. QUESTION: User has a question about the image content
3. CONVERSATION: General conversation, greetings, or requests for guidance on using the app

Output JSON only."""

INSTRUCTION = """User query: "{query}"

Classify this query as:
- "detection": if user wants to detect, find, locate, or identify specific objects
- "question": if user is asking about what's in the image, asking questions about image content, or requesting a description
- "conversation": if user is greeting, asking for help/guidance, or having general conversation not related to image analysis

Output JSON format:
{{"intent": "detection" or "question" or "conversation", "reasoning": "brief explanation"}}"""


def route_intent(llm_client, query: str) -> IntentType:
    init_tracker()

    prompt = INSTRUCTION.format(query=query)

    try:
        start_time = time.time()

        rsp = llm_client.chat.completions.create(
            model=CFG.llm_model,
            temperature=0.0,
            max_tokens=128,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        llm_duration = time.time() - start_time
        logger.info("Intent routing complete", duration_sec=round(llm_duration, 2))

        # Try to log tokens, but don't fail if tracker isn't available
        try:
            if hasattr(rsp, 'usage') and rsp.usage:
                log_tokens(rsp.usage.total_tokens)
        except RuntimeError:
            pass  # Tracker not available in this context

        text = rsp.choices[0].message.content

        if not text or text.strip() == "":
            raise ValueError("Empty response from LLM")

        data = json.loads(text.strip())
        result = IntentClassification.model_validate(data)

        logger.info("Intent classified", intent=result.intent.value, reasoning=result.reasoning)

        return result.intent

    except Exception as e:
        logger.warning("Intent classification failed, using fallback", error_type=type(e).__name__, error=str(e))
        return _fallback_classify_intent(query)


def _fallback_classify_intent(query: str) -> IntentType:
    query_lower = query.lower()

    conversation_keywords = [
        'hello', 'hi', 'hey', 'help', 'how to use', 'how do i',
        'guide', 'tutorial', 'instructions', 'thanks', 'thank you',
        'goodbye', 'bye', 'what can you do', 'how does this work'
    ]

    detection_keywords = [
        'detect', 'find', 'locate', 'identify', 'search', 'look for',
        'count', 'how many', 'segment', 'show me', 'mark', 'highlight'
    ]

    question_keywords = [
        'what', 'where', 'why', 'how', 'when', 'who',
        'describe', 'tell me', 'explain', 'is there',
        'are there', 'can you see', 'do you see'
    ]

    conversation_score = sum(1 for kw in conversation_keywords if kw in query_lower)
    detection_score = sum(1 for kw in detection_keywords if kw in query_lower)
    question_score = sum(1 for kw in question_keywords if kw in query_lower)

    if conversation_score > 0 and conversation_score >= detection_score:
        return IntentType.CONVERSATION

    if any(query_lower.startswith(qw) for qw in ['what', 'where', 'why', 'how', 'when', 'who']):
        return IntentType.QUESTION

    if detection_score > question_score:
        return IntentType.DETECTION
    elif question_score > detection_score:
        return IntentType.QUESTION
    else:
        return IntentType.DETECTION


def route_intent_node(state: ChatAgentState) -> ChatAgentState:
    with track_step("route_intent"):
        user_query = state["user_query"]
        llm_client = state["llm_client"]

        intent = route_intent(llm_client, user_query)

        logger.info("Query routed", intent=intent.value)

    return {
        **state,
        "intent": intent.value,
    }


def should_route_after_intent(state: ChatAgentState) -> str:
    intent = state.get("intent")

    if intent == IntentType.DETECTION.value:
        return "detection_flow"
    elif intent == IntentType.QUESTION.value:
        return "question_flow"
    elif intent == IntentType.CONVERSATION.value:
        return "conversation_flow"
    else:
        return "conversation_flow"
