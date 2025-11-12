from agent.config import CFG
from agent.structures import ChatAgentOutput, ChatMessage, Provenance
from agent.callbacks import track_step
from agent.conversation_utils import trim_conversation_history
from agent.logging_config import get_logger

from agent.graph.state import ChatAgentState

logger = get_logger(__name__)


SYSTEM = """You are a friendly AI assistant for an image analysis application.

Your role is to:
1. Guide users on how to use the application
2. Answer questions about features and capabilities
3. Have friendly conversations with users
4. Help troubleshoot basic issues

Key features of this application:
- Object detection: Users can upload images and ask to detect specific objects
- Image Q&A: Users can ask questions about images they've uploaded
- Interactive chat: Users can have conversations about images

To use the application:
1. Upload an image through the interface
2. Ask questions like "What's in this image?" or "Detect all cars"
3. You can refine results with follow-up questions

Be concise, friendly, and helpful. If users haven't uploaded an image yet,
gently remind them to upload one to use detection or Q&A features."""

INSTRUCTION = """User message: {user_query}

Respond to the user naturally and helpfully. If they're asking about how to use
the app or what it can do, provide clear guidance. If they're just chatting,
respond conversationally."""


def conversation_node(state: ChatAgentState) -> ChatAgentState:
    with track_step("conversation"):
        user_query = state["user_query"]
        llm_client = state["llm_client"]
        conversation_history = state.get("conversation_history", [])

        trimmed_history = trim_conversation_history(conversation_history)

        messages = [{"role": "system", "content": SYSTEM}]

        if trimmed_history:
            logger.info("Including conversation history", message_count=len(trimmed_history))
            for msg in trimmed_history:
                if isinstance(msg, ChatMessage):
                    messages.append({"role": msg.role, "content": msg.content})
                else:
                    messages.append({"role": msg["role"], "content": msg["content"]})

        prompt = INSTRUCTION.format(user_query=user_query)
        messages.append({"role": "user", "content": prompt})

        try:
            rsp = llm_client.chat.completions.create(
                model=CFG.llm_model,
                temperature=0.7,
                max_tokens=512,
                messages=messages
            )

            answer = rsp.choices[0].message.content.strip()
            logger.info("Conversation response generated", answer_preview=answer[:100])

        except Exception as e:
            logger.error("Conversation generation failed", error=str(e))
            answer = (
                "Hello! I'm your image analysis assistant. "
                "To get started, upload an image and ask me to detect objects "
                "or answer questions about it. For example, try 'Detect all people' "
                "or 'What's in this image?'"
            )

        provenance = Provenance(steps=["preprocess_image", "route_intent", "conversation"])
        output = ChatAgentOutput(
            type="conversation",
            answer=answer,
            provenance=provenance
        )

    return {
        **state,
        "final_output": output,
    }
