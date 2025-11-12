from agent.config import CFG
from agent.structures import (
    ChatAgentOutput, ChatMessage, CaptionData, Provenance
)
from agent.callbacks import track_step
from agent.conversation_utils import trim_conversation_history
from agent.logging_config import get_logger

from agent.graph.state import ChatAgentState

logger = get_logger(__name__)


SYSTEM = """You are a helpful vision AI assistant.
You have access to:
1. A detailed image caption
2. Image metadata (dimensions, format, EXIF data if available)
3. Previous conversation history (if available)

Answer the user's question based on this information.
Be concise and helpful. If the information doesn't contain enough details to answer confidently, say so."""

INSTRUCTION = """{context}

User question: {user_query}

Please answer the user's question based on the available information."""


def caption_image_node(state: ChatAgentState) -> ChatAgentState:
    with track_step("caption_image"):
        image = state.get("image")

        if image is None:
            logger.warning("No image available for captioning")
            provenance = Provenance(steps=["preprocess_image", "route_intent", "caption_image"])
            output = ChatAgentOutput(
                type="question_answer",
                answer="Please upload an image first to ask questions about it.",
                caption=None,
                provenance=provenance
            )
            return {
                **state,
                "final_output": output,
            }

        existing_caption = state.get("caption_data")
        if existing_caption:
            caption_text = existing_caption.caption if isinstance(existing_caption, CaptionData) else existing_caption["caption"]
            logger.info("Using cached caption", caption_preview=caption_text[:100])
            return state

        client = state["mcp_client"]

        try:
            caption = client.caption(
                image=image,
                task="<MORE_DETAILED_CAPTION>"
            )
            logger.info("Caption generated", caption_preview=caption[:100])

            caption_data = CaptionData(
                caption=caption,
                task="<MORE_DETAILED_CAPTION>",
                model="florence-2-large"
            )

        except Exception as e:
            logger.error("Captioning failed", error=str(e))
            caption_data = CaptionData(
                caption="Failed to generate caption",
                task="<MORE_DETAILED_CAPTION>",
                model="florence-2-base"
            )

    return {
        **state,
        "caption_data": caption_data,
    }


def answer_question_node(state: ChatAgentState) -> ChatAgentState:
    with track_step("answer_question"):
        if state.get("final_output"):
            return state

        user_query = state["user_query"]
        caption_data = state["caption_data"]
        llm_client = state["llm_client"]
        conversation_history = state.get("conversation_history", [])
        image_metadata = state.get("image_metadata")

        if isinstance(caption_data, CaptionData):
            caption = caption_data.caption
        else:
            caption = caption_data["caption"]

        context_parts = [f"Image caption: {caption}"]

        if image_metadata:
            metadata_summary = f"\nImage metadata:"
            metadata_summary += f"\n- Dimensions: {image_metadata['width']}x{image_metadata['height']} pixels"
            metadata_summary += f"\n- Aspect ratio: {image_metadata['aspect_ratio']:.2f}"
            metadata_summary += f"\n- Color mode: {image_metadata['mode']} ({image_metadata['channels']} channels)"
            if image_metadata.get('format'):
                metadata_summary += f"\n- Format: {image_metadata['format']}"

            if image_metadata.get('exif'):
                exif = image_metadata['exif']
                metadata_summary += "\n- EXIF data:"

                useful_fields = [
                    'Make', 'Model', 'DateTime', 'DateTimeOriginal',
                    'ExposureTime', 'FNumber', 'ISOSpeedRatings', 'FocalLength',
                    'Flash', 'WhiteBalance', 'Orientation'
                ]

                for field in useful_fields:
                    if field in exif:
                        metadata_summary += f"\n  * {field}: {exif[field]}"

            context_parts.append(metadata_summary)
            logger.info("Including image metadata in context")

        context = "\n".join(context_parts)

        trimmed_history = trim_conversation_history(conversation_history)

        messages = [{"role": "system", "content": SYSTEM}]

        if trimmed_history:
            logger.info("Including conversation history", message_count=len(trimmed_history))
            for msg in trimmed_history:
                if isinstance(msg, ChatMessage):
                    messages.append({"role": msg.role, "content": msg.content})
                else:
                    messages.append({"role": msg["role"], "content": msg["content"]})

        prompt = INSTRUCTION.format(context=context, user_query=user_query)

        messages.append({"role": "user", "content": prompt})

        try:
            rsp = llm_client.chat.completions.create(
                model=CFG.llm_model,
                temperature=0.3,
                max_tokens=512,
                messages=messages
            )

            answer = rsp.choices[0].message.content.strip()
            logger.info("Answer generated", answer_preview=answer[:100])

        except Exception as e:
            logger.error("Question answering failed", error=str(e))
            answer = "I'm sorry, I couldn't generate an answer to your question."

        steps = ["preprocess_image", "route_intent", "caption_image", "answer_question"]

        provenance = Provenance(steps=steps)
        output = ChatAgentOutput(
            type="question_answer",
            answer=answer,
            caption=caption,
            provenance=provenance
        )

    return {
        **state,
        "final_output": output,
    }
