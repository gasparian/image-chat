from typing import List
from agent.structures import ChatMessage
from agent.config import CFG


def trim_conversation_history(
    conversation_history: List[ChatMessage],
    max_turns: int | None = None
) -> List[ChatMessage]:
    if not conversation_history:
        return []

    if max_turns is None:
        max_turns = CFG.max_conversation_turns

    if len(conversation_history) <= max_turns * 2:
        return conversation_history

    max_messages = max_turns * 2
    trimmed = conversation_history[-max_messages:]

    if trimmed:
        first_msg = trimmed[0]
        first_role = first_msg.role if hasattr(first_msg, 'role') else first_msg["role"]
        if first_role == "assistant":
            trimmed = trimmed[1:]

    return trimmed


def format_conversation_for_context(
    conversation_history: List[ChatMessage],
    max_turns: int | None = None
) -> str:
    trimmed = trim_conversation_history(conversation_history, max_turns)

    if not trimmed:
        return ""

    formatted_lines = ["Previous conversation:"]
    for msg in trimmed:
        if hasattr(msg, 'role'):
            role = msg.role.capitalize()
            content = msg.content
        else:
            role = msg["role"].capitalize()
            content = msg["content"]
        formatted_lines.append(f"{role}: {content}")

    return "\n".join(formatted_lines)
