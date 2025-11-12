"""Gradio interface setup and handlers."""

import gradio as gr
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Tuple, Generator

from .state import ChatState
from .config import GRADIO_THEME


# Global state
chat_state = ChatState()


def new_chat_handler() -> Tuple[List[Dict], None]:
    """Handle new chat button click - clear session, chat, and image."""
    result = chat_state.new_session()

    # If session creation failed, show error in chat
    if result and "❌" in result:
        return [{"role": "assistant", "content": result}], None

    # Return: empty history, clear multimodal input
    return [], None


def send_message_handler(
    message_dict,
    history: List[Dict]
) -> Generator[Tuple[List[Dict], Dict, Dict, Dict], None, None]:
    """Handle sending a message with multimodal input."""
    # Extract text and files from message
    if isinstance(message_dict, dict):
        text = message_dict.get("text", "")
        files = message_dict.get("files", [])
    else:
        text = message_dict if message_dict else ""
        files = []

    if not text.strip():
        yield history, gr.update(value=None, interactive=True), gr.update(interactive=True), gr.update(interactive=False)
        return

    # Handle image upload if provided
    uploaded_image_path = None
    if files:
        # Take the first image
        image_path = files[0] if isinstance(files[0], str) else files[0].get("path")

        # Load and upload image
        try:
            image = Image.open(image_path)

            # Save temporarily
            temp_path = Path(".local/temp_upload.jpg")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(temp_path)

            # Upload to session
            upload_msg, uploaded_img = chat_state.upload_image(str(temp_path))
            if uploaded_img is None:
                # Upload failed
                error_history = history + [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": upload_msg}
                ]
                yield error_history, gr.update(value=None, interactive=True), gr.update(interactive=True), gr.update(interactive=False)
                return

            # Store the image path for display in chat
            uploaded_image_path = image_path
        except Exception as e:
            error_history = history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": f"❌ Error loading image: {str(e)}"}
            ]
            yield error_history, gr.update(value=None, interactive=True), gr.update(interactive=True), gr.update(interactive=False)
            return

    # Add user's text message to history
    history = history + [{"role": "user", "content": text}]

    # If image was just uploaded, show it in chat
    if uploaded_image_path:
        history = history + [{"role": "user", "content": gr.Image(uploaded_image_path)}]

    # Disable input and new button, enable cancel button while processing
    yield history, gr.update(value=None, interactive=False), gr.update(interactive=False), gr.update(interactive=True)

    # Use generator for streaming
    last_history = history
    try:
        for h in chat_state.send_message(text, history):
            # Keep input and new button disabled, only cancel button is active
            last_history = h
            yield h, gr.update(value=None, interactive=False), gr.update(interactive=False), gr.update(interactive=True)
    except GeneratorExit:
        # User cancelled - re-enable inputs and new button, disable cancel button
        # The cancellation message is already in history from ChatState
        yield last_history, gr.update(value=None, interactive=True), gr.update(interactive=True), gr.update(interactive=False)
        raise
    finally:
        # Always re-enable inputs when done
        # Final yield to re-enable everything after successful completion
        if chat_state.is_processing is False:
            yield last_history, gr.update(value=None, interactive=True), gr.update(interactive=True), gr.update(interactive=False)


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    with gr.Blocks(
        title="image chat",
        theme=gr.themes.Soft(),
        css="""
        .center-title { text-align: center; font-size: 1.2em; font-weight: 500; margin: 0.5em 0; }
        """
    ) as demo:
        # Simple centered title
        gr.HTML('<div class="center-title">image chat</div>')

        # Main chat interface
        chatbot = gr.Chatbot(
            show_label=False,
            height=600,
            show_copy_button=False,
            type="messages"
        )

        # Input area with multimodal textbox and buttons
        with gr.Row():
            cancel_btn = gr.Button("cancel", variant="stop", scale=1, min_width=30, interactive=False)
            multimodal_input = gr.MultimodalTextbox(
                file_types=["image"],
                file_count="single",
                placeholder="Type a message or click + to upload an image...",
                show_label=False,
                submit_btn=True,
                scale=10
            )
            new_chat_btn = gr.Button("new", scale=1, min_width=30)

        # Event handlers
        # Auto-create new chat session on page load
        demo.load(
            fn=new_chat_handler,
            outputs=[chatbot, multimodal_input]
        )

        # New chat clears everything
        new_chat_btn.click(
            fn=new_chat_handler,
            outputs=[chatbot, multimodal_input]
        )

        # Send message with multimodal input (text + optional image)
        # Outputs: chatbot, input (value+interactivity), new button interactivity, cancel button interactivity
        submit_event = multimodal_input.submit(
            fn=send_message_handler,
            inputs=[multimodal_input, chatbot],
            outputs=[chatbot, multimodal_input, new_chat_btn, cancel_btn]
        )

        # Cancel button stops the ongoing request and notifies server
        def cancel_handler():
            """Handle cancellation - send cancel request to server."""
            import requests
            from .config import API_BASE_URL

            if chat_state.session_id:
                try:
                    requests.post(
                        f"{API_BASE_URL}/api/chat/{chat_state.session_id}/cancel",
                        timeout=1
                    )
                except:
                    pass  # Ignore errors, the client-side cancellation is more important

            return None

        cancel_btn.click(
            fn=cancel_handler,
            cancels=[submit_event],
            show_progress=False
        )

    # Enable queue for proper streaming and stop button support
    demo.queue()

    return demo
