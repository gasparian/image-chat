"""Chat state management."""

import requests
import json
import uuid
from pathlib import Path
from PIL import Image
from typing import Optional, List, Dict, Generator
import gradio as gr

from .config import API_BASE_URL, CHAT_IMAGES_DIR
from .visualization import draw_detections_on_image
from .export import create_download_zip


def get_thinking_dots(counter: int) -> str:
    return "".join(["."] * ((counter + 1) % 4))


class ChatState:
    """State for the chat interface."""

    def __init__(self):
        self.session_id: Optional[str] = None
        self.current_image: Optional[Image.Image] = None
        self.last_detections: Optional[List] = None
        self.last_class_vocab: Optional[List] = None
        self.is_processing: bool = False

    def new_session(self) -> str:
        """Create a new chat session."""
        try:
            response = requests.post(f"{API_BASE_URL}/api/chat/new")
            response.raise_for_status()
            data = response.json()
            self.session_id = data["session_id"]
            self.current_image = None
            self.last_detections = None
            self.last_class_vocab = None
            return f"✅ New session created: {self.session_id[:8]}..."
        except Exception as e:
            return f"❌ Error creating session: {str(e)}"

    def upload_image(self, image_path: str) -> tuple[str, Optional[Image.Image]]:
        """Upload an image to the current session."""
        if self.session_id is None:
            return "❌ No active session. Click 'New Chat' first.", None

        try:
            with open(image_path, "rb") as f:
                files = {"file": (Path(image_path).name, f, "image/jpeg")}
                response = requests.post(
                    f"{API_BASE_URL}/api/chat/{self.session_id}/upload",
                    files=files
                )
                response.raise_for_status()
                data = response.json()

            # Load and store the image
            self.current_image = Image.open(image_path)
            self.last_detections = None
            self.last_class_vocab = None

            return f"✅ Image uploaded: {data['image_size']}", self.current_image
        except Exception as e:
            return f"❌ Error uploading image: {str(e)}", None

    def send_message(self, message: str, history: List[Dict]) -> Generator[List[Dict], None, None]:
        """Send a message and get response (messages format)."""
        if self.session_id is None:
            yield history + [{"role": "user", "content": message},
                           {"role": "assistant", "content": "❌ No active session. Click 'New Chat' first."}]
            return

        # User message is already in history (added by caller)

        # Mark as processing
        self.is_processing = True
        response_obj = None

        try:
            # Send message with SSE streaming
            response_obj = requests.post(
                f"{API_BASE_URL}/api/chat/{self.session_id}/message",
                json={"message": message},
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            response_obj.raise_for_status()

            assistant_message = ""
            assistant_msg_index = len(history)  # Track where assistant message will be
            thinking_counter = 0  # Counter for animated thinking dots
            is_streaming = False  # Track if we're in token streaming mode

            # Parse SSE stream
            for line in response_obj.iter_lines():
                if not line:
                    continue

                line = line.decode('utf-8')

                if line.startswith('event:'):
                    event_type = line.split(':', 1)[1].strip()
                elif line.startswith('data:'):
                    data_str = line.split(':', 1)[1].strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event_type == "status":
                        # Update status with animated thinking dots
                        thinking_counter += 1
                        dots = get_thinking_dots(thinking_counter)
                        if len(history) > assistant_msg_index:
                            history[assistant_msg_index] = {"role": "assistant", "content": dots}
                        else:
                            history.append({"role": "assistant", "content": dots})
                        yield history

                    elif event_type == "stream_start":
                        # Starting to stream tokens
                        is_streaming = True
                        assistant_message = ""
                        if len(history) > assistant_msg_index:
                            history[assistant_msg_index] = {"role": "assistant", "content": ""}
                        else:
                            history.append({"role": "assistant", "content": ""})
                        yield history

                    elif event_type == "token":
                        # Append streaming token to message
                        token = data.get("token", "")
                        assistant_message += token
                        if len(history) > assistant_msg_index:
                            history[assistant_msg_index] = {"role": "assistant", "content": assistant_message}
                        else:
                            history.append({"role": "assistant", "content": assistant_message})
                        yield history

                    elif event_type == "result":
                        # Process result
                        if data.get("message_type") == "detection":
                            detections = data.get("detections", [])
                            class_vocab = data.get("class_vocab", [])

                            # Store detections
                            self.last_detections = detections
                            self.last_class_vocab = class_vocab

                            # Draw detections on image and embed in chat
                            if detections and self.current_image:
                                detection_image = draw_detections_on_image(
                                    self.current_image,
                                    detections,
                                    class_vocab
                                )

                                # Save to temporary file for Gradio
                                temp_dir = Path(CHAT_IMAGES_DIR)
                                temp_dir.mkdir(parents=True, exist_ok=True)
                                temp_path = temp_dir / f"detection_{uuid.uuid4().hex}.jpg"
                                detection_image.save(temp_path)
                                detection_image_path = str(temp_path)
                            else:
                                detection_image_path = None

                            assistant_message = f"🎯 Found {len(detections)} objects"

                            # Update text message first
                            if len(history) > assistant_msg_index:
                                history[assistant_msg_index] = {"role": "assistant", "content": assistant_message}
                            else:
                                history.append({"role": "assistant", "content": assistant_message})
                            yield history

                            # Add image as separate message if we have detections
                            if detection_image_path:
                                # Use Gradio component for image
                                history.append({
                                    "role": "assistant",
                                    "content": gr.Image(detection_image_path)
                                })
                                yield history

                                # Create and add download bundle immediately
                                zip_path = create_download_zip(
                                    self.current_image,
                                    detections,
                                    class_vocab
                                )
                                history.append({
                                    "role": "assistant",
                                    "content": gr.File(zip_path, label="💾 Download Results Bundle")
                                })
                                yield history

                        elif data.get("message_type") in ("question_answer", "conversation"):
                            # For question_answer and conversation, if we were streaming, the message is already built
                            # Just ensure it's complete
                            if not is_streaming:
                                answer = data.get("answer", "")
                                assistant_message = answer

                                if len(history) > assistant_msg_index:
                                    history[assistant_msg_index] = {"role": "assistant", "content": assistant_message}
                                else:
                                    history.append({"role": "assistant", "content": assistant_message})
                            # else: message was already built via token streaming
                            yield history

                    elif event_type == "error":
                        error_msg = data.get("error", "Unknown error")
                        if len(history) > assistant_msg_index:
                            history[assistant_msg_index] = {"role": "assistant", "content": f"❌ Error: {error_msg}"}
                        else:
                            history.append({"role": "assistant", "content": f"❌ Error: {error_msg}"})
                        yield history

                    elif event_type == "done":
                        # Final update
                        yield history

        except GeneratorExit:
            # Request was cancelled by user
            # Close the connection immediately
            if response_obj:
                response_obj.close()

            # Remove any partial assistant message
            if len(history) > assistant_msg_index and history[assistant_msg_index]["role"] == "assistant":
                # Remove the incomplete assistant response
                history.pop(assistant_msg_index)

            # Add cancellation message
            history.append({"role": "assistant", "content": "⚠️ Request cancelled"})

            # Don't use any data from this cancelled request
            # (detections already stored in self.last_detections remain from previous successful requests)
            raise  # Re-raise to properly close the generator

        except Exception as e:
            # Close connection on any error
            if response_obj:
                response_obj.close()

            if len(history) > assistant_msg_index:
                history[assistant_msg_index] = {"role": "assistant", "content": f"❌ Error: {str(e)}"}
            else:
                history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
            yield history

        finally:
            # Always mark as not processing when done
            self.is_processing = False
            if response_obj:
                response_obj.close()
