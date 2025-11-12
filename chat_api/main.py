import asyncio
import json
import io
import traceback
from pathlib import Path
from typing import AsyncGenerator
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

from chat_api.session_manager import get_session_manager
from chat_api.models import (
    NewChatRequest, NewChatResponse,
    ChatMessageRequest, ChatMessageResponse,
    UploadImageResponse, SessionStatusResponse
)
from agent.graph.chat_graph import run_chat_graph
from agent.logging_config import get_logger

logger = get_logger(__name__)


app = FastAPI(
    title="Chat API Server",
    description="SSE-based chat API for image-chat",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat/new", response_model=NewChatResponse)
async def create_new_chat(request: NewChatRequest = NewChatRequest()) -> NewChatResponse:
    manager = get_session_manager()
    session_id = manager.create_session(request.session_id)

    return NewChatResponse(
        session_id=session_id,
        message="New chat session created"
    )


@app.post("/api/chat/{session_id}/upload", response_model=UploadImageResponse)
async def upload_image(session_id: str, file: UploadFile = File(...)) -> UploadImageResponse:
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        temp_dir = Path(".local/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_stem = Path(file.filename).stem
        filename_ext = Path(file.filename).suffix
        image_path = temp_dir / f"{session_id}_{timestamp}_{filename_stem}{filename_ext}"
        image.save(image_path)

        session.set_image(image, str(image_path))

        return UploadImageResponse(
            session_id=session_id,
            message=f"Image uploaded successfully: {file.filename}",
            image_size=(image.width, image.height)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")


@app.get("/api/chat/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str) -> SessionStatusResponse:
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionStatusResponse(
        session_id=session_id,
        has_image=session.has_image(),
        image_path=session.image_path,
        message_count=len(session.conversation_history),
        has_detections=session.last_detections is not None,
        has_caption=session.caption_data is not None
    )


async def stream_chat_response(session_id: str, message: str) -> AsyncGenerator[str, None]:
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if session is None:
        yield f"event: error\ndata: {json.dumps({'error': 'Session not found'})}\n\n"
        return

    try:
        session.reset_cancellation()
        session.add_message("user", message)

        yield f"event: status\ndata: {json.dumps({'message': 'Analyzing query...'})}\n\n"
        await asyncio.sleep(0.05)

        loop = asyncio.get_event_loop()

        if session.is_cancelled:
            yield f"event: error\ndata: {json.dumps({'error': 'Request cancelled'})}\n\n"
            return

        yield f"event: status\ndata: {json.dumps({'message': 'Processing...'})}\n\n"
        await asyncio.sleep(0.05)

        result = await loop.run_in_executor(
            None,
            run_chat_graph,
            session.llm_client,
            message,
            session.image,
            session.conversation_history,
            session.last_detections,
            session.last_class_vocab,
            session.caption_data
        )

        if session.is_cancelled:
            yield f"event: error\ndata: {json.dumps({'error': 'Request cancelled'})}\n\n"
            return

        answer = result.answer or ""

        session.update_state(result)
        session.add_message("assistant", answer)

        yield f"event: stream_start\ndata: {json.dumps({'message': 'Generating response...'})}\n\n"

        words = answer.split()
        for i, word in enumerate(words):
            if session.is_cancelled:
                yield f"event: error\ndata: {json.dumps({'error': 'Request cancelled'})}\n\n"
                return

            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"event: token\ndata: {json.dumps({'token': chunk})}\n\n"
            await asyncio.sleep(0.02) # NOTE: (@gas) simulates streaming delay

        response_data = ChatMessageResponse(
            session_id=session_id,
            message_type="conversation",
            answer=answer,
            provenance=result.provenance
        )

        yield f"event: result\ndata: {json.dumps(response_data.model_dump())}\n\n"
        yield f"event: done\ndata: {json.dumps({'message': 'Complete'})}\n\n"

    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error("Chat processing error", error=str(e), traceback=error_msg, session_id=session_id)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@app.post("/api/chat/{session_id}/cancel")
async def cancel_request(session_id: str):
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session.cancel()

    return {"message": "Cancellation requested", "session_id": session_id}


@app.post("/api/chat/{session_id}/message")
async def send_message(session_id: str, request: ChatMessageRequest):
    return StreamingResponse(
        stream_chat_response(session_id, request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.delete("/api/chat/{session_id}")
async def delete_session(session_id: str):
    manager = get_session_manager()
    success = manager.delete_session(session_id)

    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session deleted successfully"}


@app.get("/api/chat/sessions")
async def list_sessions():
    manager = get_session_manager()
    return {"sessions": manager.list_sessions()}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Chat API Server",
        "version": "1.0.0",
        "endpoints": {
            "new_chat": "POST /api/chat/new",
            "upload_image": "POST /api/chat/{session_id}/upload",
            "send_message": "POST /api/chat/{session_id}/message (SSE)",
            "session_status": "GET /api/chat/{session_id}/status",
            "delete_session": "DELETE /api/chat/{session_id}",
            "list_sessions": "GET /api/chat/sessions"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8001, 
        log_level="info",
    )
