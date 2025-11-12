# Image Chat

Extract information from images using text - an interactive chat interface powered local models.  

## Quick Start

### Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the repository:

```bash
git clone <repo-url>
cd image-chat
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For macOS
uv sync
```

Install [Ollama](https://ollama.com/download) and pull the model:

```bash
ollama pull qwen2.5:3b
```

### Run  

**Terminal 1** - Start MCP server:
```bash
uv run python -m detection_mcp_server.main
```

The server will start on `http://127.0.0.1:8000`. Verify it's running:
```bash
curl http://127.0.0.1:8000/health
```

**Terminal 2** - Start Chat API server:
```bash
uv run python -m chat_api.main
```

**Terminal 3** - Start Gradio client:
```bash
uv run python -m gradio_chat_client.main
```

Open **http://127.0.0.1:7860** in your browser. Upload an image and chat!
- "What is in this image?"
- "Detect all cars"
- "How many did you find?"
