## Step by Step Installation
This documentation file describes step-by-step installation of Image Chat Agent for Linux / Windows / MacOS.

---

### Install Python
First, install Python on your machine.

**For Example, Fedora:**
```bash
sudo dnf upgrade --refresh
sudo dnf install python3
python3 --version
```

**Or Windows:**
```bash
winget install Python.Python.3
python --version
```

**Manual Windows Installation:**
```bash
$pythonUrl = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe" # Replace with desired version
$pythonInstaller = "$($env:TEMP)\python.exe"
Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller
Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait
Remove-Item $pythonInstaller -Force
python --version
```

---

### Install UV
For package management, now we install UV. It can be used on Mac / Windows / Linux

**Linux / Mac Installation:**
```bash
sudo curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows Installation:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Run Sync Process:**
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For macOS Only
sudo uv sync
```

---

### Install Ollama
Download and Install Ollama from official website. Can be used on Mac / Linux / Windows.

**Linux Installation:**
```bash
sudo curl -fsSL https://ollama.com/install.sh | sh
```

**Windows / MacOS Installation:**
Download and Run installer from [Official Website](https://ollama.com/download)

**And Pull Required Model:**
```bash
sudo ollama pull qwen2.5:3b
```

---

### Services Running
Now, let's loot at services running.

**Our Agent separated as 3 services:**
- MCP Server;
- Chat API Server;
- Gradio Client;

#### 1. Launch MCP Server
Open first terminal and run:
```bash
uv run python -m detection_mcp_server.main
```

The server will start on `http://127.0.0.1:8000`. Verify it's running:
```bash
curl http://127.0.0.1:8000/health
```
#### 2. Start Chat API server
Open second terminal and run:
```bash
uv run python -m chat_api.main
```

#### 3. Run Gradio Client
Open third terminal and run:
```bash
uv run python -m gradio_chat_client.main
```

---

## Testing
Open **http://127.0.0.1:7860** in your browser. Upload an image and chat!
- "What is in this image?"
- "Detect all cars"
- "How many did you find?"