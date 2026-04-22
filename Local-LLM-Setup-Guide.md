# Turn Your Mac Into a Local LLM Server

> **Target machine:** Mac M3 Max, 36 GB unified memory
>
> **Goal:** Run LLMs locally and make them accessible to every device on your home/office network - other Macs, Linux boxes, iPads, or anything that can make an HTTP call.

---

## What You Will Have When Done

```
┌─────────────────────────────────────────────────────────────────┐
│                      Your Local Network                         │
│                                                                 │
│   ┌──────────────────────────┐                                  │
│   │   HOST MAC (M3 Max)      │                                  │
│   │                          │                                  │
│   │  Ollama   → port 11434   │◄──── http://<HOST_IP>:11434      │
│   │  LM Studio → port 1234   │◄──── http://<HOST_IP>:1234       │
│   └──────────────────────────┘                                  │
│         ▲          ▲          ▲                                  │
│         │          │          │                                  │
│   ┌─────┴──┐ ┌────┴───┐ ┌───┴────────┐                         │
│   │ MacBook │ │ iPad   │ │ Linux PC   │  ... any device         │
│   │ (curl,  │ │ (web   │ │ (Python,   │      on the LAN         │
│   │ Cursor) │ │  UI)   │ │ Open WebUI)│                         │
│   └────────┘ └────────┘ └────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick-Start (TL;DR)

If you just want the fastest path, run these on the **host Mac** and skip to [Section 5](#part-2--connecting-from-other-devices) for client setup.

**Ollama (3 commands):**

```bash
brew install ollama
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
launchctl setenv OLLAMA_ORIGINS "*"
# Restart the Ollama app, then:
ollama pull llama3.1:8b
```

**LM Studio (2 clicks):**

1. Download from <https://lmstudio.ai>, open it, download a model.
2. Developer tab → Start Server → enable **"Serve on Local Network"**.

Now jump to [Section 5](#part-2--connecting-from-other-devices) to test from another device.

---

## Table of Contents

### Part 1 - Server Setup (on the Host Mac)

1. [Prerequisites](#1-prerequisites)
2. [Ollama - Install, Configure, and Expose](#2-ollama--install-configure-and-expose)
3. [LM Studio - Install, Configure, and Expose](#3-lm-studio--install-configure-and-expose)
4. [macOS Firewall - Allow Incoming Connections](#4-macos-firewall--allow-incoming-connections)

### Part 2 - Connecting from Other Devices

5. [Test Connectivity from a Client](#5-test-connectivity-from-a-client)
6. [Set Up a Web UI (Open WebUI)](#6-set-up-a-web-ui-open-webui)
7. [Use as an OpenAI-Compatible Backend in Apps](#7-use-as-an-openai-compatible-backend-in-apps)

### Reference

8. [Recommended Models for 36 GB RAM](#8-recommended-models-for-36-gb-ram)
   - [Coding Models](#81-coding-models-python-go-rust-kubernetes-clouddevops)
   - [Thinking and Research Models](#82-thinking-and-research-models)
   - [Vision / Screen Reading Models](#83-vision--screen-reading-models)
   - [Memory Budget - What Fits Together](#84-memory-budget--what-fits-together)
9. [Voice Transcription (Local Whisper)](#9-voice-transcription-local-whisper)
10. [Local Meeting Notes and Screen Reading](#10-local-meeting-notes-and-screen-reading)
11. [Performance Tuning (Ollama)](#11-performance-tuning-ollama)
12. [Troubleshooting](#12-troubleshooting)
13. [Command Cheat Sheet](#13-command-cheat-sheet)

---

# Part 1 - Server Setup (on the Host Mac)

Everything in this part is done on the M3 Max Mac that will run the models.

---

## 1. Prerequisites

- **macOS Sonoma 14+** (or Sequoia 15+). Apple Silicon is required - you have an M3 Max, so you are good.
- **Homebrew** installed. If not:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

- **At least 20 GB of free disk space** for model files (more if you plan to download several models).
- Know your Mac's **local IP address** - you will need it later:

```bash
ipconfig getifaddr en0
```

Write down the output (e.g. `192.168.1.42`). This is your `HOST_IP` for the rest of the guide.

> **Tip:** Assign a static IP or create a DHCP reservation on your router so the address never changes.

---

## 2. Ollama - Install, Configure, and Expose

### Step 1: Install Ollama

Pick one method:

**Option A - Homebrew (recommended):**

```bash
brew install ollama
```

**Option B - Direct download:**

Go to <https://ollama.com/download/mac>, download the `.dmg`, and drag **Ollama.app** to `/Applications`. Open it once to finish setup.

**Verify it works:**

```bash
ollama --version
```

You should see a version number (e.g. `ollama version 0.6.x`). If the command is not found, make sure Ollama.app has been opened at least once.

---

### Step 2: Download your first model

```bash
ollama pull llama3.1:8b
```

This downloads ~5 GB. Once done, test it locally:

```bash
ollama run llama3.1:8b
```

Type a message and confirm you get a response. Press `Ctrl+D` to exit the chat.

> See [Section 8](#8-recommended-models-for-36-gb-ram) for a full list of models that fit in 36 GB.

---

### Step 3: Expose Ollama to the network

By default, Ollama only listens on `127.0.0.1:11434` (localhost). Other devices **cannot** reach it. You need to change it to `0.0.0.0:11434` so it listens on all network interfaces.

> **Why can't I just add it to `.zshrc`?** The Ollama macOS GUI app runs as a separate process and does **not** read your shell config files. You must use `launchctl` or a LaunchAgent plist.

Choose **one** of the two methods below:

#### Method A - launchctl (simple, but resets after reboot)

Run these three commands:

```bash
# 1. Tell Ollama to listen on all interfaces
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"

# 2. Allow requests from any origin (needed for web UIs on other machines)
launchctl setenv OLLAMA_ORIGINS "*"

# 3. Restart the Ollama app so it picks up the new settings
pkill -f Ollama && sleep 2 && open -a Ollama
```

That's it. After a reboot you will need to re-run the two `launchctl setenv` commands and restart Ollama.

#### Method B - LaunchAgent plist (persistent, survives reboots)

This creates a macOS service that starts Ollama automatically on login with the correct environment variables.

**1. Quit the Ollama GUI app first** (so ports don't conflict later):

```bash
pkill -f Ollama
```

**2. Find your Ollama binary path:**

```bash
which ollama
```

Typical results:
- Homebrew install: `/opt/homebrew/bin/ollama`
- Direct download: `/usr/local/bin/ollama`

Note the path - you will need it in the next step.

**3. Create the plist file:**

```bash
mkdir -p ~/Library/LaunchAgents
```

Now create `~/Library/LaunchAgents/com.ollama.server.plist` with the following content. **Replace the binary path on line 10 if yours differs:**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.ollama.server</string>

  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/ollama</string>   <!-- CHANGE if your path differs -->
    <string>serve</string>
  </array>

  <key>EnvironmentVariables</key>
  <dict>
    <key>OLLAMA_HOST</key>
    <string>0.0.0.0:11434</string>

    <key>OLLAMA_ORIGINS</key>
    <string>*</string>

    <key>OLLAMA_FLASH_ATTENTION</key>
    <string>1</string>

    <key>OLLAMA_KV_CACHE_TYPE</key>
    <string>q8_0</string>

    <key>OLLAMA_KEEP_ALIVE</key>
    <string>-1</string>
  </dict>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>/tmp/ollama.log</string>

  <key>StandardErrorPath</key>
  <string>/tmp/ollama.error.log</string>
</dict>
</plist>
```

**4. Load the service:**

```bash
launchctl load -w ~/Library/LaunchAgents/com.ollama.server.plist
```

Ollama is now running in the background and will auto-start on every login.

**To stop or unload later:**

```bash
launchctl unload -w ~/Library/LaunchAgents/com.ollama.server.plist
```

**To check logs if something goes wrong:**

```bash
tail -50 /tmp/ollama.log
tail -50 /tmp/ollama.error.log
```

---

### Step 4: Verify it is listening on the network

```bash
lsof -i :11434
```

**What to look for:** A line containing `TCP *:11434 (LISTEN)`. The `*` means "all interfaces" - other devices can connect.

If you see `TCP localhost:11434` or `TCP 127.0.0.1:11434` instead, the environment variable was not picked up. Restart Ollama and try again.

**Quick self-test from the same Mac:**

```bash
curl http://localhost:11434/api/tags
```

You should get a JSON response listing your downloaded models.

> **Checkpoint:** Ollama is installed, a model is downloaded, and the server is listening on `0.0.0.0:11434`. You can now skip to [Section 4](#4-macos-firewall--allow-incoming-connections) for firewall setup, or continue to set up LM Studio as well.

---

## 3. LM Studio - Install, Configure, and Expose

### Step 1: Install LM Studio

1. Go to <https://lmstudio.ai>.
2. Download the **Apple Silicon** (ARM64) build.
3. Open the `.dmg` and drag **LM Studio.app** to `/Applications`.
4. Launch LM Studio.

**Verify the CLI is available:**

```bash
lms --version
```

If `lms` is not found, open LM Studio once - it registers the CLI path on first launch. Then open a new terminal window and try again.

---

### Step 2: Download a model

**Via the GUI:**

1. Click the **Discover** tab (magnifying glass icon, left sidebar).
2. Search for a model (e.g. `llama 3.1 8b instruct`).
3. Click the download button next to a GGUF variant.
4. Wait for the download to complete - progress shows at the bottom.

**Via the CLI:**

```bash
lms get llama-3.1-8b-instruct
```

---

### Step 3: Start the API server

**Via the GUI (recommended first time):**

1. Click the **Developer** tab (`<>` icon, left sidebar).
2. At the top of the panel, select the model you downloaded from the dropdown.
3. Click **"Start Server"**. You should see a green indicator and the URL `http://localhost:1234`.

---

### Step 4: Enable network access

This is the critical step that makes LM Studio reachable from other devices.

**In the server settings panel (same Developer tab):**

1. Find the **"Serve on Local Network"** toggle and **turn it ON**.
2. The URL displayed will change from `http://localhost:1234` to `http://<YOUR_LAN_IP>:1234`.
3. (Optional) Turn on **"Enable CORS"** - required if browser-based tools on other machines will call the API.
4. (Optional) Turn on **"Require Authentication"** and copy the API key - useful if you want to restrict access.

That single toggle is all you need. No plist files, no launchctl.

---

### Step 5: Verify it is accessible

**From the host Mac:**

```bash
curl http://localhost:1234/v1/models
```

**From the host Mac using the LAN IP (simulates a remote device):**

```bash
curl http://$(ipconfig getifaddr en0):1234/v1/models
```

Both should return a JSON object listing the loaded model.

> **Checkpoint:** LM Studio is installed, a model is loaded, the server is running, and "Serve on Local Network" is enabled. The API is available at `http://<HOST_IP>:1234`.

---

## 4. macOS Firewall - Allow Incoming Connections

If you have the macOS firewall turned on (System Settings > Network > Firewall), it may silently block connections from other devices. You need to allow Ollama and LM Studio through.

### Option A - Via System Settings (GUI)

1. Open **System Settings**.
2. Go to **Network > Firewall**.
3. Click **Options...** (authenticate with Touch ID or password if prompted).
4. Click the **"+"** button at the bottom of the app list.
5. Navigate to `/Applications`, select **Ollama.app**, and click **Add**.
6. Make sure it says **"Allow incoming connections"** next to Ollama.
7. Repeat steps 4-6 for **LM Studio.app**.
8. Click **OK**.

### Option B - Via Terminal (Packet Filter rules)

```bash
sudo bash -c 'echo "pass in proto tcp from any to any port 11434" >> /etc/pf.conf'
sudo bash -c 'echo "pass in proto tcp from any to any port 1234" >> /etc/pf.conf'
sudo pfctl -f /etc/pf.conf
sudo pfctl -e
```

Verify the rules were loaded:

```bash
sudo pfctl -sr | grep -E "1143|1234"
```

> **Checkpoint:** The firewall now allows traffic on ports 11434 and 1234. Your server setup is complete. Move to Part 2.

---

# Part 2 - Connecting from Other Devices

Everything below is done on a **client device** (another Mac, a Linux PC, an iPad, etc.) that is on the **same local network** as the host Mac.

Replace `192.168.1.42` in all examples with your actual host Mac IP from [Section 1](#1-prerequisites).

---

## 5. Test Connectivity from a Client

Open a terminal on the client device and run these tests.

### Test Ollama

**1. Ping the host:**

```bash
ping -c 3 192.168.1.42
```

If this fails, the devices are not on the same network or a firewall is blocking.

**2. List available models:**

```bash
curl http://192.168.1.42:11434/api/tags
```

Expected: a JSON array of model objects.

**3. Send a chat request (OpenAI-compatible format):**

```bash
curl http://192.168.1.42:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [
      {"role": "user", "content": "Explain unified memory in one sentence."}
    ]
  }'
```

Expected: a JSON response with `choices[0].message.content` containing the answer.

### Test LM Studio

**1. List models:**

```bash
curl http://192.168.1.42:1234/v1/models
```

**2. Send a chat request:**

```bash
curl http://192.168.1.42:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-id-from-list-above>",
    "messages": [
      {"role": "user", "content": "Hello from across the network!"}
    ]
  }'
```

> Replace `<model-id-from-list-above>` with the actual model ID returned by the `/v1/models` call.

If both tests return valid JSON responses, your LLM server is fully operational.

---

## 6. Set Up a Web UI (Open WebUI)

[Open WebUI](https://github.com/open-webui/open-webui) gives you a ChatGPT-style browser interface that connects to your local LLM server. Run it on **any machine** with Docker.

### Pointing at Ollama

```bash
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://192.168.1.42:11434 \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

Open `http://localhost:3000` in your browser. Create an admin account on first launch. It will auto-discover all models from Ollama.

### Pointing at LM Studio

Use the OpenAI-compatible provider and set the base URL to:

```
http://192.168.1.42:1234/v1
```

You can configure this in Open WebUI's **Settings > Connections > OpenAI API**.

---

## 7. Use as an OpenAI-Compatible Backend in Apps

Both Ollama and LM Studio expose an **OpenAI-compatible API**, so any app that lets you set a custom API base URL can use your local LLM server as a drop-in replacement for OpenAI.

### Cursor IDE

1. Open Cursor Settings.
2. Go to **Models**.
3. Set **OpenAI API Base** to:
   - Ollama: `http://192.168.1.42:11434/v1`
   - LM Studio: `http://192.168.1.42:1234/v1`
4. Set **API Key** to `ollama` (Ollama ignores the key, but the field cannot be empty).

### Continue (VS Code extension)

Edit `~/.continue/config.json` on the client:

```json
{
  "models": [
    {
      "title": "Local Llama (Ollama)",
      "provider": "openai",
      "model": "llama3.1:8b",
      "apiBase": "http://192.168.1.42:11434/v1",
      "apiKey": "ollama"
    }
  ]
}
```

### Python (openai library)

Install the library on the client: `pip install openai`

```python
from openai import OpenAI

# --- Using Ollama ---
client = OpenAI(
    base_url="http://192.168.1.42:11434/v1",
    api_key="ollama",  # Ollama ignores this, but the library requires it
)

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "What is Metal GPU acceleration?"}],
)

print(response.choices[0].message.content)


# --- Using LM Studio ---
client_lms = OpenAI(
    base_url="http://192.168.1.42:1234/v1",
    api_key="lm-studio",  # or the real key if you enabled authentication
)

response = client_lms.chat.completions.create(
    model="<model-id>",
    messages=[{"role": "user", "content": "Hello from Python!"}],
)

print(response.choices[0].message.content)
```

### Any HTTP Client

POST to either endpoint with the standard OpenAI JSON body:

```
POST http://<HOST_IP>:11434/v1/chat/completions   (Ollama)
POST http://<HOST_IP>:1234/v1/chat/completions     (LM Studio)

Headers:
  Content-Type: application/json

Body:
{
  "model": "<model-name>",
  "messages": [{"role": "user", "content": "Your prompt here"}]
}
```

---

# Reference

## 8. Recommended Models for 36 GB RAM

Keep total loaded model size under ~30 GB to leave headroom for macOS and the inference engine. Models below are grouped by use case so you can pick exactly what you need.

> **LM Studio users:** Search for the same model names in the Discover tab. Pick GGUF quantisation variants - Q4_K_M is a good default balance of quality and size.

---

### 8.1 Coding Models (Python, Go, Rust, Kubernetes, Cloud/DevOps)

These models are specifically trained on code and excel at generation, refactoring, debugging, writing Terraform/Helm/Dockerfile configs, and explaining infrastructure.

| Model | Size | Strengths | Ollama Command |
|---|---|---|---|
| `qwen2.5-coder:7b` | ~5 GB | Fast code completion, great for quick edits. Supports 90+ languages. | `ollama pull qwen2.5-coder:7b` |
| `qwen2.5-coder:14b` | ~9 GB | **Best balance for daily coding.** Strong on Python, Go, Rust. Handles K8s manifests, Terraform, and cloud configs well. | `ollama pull qwen2.5-coder:14b` |
| `qwen2.5-coder:32b` | ~20 GB | Top-tier local coding model. Rivals GPT-4 on code benchmarks. Excellent multi-file context understanding. | `ollama pull qwen2.5-coder:32b` |
| `deepseek-coder-v2:16b` | ~9 GB | Strong code generation across 80+ languages. Good at explaining complex code. | `ollama pull deepseek-coder-v2:16b` |
| `codellama:34b` | ~20 GB | Meta's code specialist. Excellent for Python, infill completions, and long code context. | `ollama pull codellama:34b` |

**Recommendation for your 36 GB:** Start with `qwen2.5-coder:14b` (~9 GB) for everyday coding. When tackling complex multi-file refactors or architecture decisions, load `qwen2.5-coder:32b` (~20 GB) instead.

**For DevOps/Cloud tasks specifically:** The coder models handle Kubernetes YAML, Terraform HCL, Dockerfiles, Ansible playbooks, and AWS/GCP/Azure CLI commands well. For broader infrastructure reasoning (architecture decisions, cost analysis), pair with a thinking model from the next section.

---

### 8.2 Thinking and Research Models

These models perform step-by-step reasoning before answering. They show their "chain of thought" and excel at complex problem-solving, research analysis, and multi-step logic.

| Model | Size | Strengths | Ollama Command |
|---|---|---|---|
| `deepseek-r1:14b` | ~9 GB | Fast reasoning model distilled from DeepSeek-R1. Great balance of speed and depth. 128K context. | `ollama pull deepseek-r1:14b` |
| `deepseek-r1:32b` | ~20 GB | **Best local reasoning model.** Approaches O1/O3-level performance on math, logic, and research tasks. 128K context. | `ollama pull deepseek-r1:32b` |
| `qwen3:8b` | ~5 GB | Hybrid thinking - toggle `/think` for deep reasoning or `/nothink` for quick answers, in one model. | `ollama pull qwen3:8b` |
| `qwen3:14b` | ~9 GB | Stronger reasoning with the same think/nothink flexibility. Great for research and analysis. | `ollama pull qwen3:14b` |
| `qwen3:32b` | ~20 GB | Top-tier thinking model. Competitive with DeepSeek-R1 on complex tasks. | `ollama pull qwen3:32b` |
| `qwen3:30b-a3b` | ~19 GB | Mixture-of-Experts: 30B total params but only 3B active per token. Surprisingly capable for its speed. | `ollama pull qwen3:30b-a3b` |

**Recommendation for your 36 GB:** Use `deepseek-r1:14b` or `qwen3:14b` (~9 GB each) as your daily thinking model. For harder research problems, swap to `deepseek-r1:32b` or `qwen3:32b` (~20 GB).

**Qwen3 thinking mode:** Add `/think` to your prompt for step-by-step reasoning, or `/nothink` for a quick direct answer. This is controlled per-message so you don't need to switch models.

---

### 8.3 Vision / Screen Reading Models

These models can process images and screenshots - useful for OCR, reading UI elements, understanding diagrams, and describing what's on screen.

| Model | Size | Strengths | Ollama Command |
|---|---|---|---|
| `llava:7b` | ~5 GB | Fast image understanding. Good for quick screenshot analysis and OCR. | `ollama pull llava:7b` |
| `llava:13b` | ~8 GB | More accurate image reasoning. Better at complex diagrams and charts. | `ollama pull llava:13b` |
| `llama3.2-vision:11b` | ~8 GB | Meta's vision model. Strong at document understanding and visual QA. | `ollama pull llama3.2-vision:11b` |
| `qwen3-vl:8b` | ~5 GB | **Newest and most capable.** OCR in 32 languages, can understand GUIs, handles long documents. | `ollama pull qwen3-vl:8b` |
| `granite3.2-vision:2b` | ~2 GB | Lightweight, optimised for document understanding, tables, and charts. | `ollama pull granite3.2-vision:2b` |
| `moondream:1.8b` | ~1 GB | Ultra-light. Fast enough for real-time screen monitoring on constrained memory. | `ollama pull moondream:1.8b` |

**Sending an image to a vision model:**

```bash
# From the CLI
ollama run llava:13b "What does this screenshot show?" --images ./screenshot.png

# Via the API
curl http://localhost:11434/api/generate -d '{
  "model": "llava:13b",
  "prompt": "Describe everything on this screen.",
  "images": ["<BASE64_ENCODED_IMAGE>"]
}'
```

---

### 8.4 Memory Budget - What Fits Together

You have ~30 GB usable. Here are practical combos you can keep loaded simultaneously:

| Combo | Total Memory | Use Case |
|---|---|---|
| `qwen2.5-coder:14b` + `deepseek-r1:14b` + `qwen3-vl:8b` | ~23 GB | Coding + Thinking + Vision (best daily driver combo) |
| `qwen2.5-coder:32b` + `qwen3:8b` | ~25 GB | Max coding power + lightweight thinking |
| `deepseek-r1:32b` + `llava:7b` | ~25 GB | Deep research + quick image analysis |
| `qwen2.5-coder:14b` + `qwen3:14b` + `moondream:1.8b` | ~19 GB | Balanced trio, lots of headroom |
| `qwen2.5-coder:32b` alone | ~20 GB | Single best coding model, maximum context room |
| `deepseek-r1:32b` alone | ~20 GB | Single best thinking model, maximum context room |

**To download an entire combo at once:**

```bash
ollama pull qwen2.5-coder:14b && \
ollama pull deepseek-r1:14b && \
ollama pull qwen3-vl:8b
```

---

## 9. Voice Transcription (Local Whisper)

For offline voice-to-text - meeting transcription, voice notes, lecture capture - use **OpenAI Whisper** running locally on Apple Silicon. No audio ever leaves your Mac.

### 9.1 Whisper Model Comparison

| Model | Size | Speed (M3 Max) | Accuracy | Best For |
|---|---|---|---|---|
| `tiny` | 75 MB | ~32x real-time | Fair | Quick drafts, keyword capture |
| `small` | 461 MB | ~15x real-time | Good | Casual notes, short memos |
| `medium` | 1.5 GB | ~8x real-time | Very good | Meetings, interviews |
| `large-v3` | 3 GB | ~3x real-time | Excellent | Maximum accuracy |
| `large-v3-turbo` | 1.6 GB | ~8x real-time | Excellent | **Best pick - near-large accuracy at 5x the speed** |

`large-v3-turbo` is the sweet spot: 0.2% character error rate, 100+ languages, and only 1.6 GB on disk.

### 9.2 Option A - whisper.cpp (CLI, lightweight)

The fastest way to run Whisper natively on Apple Silicon with Metal acceleration.

**Install:**

```bash
brew install whisper-cpp
```

**Download the large-v3-turbo model:**

```bash
# whisper.cpp stores models in its own directory
whisper-cpp-download-ggml-model large-v3-turbo
```

**Transcribe an audio file:**

```bash
whisper-cpp \
  --model large-v3-turbo \
  --language auto \
  --output-txt \
  --output-srt \
  meeting-recording.wav
```

This outputs both a plain text transcript and an `.srt` subtitle file with timestamps.

**Record and transcribe in real-time (using sox):**

```bash
brew install sox
rec -r 16000 -c 1 -b 16 recording.wav
# Press Ctrl+C when done, then:
whisper-cpp --model large-v3-turbo recording.wav
```

### 9.3 Option B - LocalWhisper (GUI app)

A native macOS app with a clean interface - no terminal needed.

1. Download from <https://localwhisper.ai>.
2. Open the app and select a Whisper model to download (pick `large-v3-turbo`).
3. Drag in an audio/video file, or use the record button for live transcription.
4. Exports to TXT, SRT, VTT, or JSON.

Uses CoreML + Metal acceleration. Runs 100% offline.

### 9.4 Option C - Whisper Transcription Mac (open source GUI)

A free, open-source native Mac app: <https://github.com/Whisper-Transcription-OSX/Whisper-Transcription-Mac>

- Supports all Whisper model sizes
- Batch processing of multiple files
- Timestamped output
- Metal and Neural Engine acceleration on Apple Silicon

---

## 10. Local Meeting Notes and Screen Reading

For capturing audio, transcribing it, reading screen context, and generating structured notes - entirely locally - here are two open-source options that integrate with your Ollama server.

### 10.1 Meetily (full-featured, GUI)

A fully local meeting recorder. Records meetings, transcribes locally with Whisper, and summarises with a local LLM via Ollama.

**Install on Mac:**

```bash
brew tap zackriya-solutions/meetily
brew install --cask meetily
```

**Start the backend server:**

```bash
meetily-server --language en --model medium
```

This launches Whisper transcription on port 8178 and the FastAPI backend on port 5167.

**Then open Meetily from Applications.**

**How it works:**
- Captures system audio and microphone during meetings (Zoom, Teams, Meet, etc.)
- Transcribes in real-time using Whisper (runs locally)
- Sends transcript to your local Ollama server for summarisation
- Generates structured meeting notes with action items

**Connect to your Ollama server:**
Meetily auto-detects Ollama if it's running on the same machine. For summarisation, it works well with `llama3.1:8b`, `mistral:7b`, or `qwen3:8b`.

### 10.2 ownscribe (lightweight, CLI)

A minimal CLI tool that records, transcribes, and summarises in one command.

**GitHub:** <https://github.com/paberr/ownscribe>

**Install:**

```bash
pip install ownscribe
```

**Run:**

```bash
ownscribe
# Records audio → transcribes with WhisperX → summarises with local LLM
# Press Ctrl+C to stop recording
```

**Works with:** Ollama or LM Studio as the summarisation backend. Runs entirely offline on macOS 14.2+.

### 10.3 DIY Screen Reading with Vision Models

For reading what's on your screen (not just audio), combine a screenshot tool with a vision model:

**Capture and analyse your screen:**

```bash
# Take a screenshot
screencapture -x /tmp/screen.png

# Ask a vision model to describe/read it
ollama run qwen3-vl:8b "Read all the text on this screen and summarise it." --images /tmp/screen.png
```

**Automate it with a shell function** (add to `~/.zshrc`):

```bash
read-screen() {
    screencapture -x /tmp/_screen_capture.png
    ollama run qwen3-vl:8b "${1:-Summarise everything visible on this screen.}" \
        --images /tmp/_screen_capture.png
}
```

Then just run:

```bash
read-screen
read-screen "What code is visible? Are there any bugs?"
read-screen "Extract all the action items from this meeting notes window."
```

---

### Quick Download - All Recommended Models

Copy and run this block to download every model mentioned above. Total disk usage is ~65 GB, but you only load what you need into memory at any given time.

```bash
# --- Coding ---
ollama pull qwen2.5-coder:14b
ollama pull qwen2.5-coder:32b
ollama pull deepseek-coder-v2:16b

# --- Thinking / Research ---
ollama pull deepseek-r1:14b
ollama pull deepseek-r1:32b
ollama pull qwen3:14b

# --- Vision / Screen Reading ---
ollama pull qwen3-vl:8b
ollama pull llava:13b

# --- General Purpose ---
ollama pull llama3.1:8b
ollama pull mistral:7b
```

---

## 11. Performance Tuning (Ollama)

These environment variables are set the same way as `OLLAMA_HOST` - via `launchctl setenv` or in the plist file. They are already included in the plist from [Method B](#method-b--launchagent-plist-persistent-survives-reboots).

| Variable | Recommended Value | What It Does |
|---|---|---|
| `OLLAMA_FLASH_ATTENTION` | `1` | Dramatically reduces memory usage at large context sizes |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Quantises the KV cache to 8-bit - halves cache memory with negligible quality loss |
| `OLLAMA_KEEP_ALIVE` | `-1` | Keeps models loaded in memory forever (avoids 5-minute unload timeout) |
| `OLLAMA_NUM_PARALLEL` | `2` | Serves 2 requests in parallel per model (doubles context memory) |
| `OLLAMA_CONTEXT_LENGTH` | `8192` | Sets the default context window (tokens) |

**Check GPU offload status:**

```bash
ollama ps
```

The `PROCESSOR` column shows:
- `100% GPU` - fully on the GPU (best performance)
- `100% CPU` - fully on CPU (slow - model is too large)
- `60%/40% CPU/GPU` - partial offload (model barely fits)

**Pre-warm a model** (load it into memory without sending a prompt):

```bash
curl http://localhost:11434/api/generate -d '{"model": "llama3.1:8b"}'
```

---

## 12. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `Connection refused` from client device | Server is bound to `localhost` only | Set `OLLAMA_HOST=0.0.0.0:11434` (Ollama) or enable "Serve on Local Network" (LM Studio) |
| `Connection refused` but server binding looks correct | macOS firewall is blocking the port | Allow the app through the firewall ([Section 4](#4-macos-firewall--allow-incoming-connections)) |
| `curl` works from the host Mac but not from another device | Firewall or network isolation | Check firewall; confirm both devices are on the same subnet (`ping <HOST_IP>` from client) |
| Model loads on CPU instead of GPU | Model too large for remaining memory | Close memory-heavy apps, or use a smaller model/quantisation; check with `ollama ps` |
| Very slow first response, fast after | Model is loading into memory on first request | Pre-warm the model (see [Section 11](#11-performance-tuning-ollama)) |
| Model unloads after ~5 minutes of idle | Default `keep_alive` timeout | Set `OLLAMA_KEEP_ALIVE=-1` to keep it loaded indefinitely |
| CORS errors in a browser-based UI | Origins not whitelisted | Set `OLLAMA_ORIGINS=*` or enable CORS in LM Studio |
| `launchctl setenv` values disappear after reboot | macOS clears `launchctl` env vars on restart | Use the LaunchAgent plist method ([Method B](#method-b--launchagent-plist-persistent-survives-reboots)) |
| `Error: address already in use` | Another process is using the port | `lsof -i :11434` to find it, then `kill <PID>` |
| `lms: command not found` | LM Studio CLI not registered | Open LM Studio GUI at least once, then open a new terminal |
| Ollama plist not starting | Wrong binary path in the plist | Run `which ollama` and update the `<string>` in `ProgramArguments`; check `/tmp/ollama.error.log` |

---

## 13. Command Cheat Sheet

### Ollama

```bash
ollama list                           # List all downloaded models
ollama ps                             # Show loaded models + GPU/CPU split
ollama pull <model>                   # Download a model
ollama rm <model>                     # Delete a model
ollama run <model>                    # Interactive chat session
ollama stop <model>                   # Unload a model from memory
ollama show <model>                   # Show model details (size, params, etc.)
```

### LM Studio CLI

```bash
lms list                              # List downloaded models
lms get <model>                       # Download a model
lms server start                      # Start the API server
lms server stop                       # Stop the API server
lms server status                     # Check if the server is running
```

### Network Diagnostics

```bash
ipconfig getifaddr en0                # Your Mac's LAN IP address
lsof -i :11434                        # Check if Ollama is listening
lsof -i :1234                         # Check if LM Studio is listening
ping <HOST_IP>                        # Basic connectivity test from client
curl http://<HOST_IP>:11434/api/tags  # Test Ollama API from client
curl http://<HOST_IP>:1234/v1/models  # Test LM Studio API from client
```

### Ollama LaunchAgent Management

```bash
# Load (start on login)
launchctl load -w ~/Library/LaunchAgents/com.ollama.server.plist

# Unload (stop and disable)
launchctl unload -w ~/Library/LaunchAgents/com.ollama.server.plist

# Check logs
tail -50 /tmp/ollama.log
tail -50 /tmp/ollama.error.log
```

---

*Last updated: March 2026*
