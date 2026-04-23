# Build Muneem - Offline AI Notepad for Mac

> **Goal:** Build a fully offline AI meeting notepad that captures your screen and audio in real time, transcribes with speaker attribution, and generates structured notes - all running locally on an M3 Max (36 GB), with zero internet required after initial setup.
>
> **Quick install:** `python3 muneem-setup.py` - one command sets up everything. See [Section 1](#1-prerequisites).

> **Design decisions:**
> - **Accuracy over speed.** Transcription uses WhisperX large-v3 with forced alignment in ~5-second segments. Near real-time feedback with reliable output.
> - **Speaker attribution is built in.** Offline diarization (via the `diarize` ONNX package) labels each utterance as You / Speaker 1 / Speaker 2 etc. No accounts, tokens, or internet needed.
> - **Screen capture is opt-in.** Use `--screen` to enable vision-based speaker identification (maps "Speaker 1" to real names from the call UI). Helpful but not required.
> - **macOS 15+ (Sequoia / Tahoe) required.** Native Core Audio Taps provide reliable system-audio capture without hacks. BlackHole is kept as a fallback.
> - **Apple Silicon only.** M-series chips are required for acceptable local inference performance.

---

## What Muneem Does

A high-level map of the pipeline before we build it:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MUNEEM ARCHITECTURE                          │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐  │
│  │ System Audio │───▶│ Transcription│───▶│  Note Enhancement      │  │
│  │ (Core Audio  │    │  (WhisperX   │    │  (local LLM, Ollama)   │  │
│  │  Tap)        │    │   large-v3)  │    │                        │  │
│  ├─────────────┤    └──────┬───────┘    │  Your raw notes        │  │
│  │ Microphone   │───▶      │            │  + Full transcript     │  │
│  │ (your voice) │    ┌─────▼──────┐     │  + Screen context      │  │
│  └─────────────┘    │ Diarization │     │  = Structured notes    │  │
│                     │ (sherpa-onnx│────▶│    with templates      │  │
│  ┌─────────────┐    │  + CAM++)   │     └────────────────────────┘  │
│  │ Screen       │───▶└────────────┘                                 │
│  │ (vision VLM) │                                                   │
│  └─────────────┘                                                    │
│                                                                     │
│  Key traits:                                                        │
│  - No meeting bot - captures audio directly from the device         │
│  - Audio NEVER leaves your Mac. No cloud calls of any kind.         │
│  - Notes combine: your jots + transcript + screen context           │
│  - Templates customise output format (standup, 1:1, discovery, etc.)│
│  - Post-meeting chat: ask questions about the transcript            │
└──────────────────────────────────────────────────────────────────────┘
```

**What this guide builds:** a fully local pipeline. Audio never leaves your Mac. Transcription runs via WhisperX with forced alignment. Speaker attribution runs via sherpa-onnx (pyannote-segmentation + 3D-Speaker CAM++ embeddings) with a session-scoped SpeakerRegistry for stable labels. Note enhancement runs via Ollama. Screen reading adds context from the call UI (participant names, shared content, etc.).

---

## Our Local Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                        MUNEEM - YOUR MAC                              │
│                  Accuracy-first / Speaker Attribution                  │
│                                                                       │
│  CAPTURE LAYER                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐               │
│  │ System Audio  │  │  Microphone  │  │  Screen       │               │
│  │ ┌──────────┐ │  │  (CoreAudio) │  │  (screencapture│              │
│  │ │ Preferred│ │  │              │  │   every 1s,   │               │
│  │ │ Core     │ │  │              │  │               │               │
│  │ │ Audio Tap│ │  │              │  │               │               │
│  │ ├──────────┤ │  │              │  │               │               │
│  │ │ Fallback │ │  │              │  │               │               │
│  │ │ BlackHole│ │  │              │  │               │               │
│  │ └──────────┘ │  │              │  │               │               │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘               │
│         │                 │                   │                        │
│  PROCESSING LAYER         │                   │                        │
│         ▼                 ▼                   ▼                        │
│  ┌──────────────────────────────┐ ┌──────────────────────────┐        │
│  │  WhisperX large-v3           │ │  Vision Model (Ollama)   │        │
│  │  + forced alignment          │ │  qwen3-vl:8b             │        │
│  │  (~5s segments, accuracy)    │ │  (OCR + screen reading)  │        │
│  ├──────────────────────────────┤ └────────────┬─────────────┘        │
│  │  diarize (offline ONNX)       │              │                      │
│  │  Speaker 1, Speaker 2, etc.  │              │                      │
│  └──────────┬───────────────────┘              │                      │
│             │                                  │                      │
│             ▼                                  ▼                      │
│  ┌──────────────────────────────────────────────────────┐             │
│  │              Local LLM (Ollama)                       │             │
│  │              qwen3:14b                                │             │
│  │                                                       │             │
│  │  Inputs:                                              │             │
│  │  - Speaker-attributed transcript (from WhisperX)      │             │
│  │  - Screen context (from vision model)                 │             │
│  │  - Your manual notes                                  │             │
│  │  - Template (meeting type)                            │             │
│  │                                                       │             │
│  │  Output: Structured meeting notes with attribution    │             │
│  └───────────────────────┬──────────────────────────────┘             │
│                          │                                            │
│  STORAGE LAYER           ▼                                            │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  Markdown files in ~/.muneem/notes/                   │             │
│  │  - Raw transcript with timestamps + speaker labels    │             │
│  │  - Screen context summaries                           │             │
│  │  - Enhanced notes with speaker attribution            │             │
│  └──────────────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Prerequisites](#1-prerequisites) - macOS 15+, Apple Silicon, one-command install, audio backends, diarization setup
2. [Component 1 - Audio Capture (System + Mic)](#2-component-1--audio-capture-system--mic) - Core Audio Tap (preferred) + BlackHole (fallback)
3. [Component 2 - Transcription + Diarization (WhisperX)](#3-component-2--real-time-transcription-whisper) - WhisperX large-v3, forced alignment, offline speaker labels
4. [Component 3 - Screen Capture and Reading](#4-component-3--screen-capture-and-reading) - Vision model OCR via Ollama
5. [Component 4 - LLM Note Enhancement (Ollama)](#5-component-4--llm-note-enhancement-ollama) - Speaker-attributed structured notes
6. [Component 5 - Putting It All Together (The Pipeline)](#6-component-5--putting-it-all-together-the-pipeline) - MeetingSession, CLI, doctor
7. [Ready-Made Alternatives](#7-ready-made-alternatives) - Meetily, ownscribe, etc.
8. [Templates for Different Meeting Types](#8-templates-for-different-meeting-types)
9. [Reference - Useful Commands](#9-reference--useful-commands)

---

## 1. Prerequisites

> **Requires:** macOS 15+ (Sequoia / Tahoe), Apple Silicon (M1/M2/M3/M4), Xcode Command Line Tools.

### 1.1 One-Command Install (Recommended)

The `muneem-setup.py` script handles **everything** - Homebrew packages, native Swift audio helper, Python venv with WhisperX + diarize, Ollama models, and the `muneem` CLI:

```bash
python3 muneem-setup.py
```

After it finishes, `muneem` is available immediately (symlinked to `/usr/local/bin`):

```bash
muneem doctor    # Verify all dependencies
muneem start     # Start a meeting session
```

**That is it.** The rest of Section 1 is for reference if you want to understand what was installed.

---

### 1.2 What the Setup Script Installs

| Category | What | Why |
|---|---|---|
| Brew formulae | `ffmpeg`, `sox`, `portaudio`, `ollama` | Audio tools + LLM runtime |
| Brew cask | `blackhole-2ch` | Fallback audio driver (used when Core Audio Tap is unavailable) |
| Native binary | `~/.muneem/native/muneem-audio` | Compiled Swift helper for Core Audio Tap + mic capture |
| Python venv | `~/.muneem/venv` | Isolated environment |
| pip packages | `whisperx`, `diarize`, `torch`, `pyaudio`, `numpy`, `requests`, `Pillow`, `rich` | Transcription, offline diarization, audio, screen capture, UI |
| Ollama models | `qwen3:14b`, `qwen3-vl:8b` | Enhancement and screen reading |
| App modules | `~/.muneem/{transcriber,screen_reader,enhancer,app}.py` | The four core Muneem components |
| CLI wrapper | `~/.muneem/bin/muneem` -> `/usr/local/bin/muneem` | The `muneem` command |

### 1.3 Audio Capture: How It Works

Muneem captures system audio (what others say on Zoom/Teams/Meet) using two backends in priority order:

1. **Core Audio Taps (preferred)** - Native macOS API (macOS 14.2+). The setup script compiles a small Swift helper that creates an audio tap on all system processes. No virtual drivers needed. Requires **Screen Recording** permission for your terminal app.

2. **BlackHole (fallback)** - Virtual audio driver installed via Homebrew. Used automatically when Core Audio Tap compilation fails or the tap returns an error at runtime. Requires a one-time Multi-Output Device configuration:
   1. Open **Audio MIDI Setup** (Spotlight > "Audio MIDI Setup").
   2. Click **"+"**, select **Create Multi-Output Device**.
   3. Check both **BlackHole 2ch** and your speakers/headphones.
   4. Right-click the Multi-Output > **Use This Device For Sound Output**.

`muneem start` logs which backend is active so you always know what is happening.

### 1.4 Speaker Diarization (Fully Offline)

Speaker attribution uses the [`diarize`](https://github.com/FoxNoseTech/diarize) package - a fully offline, ONNX-based diarization engine. **No accounts, no tokens, no internet needed.** It is installed automatically by `muneem-setup.py`.

How it works:
- Mic audio is always labeled as **"You"** (separate track, no diarization needed).
- System audio (others on the call) is diarized into **Speaker 1**, **Speaker 2**, etc.
- The `diarize` package uses Silero VAD + WeSpeaker ResNet34 (ONNX) + spectral clustering - all models are bundled in the pip package.
- ~10.8% DER on VoxConverse, CPU-only, ~8x faster than real-time.

### 1.3 Manual Install (If You Prefer)

<details>
<summary>Click to expand manual installation steps</summary>

**System tools:**

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg sox portaudio python@3.12 ollama
brew install --cask blackhole-2ch
```

**Ollama models:**

```bash
ollama pull qwen3:14b        # Note enhancement
ollama pull qwen3-vl:8b      # Screen reading
```

**Python environment:**

```bash
python3.12 -m venv ~/.muneem/venv
source ~/.muneem/venv/bin/activate
pip install whisperx diarize torch torchaudio pyaudio numpy requests Pillow rich
```

</details>

---

## 2. Component 1 - Audio Capture (System + Mic)

This component captures two audio streams simultaneously:
- **System audio** (what others are saying on Zoom/Teams/Meet) - via Core Audio Tap or BlackHole
- **Microphone** (your voice) - via the native mic capture helper or default input device

### 2.1 Audio Backend Selection

Muneem automatically chooses the best available backend at startup:

| Priority | Backend | How it works | When used |
|---|---|---|---|
| 1 (preferred) | **Core Audio Tap** | Native macOS API creates a tap on all system audio processes. Compiled Swift binary at `~/.muneem/native/muneem-audio`. | macOS 14.2+ with Screen Recording permission |
| 2 (fallback) | **BlackHole** | Virtual audio driver. System audio routed through a Multi-Output Device. Captured via PyAudio. | When native helper is missing or tap creation fails |
| 3 (last resort) | **Mic only** | Only captures microphone input. | When neither Core Audio Tap nor BlackHole is available |

### 2.2 Test the Native Helper

```bash
# Record 5 seconds of system audio via Core Audio Tap
~/.muneem/native/muneem-audio system /tmp/test_system.wav 5

# Record 5 seconds of microphone audio
~/.muneem/native/muneem-audio mic /tmp/test_mic.wav 5

# Play back
afplay /tmp/test_system.wav
afplay /tmp/test_mic.wav
```

If the native helper prints `[muneem-audio] Using Core Audio Tap for system audio`, the preferred backend is working.

If it prints `[muneem-audio] Core Audio Tap failed ... falling back to BlackHole`, verify that your terminal has **Screen Recording** permission (System Settings > Privacy & Security > Screen Recording).

### 2.3 Test BlackHole Fallback

```bash
# List audio devices - look for "BlackHole 2ch"
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'  [{i}] {info[\"name\"]} (channels: {info[\"maxInputChannels\"]})')
p.terminate()
"
```

If BlackHole appears in the list and you have configured the Multi-Output Device (see [Section 1.3](#13-audio-capture-how-it-works)), the fallback path is ready.

---

## 3. Component 2 - Transcription + Diarization (WhisperX)

We use **WhisperX large-v3** for transcription because it provides the highest accuracy with forced alignment and word-level timestamps. Combined with the **diarize** package (fully offline ONNX engine), every utterance is attributed to a speaker. This is intentionally a bit laggy (~15-second segments) because accuracy is prioritized over speed.

### 3.1 Test Offline Transcription

```bash
# Activate Muneem venv and transcribe a test recording
source ~/.muneem/venv/bin/activate

python3 -c "
import whisperx

audio = whisperx.load_audio('/tmp/test_system.wav')
model = whisperx.load_model('large-v3', 'cpu', compute_type='float32', language='en')
result = model.transcribe(audio, batch_size=4)

align_model, align_meta = whisperx.load_align_model(language_code='en', device='cpu')
result = whisperx.align(result['segments'], align_model, align_meta, audio, 'cpu')

for seg in result['segments']:
    print(f'[{seg[\"start\"]:.1f}s - {seg[\"end\"]:.1f}s] {seg[\"text\"]}')
"
```

The first run downloads the WhisperX model (~3 GB). Subsequent runs load from cache.

### 3.2 Test Speaker Diarization

```bash
source ~/.muneem/venv/bin/activate

python3 -c "
import whisperx

audio = whisperx.load_audio('/tmp/test_system.wav')
model = whisperx.load_model('large-v3', 'cpu', compute_type='float32', language='en')
result = model.transcribe(audio, batch_size=4)

align_model, align_meta = whisperx.load_align_model(language_code='en', device='cpu')
result = whisperx.align(result['segments'], align_model, align_meta, audio, 'cpu')

import os
from diarize import diarize as run_diar

diar_result = run_diar('/tmp/test_system.wav')
for seg in diar_result.segments:
    print(f'[{seg.speaker}] {seg.start:.1f}s - {seg.end:.1f}s')
"
```

If you see `[SPEAKER_00]`, `[SPEAKER_01]` etc., diarization is working.

### 3.3 Accuracy-First Streaming Pipeline

The `~/.muneem/transcriber.py` module (written by `muneem-setup.py`) implements this pipeline:

1. **Capture**: Records ~5-second segments using the native helper or BlackHole fallback
2. **Transcribe**: WhisperX large-v3 with forced alignment for word-level timestamps
3. **Diarize**: the `diarize` package assigns each segment a speaker label (fully offline, no tokens)
4. **Callback**: Each segment is delivered with `{text, speaker, start, end}`

The segment duration balances real-time responsiveness with transcription accuracy. Shorter segments provide faster feedback while still giving WhisperX enough context for reliable results.

```python
# The transcriber module is automatically written by muneem-setup.py.
# Key configuration in ~/.muneem/transcriber.py:

SEGMENT_DURATION = 5        # Seconds per segment (near real-time)
SILENCE_THRESHOLD = 150     # RMS below this = silence (skip segment)
WHISPERX_MODEL = "large-v3" # Highest accuracy model
```

**Test it:**

```bash
muneem doctor         # Verify setup
python ~/.muneem/transcriber.py   # Test transcription directly
```

Start a YouTube video or Zoom call - you should see transcript text appearing every 5 seconds.

---

## 4. Component 3 - Screen Capture and Reading

This captures what is visible on your screen at regular intervals and uses a vision model to extract context. **Screen capture is opt-in** - use `muneem start --screen` to enable. It helps identify participant names by analyzing the video call UI (Zoom, Teams, etc.).

Save as `~/.muneem/screen_reader.py` (the setup script writes this automatically):

```python
"""
Periodic screen capture + vision model OCR/context extraction via Ollama.

Screenshots are taken every CAPTURE_INTERVAL seconds (default: 1s).
Because the vision model takes several seconds per image, capture and analysis
run on separate threads: a fast capture loop saves screenshots to disk (every 1s), and
a slower analysis loop picks up the latest frame whenever it finishes the
previous one (target: every 2s) - so no frames queue up and RAM stays flat.
"""

import subprocess
import base64
import os
import time
import threading
import requests

CAPTURE_INTERVAL = 1  # seconds between screenshots
OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = "qwen3-vl:8b"
SCREENSHOT_DIR = str(Path.home() / ".muneem" / "tmp" / "screens")
LATEST_FRAME = os.path.join(SCREENSHOT_DIR, "latest.png")


def _ensure_dir():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def capture_screen() -> str:
    """Take a silent screenshot, overwriting the latest frame."""
    _ensure_dir()
    subprocess.run(
        ["screencapture", "-x", "-C", LATEST_FRAME],
        check=True,
        capture_output=True,
    )
    return LATEST_FRAME


def image_to_base64(path: str) -> str:
    """Read an image file and return its base64 encoding."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_screen(prompt: str = "Extract all visible text and describe the context of what is on screen. Be concise.") -> str:
    """Capture screen and send to vision model for analysis."""
    path = capture_screen()
    img_b64 = image_to_base64(path)

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def _capture_loop(interval: int, stop_event: threading.Event):
    """Fast loop: capture a screenshot every `interval` seconds."""
    _ensure_dir()
    while not stop_event.is_set():
        subprocess.run(
            ["screencapture", "-x", "-C", LATEST_FRAME],
            capture_output=True,
        )
        stop_event.wait(interval)


def _analysis_loop(callback, stop_event: threading.Event):
    """Slow loop: analyse the latest screenshot whenever the model is free."""
    last_mtime = 0.0
    while not stop_event.is_set():
        try:
            if not os.path.exists(LATEST_FRAME):
                time.sleep(0.5)
                continue

            mtime = os.path.getmtime(LATEST_FRAME)
            if mtime == last_mtime:
                time.sleep(0.5)
                continue
            last_mtime = mtime

            img_b64 = image_to_base64(LATEST_FRAME)
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": VISION_MODEL,
                    "prompt": "Extract all visible text and describe the context of what is on screen. Be concise.",
                    "images": [img_b64],
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            context = response.json()["response"]
            if callback:
                callback(context)
            else:
                print(f"[screen] {context[:200]}...")
        except Exception as e:
            print(f"[muneem] Screen analysis error: {e}")
            time.sleep(2)


def periodic_screen_reader(interval: int = CAPTURE_INTERVAL, callback=None):
    """
    Start capture + analysis threads.

    - Capture thread: takes a screenshot every `interval` seconds.
    - Analysis thread: sends the latest frame to the vision model as fast
      as the model can process, skipping intermediate frames automatically.
    """
    print(f"[muneem] Capturing every {interval}s (analysis runs as fast as model allows)...")
    stop = threading.Event()

    cap_thread = threading.Thread(target=_capture_loop, args=(interval, stop), daemon=True)
    ana_thread = threading.Thread(target=_analysis_loop, args=(callback, stop), daemon=True)

    cap_thread.start()
    ana_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        print("\n[muneem] Screen capture stopped.")


if __name__ == "__main__":
    print(read_screen())
```

**Test it:**

```bash
python ~/.muneem/screen_reader.py   # Test screen capture directly
```

It will capture your screen once and print what the vision model sees.

---

## 5. Component 4 - LLM Note Enhancement (Ollama)

This takes raw transcript + screen context + your manual notes and produces structured meeting notes - the core note-enhancement step.

Save as `~/.muneem/enhancer.py` (the setup script writes this automatically):

```python
"""
Combine transcript, screen context, and user notes into structured meeting notes
using a local LLM via Ollama.
"""

import requests
import json
from datetime import datetime

OLLAMA_URL = "http://localhost:11434"
ENHANCE_MODEL = "qwen3:14b"


DEFAULT_TEMPLATE = """You are a meeting notes assistant. Given a raw transcript, screen context, and the user's own notes, produce clean structured meeting notes.

Output format:
## Meeting Notes - {date}

### Summary
(2-3 sentence overview)

### Key Discussion Points
(Bullet points of what was discussed)

### Decisions Made
(Any decisions that were agreed upon)

### Action Items
(Who needs to do what, with deadlines if mentioned)

### Open Questions
(Anything unresolved)
"""

TEMPLATES = {
    "default": DEFAULT_TEMPLATE,

    "standup": """Produce standup meeting notes from the transcript and context.

Output format:
## Standup - {date}

### What was done (yesterday)
(Bullet points per person if identifiable)

### What's planned (today)
(Bullet points per person)

### Blockers
(Any blockers mentioned)
""",

    "one_on_one": """Produce 1:1 meeting notes.

Output format:
## 1:1 Notes - {date}

### Topics Discussed
(Main themes)

### Feedback Given
(Any feedback exchanged)

### Career / Growth
(Development topics if discussed)

### Action Items
(Next steps for each person)
""",

    "discovery": """Produce customer/user discovery call notes.

Output format:
## Discovery Call - {date}

### About Them
(Company, role, team size, context)

### Current Situation
(What they use today, pain points)

### Requirements
(What they need)

### Budget and Timeline
(If discussed)

### Objections / Concerns
(Hesitations mentioned)

### Next Steps
(Follow-up actions)
""",
}


def enhance_notes(
    transcript: str,
    screen_context: str = "",
    user_notes: str = "",
    template: str = "default",
) -> str:
    """Send all context to the LLM and get structured notes back."""

    system_prompt = TEMPLATES.get(template, TEMPLATES["default"]).replace(
        "{date}", datetime.now().strftime("%Y-%m-%d %H:%M")
    )

    user_message = f"""Here is everything from the meeting. Produce structured notes.

--- RAW TRANSCRIPT ---
{transcript}

--- SCREEN CONTEXT (what was visible on screen during the meeting) ---
{screen_context if screen_context else "(no screen context captured)"}

--- USER'S OWN NOTES ---
{user_notes if user_notes else "(no manual notes taken)"}

Now produce the structured meeting notes following the template exactly. Use /nothink mode for speed."""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": ENHANCE_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def chat_with_transcript(transcript: str, question: str) -> str:
    """Ask a question about the meeting transcript (post-meeting Q&A)."""
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": ENHANCE_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You have access to a meeting transcript. Answer the user's question based only on what was said in the meeting. Be specific and quote relevant parts.",
                },
                {
                    "role": "user",
                    "content": f"Transcript:\n{transcript}\n\nQuestion: {question}",
                },
            ],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


if __name__ == "__main__":
    sample_transcript = """
    Alice: Let's discuss the Q2 roadmap. We need to prioritise the API migration.
    Bob: I agree. The deadline is end of April. We also need to address the auth bug.
    Alice: Right, the auth bug is blocking three customers. Bob, can you take that this week?
    Bob: Sure, I'll have a fix by Wednesday.
    Alice: Great. I'll draft the migration plan and share it by Friday.
    """

    notes = enhance_notes(
        transcript=sample_transcript,
        template="default",
    )
    print(notes)
```

**Test it:**

```bash
python ~/.muneem/enhancer.py   # Test note enhancement directly
```

---

## 6. Component 5 - Putting It All Together (The Pipeline)

This is the main CLI script that ties all components together. It handles backend selection, preflight checks (Ollama, models, audio), the recording session with speaker attribution, and note generation.

The `muneem-setup.py` script writes this as `~/.muneem/app.py` automatically.

Key features of the pipeline:
- **Preflight checks**: Validates Ollama is running, required models are pulled, and at least one audio backend is available before starting
- **Backend selection**: Core Audio Tap > BlackHole > mic-only, with clear logging
- **Speaker-attributed transcription**: Each segment includes `{text, speaker, start, end}`
- **`muneem doctor`**: Full dependency verification - macOS version, native helper, BlackHole, Python packages, HF token, Ollama models
- **`muneem status`**: Quick overview of what is running and ready

Full source:

```python
"""
Muneem - Offline AI meeting notepad.

Captures audio + screen in real time, transcribes live, and generates
structured notes when the meeting ends. Runs 100% locally.

Usage:
    muneem start                      # Default template
    muneem start --template standup   # Use standup template
    muneem start --no-screen          # Skip screen capture
"""

import argparse
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

from transcriber import stream_transcribe, get_device_index
from screen_reader import read_screen, _capture_loop, _analysis_loop, LATEST_FRAME, _ensure_dir
from enhancer import enhance_notes, chat_with_transcript, TEMPLATES

NOTES_DIR = Path.home() / ".muneem" / "notes"
NOTES_DIR.mkdir(exist_ok=True)


class MeetingSession:
    def __init__(self, template: str = "default", enable_screen: bool = True):
        self.template = template
        self.enable_screen = enable_screen
        self.transcript_segments: list[str] = []
        self.screen_contexts: list[str] = []
        self.user_notes: str = ""
        self.start_time = datetime.now()
        self._running = False
        self._stop_event = threading.Event()

    def on_transcript(self, text: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {text}"
        self.transcript_segments.append(entry)
        print(f"  \033[92m▸\033[0m {entry}")

    def on_screen_context(self, context: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.screen_contexts.append(f"[{timestamp}] {context}")
        print(f"  \033[94m◉\033[0m Screen analysed at {timestamp}")

    def start_transcription(self, device_index: int | None):
        self._running = True
        stream_transcribe(device_index=device_index, callback=self.on_transcript)

    def start_screen_capture(self, interval: int = 2):
        """Launch a fast capture thread (every `interval` seconds) and a
        separate analysis thread that processes the latest frame as fast as
        the vision model allows - intermediate frames are skipped."""
        _ensure_dir()
        cap = threading.Thread(
            target=_capture_loop, args=(interval, self._stop_event), daemon=True
        )
        ana = threading.Thread(
            target=_analysis_loop, args=(self.on_screen_context, self._stop_event), daemon=True
        )
        cap.start()
        ana.start()

    def get_full_transcript(self) -> str:
        return "\n".join(self.transcript_segments)

    def get_screen_summary(self) -> str:
        return "\n\n".join(self.screen_contexts)

    def save_raw(self) -> Path:
        ts = self.start_time.strftime("%Y%m%d_%H%M%S")
        raw_path = NOTES_DIR / f"{ts}_raw_transcript.md"
        raw_path.write_text(
            f"# Raw Transcript - {self.start_time.strftime('%Y-%m-%d %H:%M')}\n\n"
            + self.get_full_transcript()
            + "\n\n---\n\n## Screen Context\n\n"
            + self.get_screen_summary()
        )
        return raw_path

    def save_enhanced(self, notes: str) -> Path:
        ts = self.start_time.strftime("%Y%m%d_%H%M%S")
        note_path = NOTES_DIR / f"{ts}_meeting_notes.md"
        note_path.write_text(notes)
        return note_path


def main():
    parser = argparse.ArgumentParser(description="Muneem - offline AI notepad")
    parser.add_argument(
        "--template",
        choices=list(TEMPLATES.keys()),
        default="default",
        help="Meeting notes template",
    )
    parser.add_argument(
        "--no-screen",
        action="store_true",
        help="Disable screen capture",
    )
    parser.add_argument(
        "--screen-interval",
        type=int,
        default=2,
        help="Seconds between screen captures (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="BlackHole",
        help="Audio input device name substring (default: BlackHole)",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              MUNEEM - Offline AI Notepad                ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Template:    {args.template:<42}║")
    print(f"║  Screen:      {'ON (every ' + str(args.screen_interval) + 's)' if not args.no_screen else 'OFF':<42}║")
    print(f"║  Audio:       {args.device:<42}║")
    print("║                                                          ║")
    print("║  Press Ctrl+C to stop and generate notes.                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    device_index = get_device_index(args.device)
    if device_index is None:
        print(f"⚠  '{args.device}' not found. Using default microphone.")
    else:
        print(f"✓  Audio device: {args.device} (index {device_index})")

    session = MeetingSession(
        template=args.template,
        enable_screen=not args.no_screen,
    )

    # Start screen capture (fast capture thread + separate analysis thread)
    if not args.no_screen:
        session.start_screen_capture(interval=args.screen_interval)
        print(f"✓  Screen capture started (every {args.screen_interval}s, analysis runs in parallel).")

    print("✓  Transcription starting... speak or play audio.\n")

    try:
        session.start_transcription(device_index=device_index)
    except KeyboardInterrupt:
        pass

    session._running = False
    session._stop_event.set()

    print("\n")
    print("═══════════════════════════════════════════════════════════")
    print("  Meeting ended. Processing notes...")
    print("═══════════════════════════════════════════════════════════")

    # Save raw transcript
    raw_path = session.save_raw()
    print(f"\n  Raw transcript saved: {raw_path}")

    if not session.transcript_segments:
        print("  No transcript captured. Exiting.")
        return

    # Generate enhanced notes
    print(f"  Generating enhanced notes (template: {args.template})...")
    print("  This may take 30-60 seconds...\n")

    enhanced = enhance_notes(
        transcript=session.get_full_transcript(),
        screen_context=session.get_screen_summary(),
        user_notes=session.user_notes,
        template=args.template,
    )

    note_path = session.save_enhanced(enhanced)
    print(enhanced)
    print(f"\n  Enhanced notes saved: {note_path}")

    # Post-meeting chat loop
    print("\n  ─── Ask questions about this meeting (type 'quit' to exit) ───\n")
    transcript = session.get_full_transcript()
    while True:
        try:
            question = input("  You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            answer = chat_with_transcript(transcript, question)
            print(f"\n  AI: {answer}\n")
        except (KeyboardInterrupt, EOFError):
            break

    print("\n  Done. Notes saved to ~/.muneem/notes/")


if __name__ == "__main__":
    main()
```

### Setup (Automatic via setup script)

If you ran `python3 muneem-setup.py`, all four modules are already installed to `~/.muneem/` and the `muneem` CLI is on your PATH. No manual steps needed.

### Running It

```bash
# Default meeting
muneem start

# Standup meeting
muneem start --template standup

# 1:1 meeting
muneem start --template one_on_one

# Customer discovery call
muneem start --template discovery

# Without screen capture (faster, less resource usage)
muneem start --no-screen

# Use microphone instead of system audio
muneem start --device "MacBook"

# List saved notes
muneem notes

# Open the latest note
muneem notes last

# Ask about the last meeting
muneem ask "what were the action items?"

# Check system status
muneem status

# Verify all dependencies
muneem doctor
```

### What Happens

1. Audio capture starts (system audio via BlackHole or microphone).
2. Transcript appears in real time in your terminal (green `▸` markers).
3. Screen is captured every 2 seconds and sent to the vision model (blue `◉` markers).
4. Press **Ctrl+C** when the meeting ends.
5. Ollama generates structured notes from transcript + screen context.
6. Notes are saved to `~/.muneem/notes/` as Markdown files.
7. A Q&A loop opens so you can ask questions about the meeting.

---

## 7. Ready-Made Alternatives

If you prefer a polished app over a DIY build, these open-source projects do most of the above out of the box. All run fully offline.

### 7.1 ownscribe (Recommended CLI tool)

The closest to our pipeline, already packaged. Python + Swift. No virtual audio driver needed (uses Core Audio Taps on macOS 14.2+).

**Install:**

```bash
pip install ownscribe

# Also needs:
xcode-select --install  # Xcode CLI tools
brew install ffmpeg
```

**Run:**

```bash
ownscribe
# Records system audio → transcribes with WhisperX → summarises with local LLM
# Press Ctrl+C to stop
```

**Features:**
- System audio capture via Core Audio Taps (no BlackHole needed)
- Microphone support with `--mic` flag
- Speaker diarization (who said what)
- Summarisation via built-in Phi-4-mini or your Ollama server
- Natural language querying across past meetings
- Auto-stop after 5 minutes of silence

**GitHub:** <https://github.com/paberr/ownscribe>

### 7.2 Meetily (GUI app)

Full desktop app with a polished notepad UI. Whisper transcription + Ollama summarisation.

```bash
brew tap zackriya-solutions/meetily
brew install --cask meetily
meetily-server --language en --model medium
```

Then open Meetily from Applications. Works with Zoom, Teams, Meet.

**GitHub:** <https://github.com/Zackriya-Solutions/meeting-minutes>

### 7.3 Notes4Me (Simple Python + BlackHole)

Minimal Python tool, very similar to our DIY approach. Good reference implementation.

**GitHub:** <https://github.com/andyj/Notes4Me>

Requires BlackHole + Whisper + Ollama, configured via `.env` file.

---

## 8. Templates for Different Meeting Types

These are the templates built into the `enhancer.py` above. You can add your own by editing the `TEMPLATES` dictionary.

| Template | Command | Best For |
|---|---|---|
| `default` | `muneem start` | General meetings, ad-hoc calls |
| `standup` | `muneem start --template standup` | Daily standups, sprint syncs |
| `one_on_one` | `muneem start --template one_on_one` | 1:1s, manager check-ins |
| `discovery` | `muneem start --template discovery` | Customer/user discovery calls, sales calls |

**To add a custom template**, add a new entry to the `TEMPLATES` dict in `~/.muneem/enhancer.py`:

```python
TEMPLATES["retro"] = """Produce sprint retrospective notes.

Output format:
## Retrospective - {date}

### What went well

### What didn't go well

### Action items for next sprint
"""
```

Then use it: `muneem start --template retro`

---

## 9. Reference - Useful Commands

```bash
# --- Muneem CLI ---
muneem start                              # Default meeting (Core Audio Tap + WhisperX)
muneem start --template standup           # Standup template
muneem start --template one_on_one        # 1:1 template
muneem start --template discovery         # Discovery call template
muneem start --no-screen                  # Audio only, no screen reading
muneem start --screen-interval 5          # Capture screen every 5s instead of default 2s
muneem notes                              # List all saved notes
muneem notes last                         # Open the most recent note
muneem ask "who has the action items?"    # Ask about the last meeting
muneem status                             # Check Ollama, models, audio backend, diarization
muneem doctor                             # Verify all dependencies
muneem help                               # Show usage

# --- Test individual components ---
ollama ps                                 # Is the LLM loaded?
ollama run qwen3:14b "test"               # Test the LLM
python ~/.muneem/screen_reader.py         # Test screen capture
python ~/.muneem/transcriber.py           # Test audio transcription

# --- View saved notes ---
ls ~/.muneem/notes/
open ~/.muneem/notes/                     # Open in Finder

# --- Audio backend debugging ---
~/.muneem/native/muneem-audio system /tmp/test_sys.wav 5   # Test Core Audio Tap (5s)
~/.muneem/native/muneem-audio mic /tmp/test_mic.wav 5      # Test mic capture (5s)
python3 -c "
import pyaudio; p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'  [{i}] {info[\"name\"]}')
p.terminate()
"

# --- BlackHole (fallback) ---
brew list blackhole-2ch                   # Verify BlackHole is installed
open -a "Audio MIDI Setup"               # Configure Multi-Output

# --- Diarization (fully offline, no setup needed) ---
python3 -c "from diarize import diarize; print(diarize)"  # Verify installed

# --- Model management ---
ollama pull qwen3:14b                     # Thinking/enhancement model
ollama pull qwen3-vl:8b                   # Vision/screen reading model

# --- Re-run setup if needed ---
python3 muneem-setup.py                   # Idempotent - safe to re-run
```

---

## Muneem Feature Summary

| Feature | Muneem |
|---|---|
| Audio transcription | WhisperX large-v3 with forced alignment, offline |
| Speaker diarization | sherpa-onnx (pyannote-segmentation + 3D-Speaker CAM++), offline. Stable Speaker 0/1/2 labels across the meeting via a session-scoped SpeakerRegistry. |
| Screen reading | Vision model + `screencapture`. Reads participant names and shared content. |
| Note enhancement | Ollama (qwen3:14b default; qwen3:32b if 48 GB+ RAM), offline |
| Meeting templates | default, standup, 1:1, discovery (customisable) |
| Post-meeting Q&A | `muneem ask "your question"` |
| Note management | `muneem notes` / `muneem notes last` |
| System health check | `muneem doctor` |
| System audio capture | Core Audio Tap (preferred) + BlackHole (fallback) |
| Modes | Real-time (default) or `--post-process` (record now, transcribe after stop) |
| Setup | `python3 muneem-setup.py` - one command |
| macOS requirement | macOS 15+ (Sequoia / Tahoe), Apple Silicon |
| Cost | Free (after hardware) |
| Privacy | 100% local, zero network calls after install |
| Internet required | No (after initial model download) |

---

*Last updated: March 2026*
