# Muneem - Install Guide

Offline AI meeting notepad for macOS Apple Silicon. Runs entirely on your Mac - no cloud, no accounts, no tokens. One command installs everything.

## Requirements

| | |
|---|---|
| **OS** | macOS 15 (Sequoia) or newer - Core Audio Taps |
| **Chip** | Apple Silicon (M1/M2/M3/M4) |
| **Python** | 3.10 - 3.13 (installer verifies) |
| **Tools** | Homebrew, Xcode Command Line Tools (installer auto-prompts if missing) |
| **RAM** | 16 GB minimum (14B LLM). 48 GB+ unlocks the 32B LLM option. |
| **Disk** | ~15 GB free (models + venv) |
| **Network** | Needed for initial install only. Everything runs offline afterwards. |

## Install

```bash
python3 "/path/to/muneem-setup.py"
```

The installer runs 10 steps (~15 - 30 min on first install, mostly model downloads):

1. Verify prereqs (macOS version, arm64, Python, Homebrew, Xcode CLT)
2. `brew install` → ffmpeg, sox, portaudio, ollama, blackhole-2ch
3. Create `~/.muneem/` with `notes/`, `native/`, `models/`, `venv/`, `tmp/`
4. Compile + sign the Swift Core Audio Tap helper (`.app` bundle)
5. Detect RAM & chip, pick the LLM tier (14B default, 32B if ≥48 GB)
6. Create Python venv + install pip packages + pre-cache WhisperX large-v3 and alignment models
7. Download diarization ONNX models (pyannote-seg + 3D-Speaker CAM++, ~34 MB)
8. Write embedded app modules (`transcriber.py`, `screen_reader.py`, `enhancer.py`, `app.py`)
9. Start Ollama and `ollama pull` the LLM + vision models (~10 - 20 GB)
10. Install the `muneem` CLI at `/usr/local/bin/muneem`

## First-time permissions

The first `muneem start` will prompt for macOS permissions. **Grant all three** - then quit and restart the Terminal (macOS caches the grants per-process):

- **Microphone** - for `muneem-audio.app` and your terminal
- **Screen Recording** - for your terminal (needed for screen capture + vision context)
- **Audio Capture (System Audio)** - for `muneem-audio.app` (this is the macOS 15+ Core Audio Tap grant)

Check status any time with:

```bash
muneem doctor
```

## Uninstall

```bash
muneem uninstall           # keeps notes in ~/.muneem/notes/
muneem uninstall --all     # removes notes too
```

Uninstall does not touch Homebrew, Ollama, or downloaded Ollama models.
