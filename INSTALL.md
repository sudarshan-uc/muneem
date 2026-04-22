# Muneem - Install Guide

## What Muneem does

Muneem is an offline AI meeting notepad for macOS Apple Silicon that records your call audio, transcribes it with WhisperX large-v3, and writes speaker-labelled notes using a local Ollama LLM. Nothing leaves your Mac: no cloud, no accounts, no API tokens, and everything runs offline after the one-time install.

## Prerequisites & Requirements

| Requirement | Minimum | How to get it |
|---|---|---|
| **macOS** | 15 (Sequoia) or newer - needed for Core Audio Taps | Apple Menu > System Settings > General > Software Update |
| **Chip** | Apple Silicon (M1 / M2 / M3 / M4) | Hardware only - Intel Macs are not supported |
| **RAM** | 16 GB (14B LLM). 48 GB+ unlocks the 32B LLM | Hardware only |
| **Disk** | ~15 GB free (models + Python venv) | Free up space before running the installer |
| **Xcode Command Line Tools** | any current version | `xcode-select --install` (macOS prompts a GUI installer) |
| **Homebrew** | any current version | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |
| **Python** | 3.10 - 3.13 | `brew install python@3.12` (3.11 or 3.12 recommended; 3.14+ is not yet supported) |
| **Network** | working internet, install-time only | Wi-Fi / Ethernet during install; everything runs offline afterwards |

The installer verifies every row above and stops early with a clear error if any prerequisite is missing, so you do not need to pre-check manually.

## Validate dependency resolution

The installer pulls and validates the following for you. You can also check any of them independently:

| Dependency | What installs it | How to verify |
|---|---|---|
| **ffmpeg, sox, portaudio** | `brew install ffmpeg sox portaudio` | `brew list ffmpeg sox portaudio` |
| **Ollama runtime** | `brew install ollama` | `ollama --version` and `curl http://localhost:11434/api/tags` |
| **BlackHole 2ch** (audio fallback) | `brew install --cask blackhole-2ch` | `brew list --cask blackhole-2ch` |
| **Core Audio Tap helper** | compiled + codesigned by the installer | `ls ~/.muneem/native/muneem-audio.app` |
| **Python venv + pip packages** | `~/.muneem/venv/` | `~/.muneem/venv/bin/pip list` |
| **WhisperX large-v3 + alignment** | pre-cached from HuggingFace | `ls ~/.cache/whisperx` and `ls ~/.cache/huggingface` |
| **Diarization ONNX models** | downloaded to `~/.muneem/models/` | `ls ~/.muneem/models/` (expect ~34 MB total) |
| **Ollama models** (LLM + vision) | `ollama pull qwen3:14b qwen3-vl:8b` | `ollama list` |
| **`muneem` CLI shim** | symlinked to `/usr/local/bin/muneem` | `which muneem && muneem --help` |

Run the bundled end-to-end audit any time:

```bash
muneem doctor
```

`doctor` verifies macOS version, Homebrew formulae, the Python venv, every pip package, the Swift audio helper, the BlackHole fallback, the diarization ONNX models, Ollama connectivity, and that the required Ollama models are present. It reports green / yellow / red per dependency and ends with an actionable fix suggestion for anything that is not green.

## Install Muneem

From a clone of this repo (or the standalone installer), run:

```bash
python3 "/path/to/muneem-setup.py"
```

The installer executes ten steps (~15-30 min on first install; subsequent re-runs skip anything already present):

1. **Prereqs** - macOS 15+, arm64, Python 3.10-3.13, Homebrew, Xcode CLT
2. **Homebrew packages** - `ffmpeg`, `sox`, `portaudio`, `ollama`, `blackhole-2ch`
3. **Project directories** - creates `~/.muneem/` with `notes/`, `native/`, `models/`, `venv/`, `tmp/`
4. **Core Audio Tap helper** - compiles, ad-hoc codesigns, and registers the Swift `.app` bundle
5. **Resource check** - detects RAM and chip; gates the 32B LLM behind 48+ GB RAM
6. **LLM tier** - prompts for the notes LLM (14B default, 32B if RAM permits) and writes `~/.muneem/config.json`
7. **Python environment** - creates the venv, installs all pip packages, pre-caches WhisperX large-v3 + the wav2vec2 alignment model
8. **Diarization ONNX models** - downloads pyannote-segmentation + 3D-Speaker CAM++ embeddings (~34 MB)
9. **Ollama models** - starts Ollama and pulls the selected LLM + `qwen3-vl:8b` vision model (~10-20 GB)
10. **CLI** - installs `muneem` at `/usr/local/bin/muneem` (single `sudo` prompt)

### First-time macOS permissions

Your first `muneem start` triggers three macOS permission prompts. **Grant all three**, then quit and re-open Terminal (macOS caches grants per-process):

- **Microphone** - for `muneem-audio.app` and your terminal
- **Screen Recording** - for your terminal (screen capture + vision context)
- **Audio Capture (System Audio)** - for `muneem-audio.app` (macOS 15+ Core Audio Tap grant)

Re-run `muneem doctor` at any time to confirm permissions and dependency state.

## Uninstall

```bash
muneem uninstall           # keeps notes in ~/.muneem/notes/
muneem uninstall --all     # removes notes too
```

Uninstall does not touch Homebrew, Ollama, or downloaded Ollama models.
