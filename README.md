# Muneem

**Offline AI meeting notepad for macOS Apple Silicon.**

Muneem records your call audio, transcribes it with WhisperX large-v3, tracks
who's speaking with a local vision model, and writes speaker-labelled notes
using a local LLM. Everything runs on your Mac — no cloud, no accounts, no
API keys, no tokens, no telemetry. After the one-time install, Muneem works
offline.

- **Docs site:** <https://sudarshan-uc.github.io/muneem/>
- **License:** see [`LICENSE`](LICENSE)

---

## How it works

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Audio capture   │    │  Screen capture  │    │  Mic capture     │
│  (Core Audio Tap │    │  (CGWindowList)  │    │  (portaudio)     │
│   or BlackHole)  │    │                  │    │                  │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         └───────────┬───────────┴───────────┬───────────┘
                     │                       │
                     ▼                       ▼
         ┌───────────────────────┐ ┌───────────────────────┐
         │  WhisperX large-v3    │ │  Qwen3-VL vision      │
         │  30 s segments +      │ │  "who is on screen,   │
         │  sherpa-onnx diariz.  │ │   who is talking"     │
         └──────────┬────────────┘ └──────────┬────────────┘
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                    ┌───────────────────────┐
                    │  Local LLM (Ollama)   │
                    │  summary, decisions,  │
                    │  action items         │
                    └──────────┬────────────┘
                               ▼
                    ┌───────────────────────┐
                    │  ~/.muneem/notes/     │
                    │    *_raw_transcript   │
                    │    *_transcript       │
                    │    *_notes            │
                    └───────────────────────┘
```

The pipeline:

1. **Capture.** Core Audio Tap records system audio from a selected window
   (or display), portaudio records the mic, and CGWindowList grabs frames of
   the call window for the vision model.
2. **Transcribe.** WhisperX large-v3 runs in 30-second segments with
   wav2vec2 alignment; sherpa-onnx diarizes speakers across the session.
3. **Name speakers.** Qwen3-VL reads each frame and returns the visible
   participant roster plus the active speaker. These signals rewrite
   generic `Speaker 0 / 1 / 2` labels into real names wherever the vision
   model saw them.
4. **Stop cleanly.** Hit `Ctrl+C`, close the call window, or quit the
   selected app — Muneem notices and finalises the session either way.
5. **Enhance.** A local Ollama LLM (default `qwen3:14b`, or `qwen3:32b` on
   machines with 48 GB+ RAM) turns the transcript into structured notes.

See [`Build-Muneem.md`](Build-Muneem.md) for a deep-dive on every component.

---

## Requirements

| Requirement | Minimum |
|---|---|
| **macOS** | 15 (Sequoia) or newer |
| **Chip** | Apple Silicon (M1 / M2 / M3 / M4). Intel is not supported. |
| **RAM** | 16 GB (14B LLM). 48 GB+ unlocks the 32B LLM. |
| **Disk** | ~15 GB free (models + Python venv) |
| **Python** | 3.10 – 3.13 |
| **Network** | install-time only. Runs offline afterwards. |

The installer verifies every row above and stops with a clear error if any
prerequisite is missing.

---

## Install

Clone this repo and run the single-file installer:

```bash
git clone https://github.com/sudarshan-uc/muneem.git
cd muneem
python3 muneem-setup.py
```

Or grab just the installer:

```bash
curl -LO https://raw.githubusercontent.com/sudarshan-uc/muneem/main/muneem-setup.py
python3 muneem-setup.py
```

The installer runs 10 idempotent steps (~15–30 min on first install,
fast on re-runs):

1. Verify prereqs (macOS 15+, arm64, Python 3.10–3.13, Homebrew, Xcode CLT)
2. Install Homebrew packages (`ffmpeg`, `sox`, `portaudio`, `ollama`, `blackhole-2ch`)
3. Create `~/.muneem/` layout
4. Compile + ad-hoc codesign the Core Audio Tap helper (Swift `.app`)
5. Detect RAM and chip
6. Prompt for LLM tier and write `~/.muneem/config.json`
7. Create the Python venv, install pip packages, pre-cache WhisperX + alignment
8. Download diarization ONNX models (~34 MB)
9. Start Ollama and pull the selected LLM + `qwen3-vl:8b` vision model
10. Install the `muneem` CLI at `/usr/local/bin/muneem` (one `sudo` prompt)

Full details, verification commands, and troubleshooting:
[`INSTALL.md`](INSTALL.md).

### First-time macOS permissions

Your first `muneem start` triggers three prompts — **grant all three**, then
quit and reopen Terminal (macOS caches grants per-process):

- **Microphone** — for `muneem-audio.app` and your terminal
- **Screen Recording** — for your terminal
- **Audio Capture (System Audio)** — for `muneem-audio.app` on macOS 15+

---

## Use

```bash
muneem start              # pick a window / display, record + transcribe live
muneem start --window Zoom
muneem start --post-process  # low-CPU mode: transcribe after the call
muneem notes last         # open the latest note
muneem ask "what did we decide about Q3?"
muneem doctor             # full dependency + permission audit
muneem status             # quick live check
muneem config llm qwen3:32b
muneem uninstall          # keeps notes; --all to remove notes too
```

A meeting produces three files in `~/.muneem/notes/`:

| File | Content |
|---|---|
| `*_raw_transcript.md` | Unedited raw transcript + screen context |
| `*_transcript.md` | Speaker-grouped paragraphs, real names where detected |
| `*_notes.md` | LLM-enhanced summary, decisions, action items |

Full command reference: [`USAGE.md`](USAGE.md).

---

## Repo layout

```
muneem-setup.py              Single-file installer. Embeds four Python modules
                             (transcriber, screen_reader, enhancer, app) and a
                             Swift audio helper. Running it writes to ~/.muneem/.
INSTALL.md                   Install guide + prerequisite matrix.
USAGE.md                     Command reference.
Build-Muneem.md              Architecture deep-dive.
Local-LLM-Setup-Guide.md     Picking and running a local LLM + vision model.
mkdocs.yml                   Site config for https://sudarshan-uc.github.io/muneem/
scripts/sanity_check.py      Repo self-check: versions, embedded modules, deps.
.github/workflows/
  ci.yml                     Pre-commit + sanity check on PRs.
  docs.yml                   Builds and deploys the docs site on push to main.
```

Runtime state lives at `~/.muneem/` (venv, models, notes, config). The repo
itself contains only source and docs.

---

## Contributing

Pre-commit hooks enforce trailing-whitespace fixes, ruff E9/F63/F7/F82,
`py_compile muneem-setup.py`, and `scripts/sanity_check.py`:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

All four embedded modules inside `muneem-setup.py` must parse as valid
Python — `sanity_check.py` extracts each one with the same regex the
installer uses and runs `ast.parse` on it. Bump `__version__` in
`muneem-setup.py` when cutting a release.

---

## License

See [`LICENSE`](LICENSE).
