# Muneem - Usage Guide

A meeting has three phases: **start → talk → stop**. Muneem records, transcribes, diarizes, watches the screen, and writes the notes. You just press Ctrl+C when the call ends.

## Start a meeting

```bash
muneem start
```

Muneem picks the target interactively: full screen, a specific display, or a specific app window (Zoom, Teams, Webex, Slack, browser tabs of Meet / web-Zoom, etc.). Then it starts capturing audio + screen + transcribing in real time.

### Common flags

| Flag | Effect |
|---|---|
| `--template <name>` | `default`, `standup`, `one_on_one`, or `discovery` |
| `--no-screen` | Audio only (no vision, no screen-based speaker names) |
| `--screen` | Force screen capture even if no call app is detected |
| `--window Zoom` | Skip the picker, follow a specific app window |
| `--display 2` | Skip the picker, follow display 2 |
| `--record-share 1` | Also record display 1 as the screen you're *sharing* |
| `--mic "Galaxy Buds2"` | Pick a specific mic by name substring |
| `--output "MacBook Pro Speakers"` | Pick a specific output device |
| `--no-audio-prompt` | Skip mic/output prompts, use current system defaults |
| `--post-process` | Only record raw audio during the call; transcribe after stop (low CPU - ideal for heavy video calls) |

### Modes

- **Real-time (default)** - WhisperX + diarization run while you're on the call. You see transcripts appear live.
- **Post-process** - audio is recorded only; all transcription + diarization + enhancement happens after you hit Ctrl+C. Uses less CPU during the call. Timestamps in the final transcript still reflect meeting time.

### While recording

- Transcripts stream live: `▸ [HH:MM:SS] [Speaker 1] ...`
- Vision model ticks periodically: `◉ vision 3800ms - active: <Name>`
- A backlog warning fires if transcription falls behind recording.
- `Ctrl+C` stops the session and triggers note generation.

## After the meeting

Muneem produces three files in `~/.muneem/notes/`:

| File | What it is |
|---|---|
| `YYYYMMDD_HHMMSS_raw_transcript.md` | Unedited raw transcript + screen context |
| `YYYYMMDD_HHMMSS_transcript.md` | Full transcript - speaker-grouped paragraphs |
| `YYYYMMDD_HHMMSS_notes.md` | LLM-enhanced notes - summary, decisions, action items |

Generic `Speaker 0 / 1 / 2` labels are rewritten to real names using the vision model's active-speaker signals (`_rewrite_speaker_labels`) whenever participant names were visible on screen.

## Browse notes

```bash
muneem notes            # list all saved notes
muneem notes last       # open the latest note
```

## Ask questions about a meeting

```bash
muneem ask "what was decided about the Q3 roadmap?"
muneem ask "list every action item assigned to me"
```

By default, `ask` uses the most recent meeting. The question is answered by the local LLM using that meeting's transcript + screen context.

## Check system health

```bash
muneem status           # live check: audio backend, models, permissions
muneem doctor           # full dependency audit
```

`doctor` checks macOS version, brew formulae, Python venv, every pip package, the Swift native helper, BlackHole fallback, ONNX diarization models, and Ollama model availability.

## Configure

```bash
muneem config show           # show current config
muneem config llm qwen3:14b  # default LLM (fits in 16 GB RAM)
muneem config llm qwen3:32b  # higher quality (requires ≥48 GB RAM)
```

Config lives at `~/.muneem/config.json`. Whisper and the vision model are fixed (swapping them would change note quality between machines). Only the LLM is user-selectable.

## Troubleshooting

**No system audio in transcripts.** Run `muneem doctor`. If Core Audio Tap is unavailable and BlackHole isn't routing, set up a Multi-Output Device in **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup" → `+` → Multi-Output → check both BlackHole 2ch and your speakers → right-click → "Use This Device For Sound Output").

**Transcript says "Speaker 1" / "Speaker 2" and they're swapped.** This should be rare - sherpa-onnx's session-wide SpeakerRegistry keeps IDs stable. If it happens, check `muneem doctor` - the ONNX diarization models must be present; falling back to the `diarize` package degrades to per-segment clustering.

**Ollama model missing.** `ollama pull qwen3:14b` (or whichever `muneem doctor` reports as missing).

**Permission prompts never appeared.** macOS caches grants per-process. Quit Terminal completely (`Cmd+Q`), reopen, try again. Or toggle the permission off then on in **System Settings → Privacy & Security**.

**Transcription backlog warnings during a heavy call.** Re-run with `--post-process` - no transcription load while the call is live.

## Uninstall

See `INSTALL.md`. Your notes survive `muneem uninstall` by default.
