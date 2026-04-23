# Muneem 0.1.0 - pre-release

Pre-release for internal testers. Offline AI meeting notepad for macOS Apple Silicon.

## Install

```bash
# 1. Download the binary from this release page (muneem-installer)
curl -LO https://github.com/sudarshan-uc/muneem-release/releases/latest/download/muneem-installer

# 2. Remove macOS quarantine (binary is ad-hoc signed, not notarized)
xattr -d com.apple.quarantine muneem-installer

# 3. Run the installer
chmod +x muneem-installer
./muneem-installer
```

The installer runs the same ten steps as the source install: Homebrew packages,
Core Audio Tap helper, Python venv, WhisperX large-v3, diarization models,
Ollama LLM + vision model, and the `muneem` CLI at `/usr/local/bin/muneem`.

Verify the binary before running:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

## Requirements

- macOS 15 (Sequoia) or newer
- Apple Silicon (M1 / M2 / M3 / M4)
- 16 GB RAM (14B LLM); 48 GB+ unlocks the 32B LLM
- ~15 GB free disk
- Working internet during install (runs offline afterwards)

## What changed

Populated by the release workflow or hand-edited per release.

## Known limitations

- Unsigned / ad-hoc only. Gatekeeper will block first run unless you
  clear the quarantine attribute as shown above.
- No auto-updater yet. Re-run the installer to pick up a new version.
