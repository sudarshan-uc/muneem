#!/usr/bin/env python3
"""
Muneem Setup - standalone installer for the Muneem offline AI notepad.

Usage:
  python3 muneem-setup.py                  Install / reinstall
  python3 muneem-setup.py uninstall        Uninstall (keeps notes)
  python3 muneem-setup.py uninstall --all  Uninstall including notes

This script is fully self-contained. It can be run from any directory.
Everything is installed to ~/.muneem/. A copy of this script is placed
there too, so re-runs and uninstalls work without the original source.

What it does:
  1. Checks macOS 15+ / Apple Silicon / Python 3.10-3.13 prerequisites
  2. Installs Homebrew packages (ffmpeg, sox, portaudio, ollama, blackhole-2ch)
  3. Creates project directories at ~/.muneem/
  4. Compiles the Core Audio Tap helper (default system audio backend)
  5. Detects system resources (RAM, chip; gates 32B LLM behind 48+ GB RAM)
  6. Prompts for notes LLM (14B default, 32B optional if RAM permits)
  7. Creates a Python venv and installs pip dependencies (torch, whisperx, ...)
  8. Writes the embedded application modules + copies this installer
  9. Pulls required Ollama models (selected LLM + qwen3-vl:8b vision)
 10. Creates the `muneem` CLI wrapper and symlinks it into /usr/local/bin

Requirements:
  - macOS 15+ (Sequoia or later) - Apple Silicon only
  - System audio: Core Audio Tap by default, BlackHole as fallback
"""

import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

__version__ = "0.1.0"

# ─── Configuration ───────────────────────────────────────────────────────────

MUNEEM_HOME = Path.home() / ".muneem"
VENV_DIR = MUNEEM_HOME / "venv"
BIN_DIR = MUNEEM_HOME / "bin"
NOTES_DIR = MUNEEM_HOME / "notes"
TMP_DIR = MUNEEM_HOME / "tmp"
NATIVE_DIR = MUNEEM_HOME / "native"

BREW_PACKAGES = ["ffmpeg", "sox", "portaudio", "ollama"]
BREW_CASKS = ["blackhole-2ch"]  # Fallback when Core Audio Tap unavailable
PIP_PACKAGES = [
    "whisperx",
    "diarize",        # kept as fallback only; sherpa-onnx is primary (much better labels)
    "sherpa-onnx",    # OFFLINE diarization: pyannote-segmentation + 3D-Speaker embeddings
                      #   via ONNX Runtime. Apache-2.0, no HuggingFace gating. Replaces
                      #   `diarize` package as the primary diarization backend - fixes the
                      #   male/female-swapped-between-Speaker-1-and-2 issue caused by
                      #   per-segment clustering in the old path.
    # pyaudio removed: crashes on macOS 26 Tahoe (PortAudio null fn ptr in PaMacCore_Initialize)
    # Audio recording now uses sox (CLI) and the Swift muneem-audio binary instead.
    "numpy",          # Let pip resolve; torch>=2.4 supports numpy 2.x (old conflict was torch 2.2.x only)
    "requests",
    "Pillow",
    "rich",
    "torch>=2.4",     # WhisperX requires torch>=2.4; 2.2.x disables the transcription engine
    "torchaudio>=2.4",  # Must match torch version
    "pyobjc-framework-Quartz",
]
# ─── Ollama model policy ──────────────────────────────────────────────────────
# Whisper (large-v3) and the vision model are FIXED: every install produces
# identical speech-to-text and screen-reading output. Varying those models
# would make the same meeting yield different notes on different machines -
# not acceptable for a note-taking tool.
#
# The notes LLM is user-selectable between 14B (default, fits in 16 GB RAM)
# and 32B (higher quality, requires 48+ GB RAM). Default stays at 14B so
# machines with less RAM get a predictable experience. The chosen model
# is persisted in ~/.muneem/config.json and can be switched any time via
# `muneem config llm <name>`.
VISION_OLLAMA_MODEL = "qwen3-vl:8b"
LLM_DEFAULT   = "qwen3:14b"   # ~9.3 GB on disk, ~12 GB in use
LLM_HIGH_TIER = "qwen3:32b"   # ~20 GB on disk, ~22 GB in use
LLM_MIN_RAM_GB = 16           # minimum to run muneem at all
LLM_HIGH_TIER_MIN_RAM_GB = 48 # minimum to offer the 32B option
# Kept as a backwards-compat alias for any external callers - not used internally.
OLLAMA_MODELS = [LLM_DEFAULT, VISION_OLLAMA_MODEL]

MIN_MACOS_MAJOR = 15  # Sequoia (Core Audio Taps)

# ─── Python version policy ────────────────────────────────────────────────────
# The only hard constraint comes from `ctranslate2` (pulled by whisperx) and
# `torch`, both of which ship manylinux/macOS wheels for CPython 3.10 - 3.13.
#
#   - Python 3.9 and earlier: torch 2.4+ has no wheels. Unsupported.
#   - Python 3.10 - 3.13:     supported - wheels exist for every dependency.
#   - 3.11 / 3.12:            recommended (longest track record with WhisperX).
#   - 3.14+:                  unsupported until ctranslate2 ships wheels.
#
# Everything else in the stack (Quartz, Pillow, numpy, rich, requests, pyobjc)
# is wide-compat and doesn't constrain the choice.
SUPPORTED_PY = [(3, 10), (3, 11), (3, 12), (3, 13)]
RECOMMENDED_PY = [(3, 12), (3, 11)]
SUPPORTED_PY_STR = "3.10 - 3.13"

# ─── Helpers ─────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def info(msg):
    print(f"  {GREEN}✓{RESET}  {msg}")


def warn(msg):
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def fail(msg):
    print(f"  {RED}✗{RESET}  {msg}")
    sys.exit(1)


def header(msg):
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"  {BOLD}{msg}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}\n")


def run(cmd, check=True, capture=False, **kwargs):
    if capture:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
        if check and r.returncode != 0:
            fail(f"Command failed: {cmd}\n{r.stderr.strip()}")
        return r
    else:
        r = subprocess.run(cmd, shell=True, **kwargs)
        if check and r.returncode != 0:
            fail(f"Command failed: {cmd}")
        return r


def is_installed(cmd):
    return shutil.which(cmd) is not None


def _find_supported_python() -> tuple[int, int, str | None]:
    """Return (major, minor, absolute_path) for the best available Python in SUPPORTED_PY.

    Preference order: RECOMMENDED_PY first (3.12, 3.11), then remaining SUPPORTED_PY
    newest-first (3.13, 3.10). Looks on PATH, then Homebrew canonical locations.
    Returns (0, 0, None) if nothing supported is found.
    """
    # Walk: recommended first, then remaining supported newest-first.
    extras = sorted((p for p in SUPPORTED_PY if p not in RECOMMENDED_PY), reverse=True)
    for maj, min_ in RECOMMENDED_PY + extras:
        name = f"python{maj}.{min_}"
        candidates = [
            shutil.which(name),
            f"/opt/homebrew/opt/python@{maj}.{min_}/bin/{name}",
            f"/usr/local/opt/python@{maj}.{min_}/bin/{name}",
        ]
        for c in candidates:
            if not c or not Path(c).exists():
                continue
            # Confirm the binary actually reports a supported version.
            try:
                r = subprocess.run(
                    [c, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
                    capture_output=True, text=True, timeout=10,
                )
                pmaj, pmin = (int(x) for x in r.stdout.strip().split())
            except Exception:
                continue
            if (pmaj, pmin) in SUPPORTED_PY:
                return pmaj, pmin, c
    return 0, 0, None


def get_macos_version() -> tuple[int, int]:
    ver = platform.mac_ver()[0]
    parts = ver.split(".")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


# ─── Step 1: Prerequisites ──────────────────────────────────────────────────

def check_prerequisites():
    header("Step 1/10 - Checking prerequisites")

    if platform.system() != "Darwin":
        fail("Muneem requires macOS. Detected: " + platform.system())
    info("macOS detected.")

    major, minor = get_macos_version()
    if major < MIN_MACOS_MAJOR:
        fail(f"Muneem requires macOS {MIN_MACOS_MAJOR}+ (Sequoia or later) for Core Audio Taps. Detected: {major}.{minor}")
    info(f"macOS {major}.{minor} - meets minimum requirement.")

    arch = platform.machine()
    if arch != "arm64":
        fail(f"Muneem requires Apple Silicon (arm64). Detected: {arch}")
    info("Apple Silicon (arm64) detected.")

    # Front-load Python availability check. WhisperX/ctranslate2/torch ship wheels
    # for CPython 3.10 - 3.13; anything older/newer is unsupported. Fail here before
    # Homebrew + Xcode + native compile run so users hit the actionable error in
    # seconds instead of after a long install.
    _py_found_major, _py_found_minor, _py_found_path = _find_supported_python()
    if not _py_found_path:
        fail(
            f"Python {SUPPORTED_PY_STR} required (ctranslate2/torch wheel availability).\n"
            "  Install one of:\n"
            "    brew install python@3.12   (recommended - longest track record)\n"
            "    brew install python@3.11\n"
            "    brew install python@3.13   (works; newest)\n"
            "    brew install python@3.10   (works; oldest supported)\n"
            "  Then re-run this installer."
        )
    if (_py_found_major, _py_found_minor) in RECOMMENDED_PY:
        info(f"Python {_py_found_major}.{_py_found_minor} available at {_py_found_path} (recommended)")
    else:
        warn(
            f"Python {_py_found_major}.{_py_found_minor} is supported but untested with "
            f"Muneem. If you hit wheel errors, fall back to 3.12: brew install python@3.12"
        )
        info(f"Python {_py_found_major}.{_py_found_minor} available at {_py_found_path}")

    if not is_installed("brew"):
        warn("Homebrew not found. Installing...")
        run('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")
        if not is_installed("brew"):
            fail("Homebrew installation failed. Install manually: https://brew.sh")
    info("Homebrew is available.")

    r = run("xcode-select -p", check=False, capture=True)
    if r.returncode != 0:
        print("  Installing Xcode Command Line Tools...")
        run("xcode-select --install", check=False)
        fail("Xcode CLT installation started. Re-run this script after it completes.")
    info("Xcode Command Line Tools available.")


# ─── Step 2: Brew packages ──────────────────────────────────────────────────

def install_brew_packages():
    header("Step 2/10 - Installing Homebrew packages")

    installed = run("brew list --formula -1", capture=True).stdout.splitlines()
    installed_casks = run("brew list --cask -1", capture=True).stdout.splitlines()

    for pkg in BREW_PACKAGES:
        if pkg in installed:
            info(f"{pkg} already installed.")
        else:
            print(f"  Installing {pkg}...")
            run(f"brew install {pkg}")
            info(f"{pkg} installed.")

    for cask in BREW_CASKS:
        if cask in installed_casks:
            info(f"{cask} already installed (fallback when Core Audio Tap unavailable).")
        else:
            print(f"  Installing {cask} (fallback when Core Audio Tap unavailable)...")
            run(f"brew install --cask {cask}")
            info(f"{cask} installed.")


# ─── Step 3: Project directories ────────────────────────────────────────────

def create_directories():
    header("Step 3/10 - Creating project directories")

    for d in [MUNEEM_HOME, BIN_DIR, NOTES_DIR, TMP_DIR, NATIVE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        info(f"{d}")


# ─── Step 4: Compile native Core Audio helper ───────────────────────────────

SWIFT_AUDIO_HELPER = textwrap.dedent('''\
    import Foundation
    import CoreAudio
    import AVFoundation
    import Darwin
    import Dispatch

    let args = CommandLine.arguments
    guard args.count >= 3 else {
        fputs("Usage: muneem-audio <system|mic> <output.wav> [seconds]\\n", stderr)
        exit(1)
    }

    let mode = args[1]
    let outputPath = args[2]
    let maxSeconds: Double = args.count > 3 ? Double(args[3]) ?? 0 : 0

    let sampleRate: Double = 16000
    let channels: UInt32 = 1

    func getDefaultInputDevice() -> AudioDeviceID {
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID)
        return deviceID
    }

    // BlackHole detection removed - Python-level fallback handles BlackHole via
    // sox (CoreAudio driver) with the device name. The Swift helper only does Core Audio Tap.

    // Per-process tap support (macOS 14.2+). Resolves OS PIDs to the
    // AudioObjectIDs that CATapDescription(stereoMixdownOfProcesses:) expects.
    // Called when MUNEEM_TAP_PIDS="1234,5678" is set - lets muneem tap a
    // SPECIFIC process (e.g. Arc running a Zoom web call) regardless of what
    // the system default output is. Global output taps fail on Bluetooth
    // outputs (known macOS limitation: Bluetooth-encoded streams bypass the
    // tap infrastructure); per-process taps intercept the app's audio BEFORE
    // it reaches the output device, so they work regardless of output.
    @available(macOS 14.2, *)
    func audioProcessObjects(forPIDs pids: [pid_t]) -> [AudioObjectID] {
        guard !pids.isEmpty else { return [] }
        var size: UInt32 = 0
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyProcessObjectList,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        guard AudioObjectGetPropertyDataSize(
                AudioObjectID(kAudioObjectSystemObject),
                &addr, 0, nil, &size) == noErr, size > 0 else {
            return []
        }
        let count = Int(size) / MemoryLayout<AudioObjectID>.size
        guard count > 0 else { return [] }
        var procs = [AudioObjectID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject),
                &addr, 0, nil, &size, &procs) == noErr else {
            return []
        }
        let pidSet = Set(pids)
        var out: [AudioObjectID] = []
        for p in procs {
            var pid: pid_t = 0
            var s = UInt32(MemoryLayout<pid_t>.size)
            var pidAddr = AudioObjectPropertyAddress(
                mSelector: kAudioProcessPropertyPID,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            if AudioObjectGetPropertyData(p, &pidAddr, 0, nil, &s, &pid) == noErr,
               pidSet.contains(pid) {
                out.append(p)
            }
        }
        return out
    }

    // Parse MUNEEM_TAP_PIDS="1234,5678,9012" env var into pid_t list.
    func parseTapPIDsFromEnv() -> [pid_t] {
        guard let env = ProcessInfo.processInfo.environment["MUNEEM_TAP_PIDS"],
              !env.isEmpty else { return [] }
        return env.split(separator: ",").compactMap {
            pid_t($0.trimmingCharacters(in: .whitespaces))
        }
    }

    // Resolve the default output device's UID. The aggregate device that wraps the
    // process tap uses this as its main sub-device so clock domain and driver model
    // are well-defined (tap alone has no clock).
    func defaultOutputDeviceUID() -> String? {
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &deviceID) == noErr else {
            return nil
        }
        var uid: CFString = "" as CFString
        var uidSize = UInt32(MemoryLayout<CFString>.size)
        var uidAddr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let status = withUnsafeMutablePointer(to: &uid) {
            AudioObjectGetPropertyData(deviceID, &uidAddr, 0, nil, &uidSize, $0)
        }
        guard status == noErr else { return nil }
        return uid as String
    }

    class Recorder {
        var file: AVAudioFile?
        let engine = AVAudioEngine()

        // Core Audio Tap resources (system mode). Released in cleanupAll().
        var tapID: AudioObjectID = AudioObjectID(kAudioObjectUnknown)
        var aggregateID: AudioDeviceID = 0
        var ioProcID: AudioDeviceIOProcID? = nil
        var wavOutputFormat: AVAudioFormat? = nil
        var ioFireCount: Int = 0
        var decimateN: Int = 3  // ratio aggregateRate/16000; set at tap start
        let writeQueue = DispatchQueue(label: "com.muneem.audio.write")

        func recordMic(to path: String, seconds: Double) {
            let input = engine.inputNode
            // Use the input node's NATIVE hardware format for the tap - passing a
            // mismatched format (e.g. 16kHz/Float32 when hardware is 48kHz) throws
            // 'Failed to create tap due to format mismatch'. Convert to the WAV
            // format inside the tap callback instead.
            let hwFormat = input.inputFormat(forBus: 0)
            let wavFormat = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: sampleRate,
                channels: AVAudioChannelCount(channels),
                interleaved: true
            )!

            // Open the output file with EXPLICIT processing format matching the
            // buffers we'll hand to it (same trick as recordSystem). Without this,
            // AVAudioFile routes writes through an internal ExtAudioFile converter
            // that can trap at runtime with a CoreAudio assertion.
            do {
                file = try AVAudioFile(
                    forWriting: URL(fileURLWithPath: path),
                    settings: wavFormat.settings,
                    commonFormat: wavFormat.commonFormat,
                    interleaved: wavFormat.isInterleaved
                )
            } catch {
                fputs("[muneem-audio] Failed to create audio file: \\(error)\\n", stderr)
                exit(3)
            }

            let micConverter = AVAudioConverter(from: hwFormat, to: wavFormat)
            guard let micConv = micConverter else {
                fputs("[muneem-audio] Could not build mic AVAudioConverter.\\n", stderr)
                exit(4)
            }

            input.installTap(onBus: 0, bufferSize: 4096, format: hwFormat) { [weak self] inBuf, _ in
                guard let self = self else { return }
                let frames = inBuf.frameLength
                guard frames > 0 else { return }
                let cap = AVAudioFrameCount(Double(frames) * wavFormat.sampleRate / hwFormat.sampleRate) + 1024
                guard let outBuf = AVAudioPCMBuffer(pcmFormat: wavFormat, frameCapacity: cap) else { return }
                var supplied = false
                var err: NSError? = nil
                let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                    if supplied { outStatus.pointee = .noDataNow; return nil }
                    supplied = true
                    outStatus.pointee = .haveData
                    return inBuf
                }
                micConv.convert(to: outBuf, error: &err, withInputFrom: inputBlock)
                if err != nil { return }
                do { try self.file?.write(from: outBuf) }
                catch { fputs("[muneem-audio] Failed to write audio: \\(error)\\n", stderr) }
            }

            do { try engine.start() }
            catch {
                fputs("[muneem-audio] Failed to start audio engine: \\(error)\\n", stderr)
                exit(4)
            }

            if seconds > 0 {
                Thread.sleep(forTimeInterval: seconds)
                stop()
            } else {
                signal(SIGINT) { _ in exit(0) }
                dispatchMain()
            }
        }

        // System audio via Core Audio Process Tap (macOS 14.2+).
        //
        // Flow:
        //   1. Build CATapDescription (global mixdown, no exclusions)
        //   2. AudioHardwareCreateProcessTap → tapID
        //   3. Query tap's native ASBD (kAudioTapPropertyFormat)
        //   4. AudioHardwareCreateAggregateDevice wrapping the tap + default output
        //   5. AVAudioConverter from tap format → 16 kHz mono Int16 (WhisperX input)
        //   6. AudioDeviceCreateIOProcIDWithBlock: in the callback, convert and
        //      hand writes off to a serial dispatch queue to keep the RT thread fast
        //   7. AudioDeviceStart
        //   8. On stop: AudioDeviceStop / DestroyIOProcID / DestroyAggregate / DestroyTap
        //
        // Caveat: an unsigned binary without entitlements may get a TCC denial.
        // The first run will prompt for microphone access for the parent terminal;
        // if denied, the call fails cleanly and exit code 2 triggers the sox fallback.
        func recordSystem(to path: String, seconds: Double) {
            guard #available(macOS 14.2, *) else {
                fputs("[muneem-audio] Core Audio Tap requires macOS 14.2+. Exit 2 for Python fallback.\\n", stderr)
                exit(2)
            }
            guard let outputUID = defaultOutputDeviceUID() else {
                fputs("[muneem-audio] Could not resolve default output device. Exit 2.\\n", stderr)
                exit(2)
            }
            fputs("[muneem-audio] Attempting Core Audio Tap for system audio (output UID: \\(outputUID))\\n", stderr)

            // 1. Pick tap description.
            //    If MUNEEM_TAP_PIDS is set, tap JUST those processes (per-app
            //    tap that works even on Bluetooth output or virtual/stacked
            //    default outputs). Otherwise, a system-wide tap (captures
            //    everything routed through the default output - fails on
            //    Bluetooth in practice).
            let envPIDs = parseTapPIDsFromEnv()
            let procObjIDs = audioProcessObjects(forPIDs: envPIDs)
            let desc: CATapDescription
            if !envPIDs.isEmpty && !procObjIDs.isEmpty {
                fputs("[muneem-audio] Per-process tap on PIDs: \\(envPIDs) (resolved \\(procObjIDs.count) audio process object(s))\\n", stderr)
                desc = CATapDescription(stereoMixdownOfProcesses: procObjIDs)
            } else {
                if !envPIDs.isEmpty {
                    fputs("[muneem-audio] MUNEEM_TAP_PIDS=\\(envPIDs) but no matching audio process objects found. Falling back to global tap.\\n", stderr)
                }
                desc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
            }
            desc.uuid = UUID()
            desc.muteBehavior = .unmuted
            // Global tap needs isPrivate=true so it doesn't show up in Audio
            // MIDI Setup. Per-process tap follows AudioCap's pattern (no
            // isPrivate, no isExclusive): with those flags set on a
            // stereoMixdownOfProcesses description, IOProc fires on schedule
            // but every buffer is silence. Empirically verified on macOS 26.
            if envPIDs.isEmpty || procObjIDs.isEmpty {
                desc.isPrivate = true
                desc.isExclusive = false
            }

            // 2. Create the tap.
            var tap = AudioObjectID(kAudioObjectUnknown)
            let tapStatus = AudioHardwareCreateProcessTap(desc, &tap)
            guard tapStatus == noErr, tap != kAudioObjectUnknown else {
                fputs("[muneem-audio] AudioHardwareCreateProcessTap failed (status \\(tapStatus)). Exit 2 for Python fallback.\\n", stderr)
                exit(2)
            }
            tapID = tap

            // 3. Query tap's native stream format.
            var tapASBD = AudioStreamBasicDescription()
            var asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
            var asbdAddr = AudioObjectPropertyAddress(
                mSelector: kAudioTapPropertyFormat,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            let asbdStatus = AudioObjectGetPropertyData(tap, &asbdAddr, 0, nil, &asbdSize, &tapASBD)
            guard asbdStatus == noErr else {
                fputs("[muneem-audio] kAudioTapPropertyFormat query failed (status \\(asbdStatus)). Exit 2.\\n", stderr)
                cleanupTap()
                exit(2)
            }

            // 4. Aggregate device wrapping the tap.
            //
            // Aggregate device rules (empirically determined on macOS 15/26):
            //
            //   GLOBAL tap (stereoGlobalTapButExcludeProcesses):
            //     - Tap alone has a clock - no sub-devices needed.
            //     - MUST NOT include kAudioAggregateDeviceMainSubDeviceKey or
            //       kAudioAggregateDeviceTapAutoStartKey (blocks IOProc).
            //
            //   PER-PROCESS tap (stereoMixdownOfProcesses):
            //     - Tap alone does NOT drive IOProc - buffers arrive silent.
            //     - MUST include the default output as a sub-device and set
            //       kAudioAggregateDeviceTapAutoStartKey so the aggregate has
            //       a real clock domain and actually captures the tap stream.
            //     - Matches the insidegui/AudioCap reference implementation.
            let aggUID = "com.muneem.tap.agg.\\(UUID().uuidString)"
            let tapUIDStr = desc.uuid.uuidString
            let isPerProcess = !envPIDs.isEmpty && !procObjIDs.isEmpty
            var aggDesc: [String: Any] = [
                kAudioAggregateDeviceUIDKey: aggUID,
                kAudioAggregateDeviceNameKey: "Muneem Tap Aggregate",
                kAudioAggregateDeviceIsPrivateKey: 1,
                kAudioAggregateDeviceIsStackedKey: 0,
                kAudioAggregateDeviceTapListKey: [
                    [
                        kAudioSubTapUIDKey: tapUIDStr,
                        kAudioSubTapDriftCompensationKey: 1,
                    ]
                ],
            ]
            if isPerProcess {
                aggDesc[kAudioAggregateDeviceSubDeviceListKey] = [
                    [kAudioSubDeviceUIDKey: outputUID],
                ]
                aggDesc[kAudioAggregateDeviceMainSubDeviceKey] = outputUID
                aggDesc[kAudioAggregateDeviceTapAutoStartKey] = 1
            }
            var agg: AudioDeviceID = 0
            let aggStatus = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &agg)
            guard aggStatus == noErr, agg != 0 else {
                fputs("[muneem-audio] AudioHardwareCreateAggregateDevice failed (status \\(aggStatus)). Exit 2.\\n", stderr)
                cleanupTap()
                exit(2)
            }
            aggregateID = agg

            // Read the aggregate's actual running sample rate. For per-
            // process taps, the aggregate inherits the MAIN sub-device's
            // rate (MacBook Pro Speakers default to 96kHz) and Core Audio
            // silently upsamples tap buffers to match - so the IOProc sees
            // data at the aggregate rate, not the tap's native 48k.
            // Decimation ratio must therefore be computed dynamically.
            var aggSampleRate: Float64 = tapASBD.mSampleRate
            var aggRateAddr = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyNominalSampleRate,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var aggRateSize = UInt32(MemoryLayout<Float64>.size)
            _ = AudioObjectGetPropertyData(agg, &aggRateAddr, 0, nil, &aggRateSize, &aggSampleRate)
            if aggSampleRate <= 0 { aggSampleRate = tapASBD.mSampleRate }
            self.decimateN = max(1, Int((aggSampleRate / sampleRate).rounded()))
            fputs("[muneem-audio] Aggregate rate: \\(aggSampleRate)Hz, tap rate: \\(tapASBD.mSampleRate)Hz, decimate-by: \\(self.decimateN) -> \\(sampleRate)Hz.\\n", stderr)

            // 5. Output format: 16 kHz mono Int16 (WhisperX expects this).
            //
            // Tap always emits 48 kHz interleaved stereo Float32 (verified via
            // kAudioTapPropertyFormat). 48000/16000 = 3 exactly, so we can do a
            // trivial decimate-by-3 + L/R downmix inline in the IOProc. Using
            // AVAudioConverter + AVAudioFile's internal converter was crashing
            // inside ExtAudioFile::WriteInputProc due to a format-chain mismatch
            // - the manual path is simpler, faster, and crash-proof.
            let wav = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: sampleRate,
                channels: AVAudioChannelCount(channels),
                interleaved: true
            )!
            wavOutputFormat = wav

            // 6. Open output file with EXPLICIT processing format so AVAudioFile
            // writes bytes directly without running an internal converter.
            do {
                file = try AVAudioFile(
                    forWriting: URL(fileURLWithPath: path),
                    settings: wav.settings,
                    commonFormat: wav.commonFormat,
                    interleaved: wav.isInterleaved
                )
            } catch {
                fputs("[muneem-audio] Failed to create audio file: \\(error)\\n", stderr)
                cleanupAll()
                exit(3)
            }

            // 7. Install IOProc. Callback runs on a real-time audio thread; write
            // is dispatched to a serial queue so the RT thread stays fast.
            var procID: AudioDeviceIOProcID? = nil
            // When MUNEEM_TAP_DEBUG=1, log source-buffer peaks so we can tell
            // whether the tap is silently dropping audio (e.g. TCC denial)
            // versus a downstream decimation bug.
            let debug = ProcessInfo.processInfo.environment["MUNEEM_TAP_DEBUG"] == "1"
            let createStatus = AudioDeviceCreateIOProcIDWithBlock(&procID, agg, nil) { [weak self] (_, inInputData, _, _, _) in
                guard let self = self, let outFmt = self.wavOutputFormat else { return }
                let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
                guard abl.count > 0, let srcPtr = abl[0].mData else { return }
                // 8 bytes/frame = 2 channels * 4 bytes (Float32 interleaved).
                let srcFrames = Int(abl[0].mDataByteSize) / 8
                guard srcFrames > 0 else { return }
                let srcFloat = srcPtr.assumingMemoryBound(to: Float.self)
                if debug {
                    var peak: Float = 0
                    let probe = min(srcFrames * 2, 128)
                    for k in 0..<probe { let a = abs(srcFloat[k]); if a > peak { peak = a } }
                    self.ioFireCount &+= 1
                    if self.ioFireCount <= 5 || self.ioFireCount % 50 == 0 {
                        fputs("[muneem-audio] IOProc fire #\\(self.ioFireCount): srcFrames=\\(srcFrames) peak=\\(peak)\\n", stderr)
                    }
                }

                // Decimate src-rate -> 16k by picking every Nth frame; downmix L+R -> mono.
                // N is computed at tap start from the aggregate's running rate
                // (3 for 48k, 6 for 96k), not hardcoded.
                let decimateN = max(1, self.decimateN)
                let outFrames = srcFrames / decimateN
                guard outFrames > 0,
                      let outBuf = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: AVAudioFrameCount(outFrames)),
                      let dst = outBuf.int16ChannelData?[0]
                else { return }
                outBuf.frameLength = AVAudioFrameCount(outFrames)
                let stride = decimateN * 2  // 2 samples per frame (interleaved L,R)
                for i in 0..<outFrames {
                    let srcIdx = i * stride
                    let l = srcFloat[srcIdx]
                    let r = srcFloat[srcIdx + 1]
                    var mono = (l + r) * 0.5
                    if mono >  1.0 { mono =  1.0 }
                    if mono < -1.0 { mono = -1.0 }
                    dst[i] = Int16(mono * 32767)
                }

                self.writeQueue.async { [weak self] in
                    guard let self = self, let outFile = self.file else { return }
                    do { try outFile.write(from: outBuf) }
                    catch { fputs("[muneem-audio] write error: \\(error)\\n", stderr) }
                }
            }
            guard createStatus == noErr, let proc = procID else {
                fputs("[muneem-audio] AudioDeviceCreateIOProcIDWithBlock failed (status \\(createStatus)). Exit 2.\\n", stderr)
                cleanupAll()
                exit(2)
            }
            ioProcID = proc

            let startStatus = AudioDeviceStart(agg, proc)
            guard startStatus == noErr else {
                fputs("[muneem-audio] AudioDeviceStart failed (status \\(startStatus)). Exit 2.\\n", stderr)
                cleanupAll()
                exit(2)
            }
            fputs("[muneem-audio] Core Audio Tap capture started (sampleRate=\\(tapASBD.mSampleRate), channels=\\(tapASBD.mChannelsPerFrame)).\\n", stderr)

            if seconds > 0 {
                Thread.sleep(forTimeInterval: seconds)
                cleanupAll()
            } else {
                signal(SIGINT) { _ in exit(0) }
                dispatchMain()
            }
        }

        func cleanupTap() {
            if tapID != kAudioObjectUnknown {
                if #available(macOS 14.2, *) {
                    AudioHardwareDestroyProcessTap(tapID)
                }
                tapID = kAudioObjectUnknown
            }
        }

        func cleanupAll() {
            if let proc = ioProcID, aggregateID != 0 {
                AudioDeviceStop(aggregateID, proc)
                AudioDeviceDestroyIOProcID(aggregateID, proc)
                ioProcID = nil
            }
            // Drain any queued writes before closing the file.
            writeQueue.sync { }
            if aggregateID != 0 {
                AudioHardwareDestroyAggregateDevice(aggregateID)
                aggregateID = 0
            }
            cleanupTap()
            wavOutputFormat = nil
            file = nil
        }

        func stop() {
            cleanupAll()
            engine.stop()
            engine.inputNode.removeTap(onBus: 0)
        }
    }

    let recorder = Recorder()
    switch mode {
    case "mic":
        recorder.recordMic(to: outputPath, seconds: maxSeconds)
    case "system":
        recorder.recordSystem(to: outputPath, seconds: maxSeconds)
    default:
        fputs("Unknown mode: \\(mode). Use 'system' or 'mic'.\\n", stderr)
        exit(1)
    }
''')


_MUNEEM_AUDIO_INFO_PLIST = '''\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.muneem.audio</string>
    <key>CFBundleName</key>
    <string>Muneem Audio Capture</string>
    <key>CFBundleDisplayName</key>
    <string>Muneem Audio</string>
    <key>CFBundleExecutable</key>
    <string>muneem-audio</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.2</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSAudioCaptureUsageDescription</key>
    <string>Muneem captures system audio to transcribe meetings offline on your Mac.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>Muneem records your microphone to transcribe meetings offline on your Mac.</string>
</dict>
</plist>
'''


def compile_native_helper():
    """Compile the Core Audio Tap helper and package it into a .app bundle.

    Why the bundle: macOS 14.4+ gates process taps behind a new TCC service
    (kTCCServiceAudioCapture). TCC silently returns zero-filled buffers when
    the RESPONSIBLE process (the shell/terminal invoking muneem) lacks the
    NSAudioCaptureUsageDescription Info.plist key - which is the case for
    Terminal.app, iTerm2, and every other standard terminal. Packaging the
    helper as an .app bundle and launching it via `open -W -n -a` makes
    launchd the parent; TCC then attributes the request to the bundle's
    Info.plist (which DOES declare the key) so permission can actually be
    granted and audio buffers contain real samples.
    """
    header("Step 4/10 - Compiling Core Audio Tap helper (.app bundle)")

    swift_src = NATIVE_DIR / "muneem_audio.swift"
    binary = NATIVE_DIR / "muneem-audio"
    app_bundle = NATIVE_DIR / "muneem-audio.app"
    app_macos = app_bundle / "Contents" / "MacOS"
    app_plist = app_bundle / "Contents" / "Info.plist"

    swift_src.write_text(SWIFT_AUDIO_HELPER)
    info(f"Wrote Swift source to {swift_src}")

    print("  Compiling muneem-audio (Swift)...")
    r = subprocess.run(
        ["swiftc", "-O", "-o", str(binary), str(swift_src),
         "-framework", "Foundation", "-framework", "CoreAudio", "-framework", "AVFoundation"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        warn(f"Swift compilation failed. BlackHole will be used as fallback.\n  {r.stderr.strip()}")
        return False

    binary.chmod(0o755)
    info(f"Compiled {binary}")

    # Build the app bundle around the compiled binary.
    try:
        if app_bundle.exists():
            import shutil as _shutil
            _shutil.rmtree(app_bundle)
        app_macos.mkdir(parents=True, exist_ok=True)
        (app_bundle / "Contents" / "Resources").mkdir(parents=True, exist_ok=True)
        app_plist.write_text(_MUNEEM_AUDIO_INFO_PLIST)
        # Copy (not symlink) - TCC checks the binary path's bundle ancestry.
        import shutil as _shutil
        _shutil.copy2(str(binary), str(app_macos / "muneem-audio"))
        (app_macos / "muneem-audio").chmod(0o755)

        # Ad-hoc sign the bundle. Without a signature, TCC treats the bundle
        # as an ad-hoc anonymous executable and will not honor the Info.plist.
        sr = subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", str(app_bundle)],
            capture_output=True, text=True
        )
        if sr.returncode != 0:
            warn(f"codesign failed: {sr.stderr.strip()}")
            # Keep going - tap may still work if user's system allows unsigned
        else:
            info(f"Signed bundle: {app_bundle}")
    except Exception as e:
        warn(f"Could not assemble .app bundle: {e}. BlackHole will be used as fallback.")
        return False

    # Quick sanity check on the bundled binary (not a functional tap test -
    # that needs a real audio source and the TCC prompt flow).
    try:
        vr = subprocess.run(
            [str(app_macos / "muneem-audio"), "system", "/tmp/test_audio.wav", "0"],
            capture_output=True, text=True, timeout=2
        )
        info(f"Core Audio Tap binary verified (exit code {vr.returncode}).")
    except subprocess.TimeoutExpired:
        info("Core Audio Tap binary verified (timeout = waiting for audio).")
    except Exception as e:
        warn(f"Core Audio Tap verification encountered: {e}. Will use fallback.")
        return False

    info("Core Audio Tap configured as default system audio backend.")
    info(f"App bundle: {app_bundle} (launched via `open -W -n -a` for TCC)")
    return True


# ─── Step 5: System resources ────────────────────────────────────────────────

def _detect_ram_gb() -> int:
    """Total physical RAM in GiB via sysctl. 0 if detection fails."""
    try:
        r = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return int(r.stdout.strip()) // (1024 ** 3)
    except Exception:
        return 0


def _detect_chip() -> str:
    """Apple Silicon chip descriptor from sysctl brand string."""
    try:
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return (r.stdout.strip() or "Unknown")
    except Exception:
        return "Unknown"


def check_system_resources() -> dict:
    """Detect RAM + chip, enforce the 16 GB minimum, warn below 24 GB.

    Returns {'ram_gb': int, 'chip': str} for callers (select_llm uses this).
    On Apple Silicon, GPU memory is unified with system RAM, so hw.memsize
    alone is the right signal for model-size gating.
    """
    header("Step 5/10 - System resource check")
    ram_gb = _detect_ram_gb()
    chip = _detect_chip()
    info(f"Chip: {chip}")
    info(f"RAM:  {ram_gb} GB (Apple Silicon unified memory - GPU shares this pool)")
    if ram_gb == 0:
        warn("Could not detect RAM via sysctl. Proceeding without gating.")
    elif ram_gb < LLM_MIN_RAM_GB:
        fail(
            f"Muneem requires at least {LLM_MIN_RAM_GB} GB RAM (detected {ram_gb} GB).\n"
            f"  The transcription model + notes LLM + vision model won't fit otherwise."
        )
    elif ram_gb < 24:
        warn(
            f"Only {ram_gb} GB detected. 24+ GB recommended for smooth operation; "
            f"you may see sluggish responses when all three models are loaded."
        )
    return {"ram_gb": ram_gb, "chip": chip}


# ─── Step 6: LLM selection ───────────────────────────────────────────────────

def select_llm(resources: dict) -> str:
    """Interactive picker for the notes LLM. Writes ~/.muneem/config.json.

    Reproducibility policy:
      - Whisper model and vision model are FIXED for every install.
      - Only the notes LLM varies - between 14B (default) and 32B (if 48+ GB RAM).
      - Default is always 14B so machines below the 48 GB threshold get a
        predictable, identical experience.

    Re-runs of the installer respect the previously chosen model unless the
    machine can no longer support it (user downgraded, or ran on a new box).
    """
    import json as _json
    header("Step 6/10 - Notes LLM selection")

    ram_gb = resources.get("ram_gb", 0)
    can_use_high = ram_gb >= LLM_HIGH_TIER_MIN_RAM_GB

    config_path = MUNEEM_HOME / "config.json"
    prior = None
    if config_path.exists():
        try:
            prior = _json.loads(config_path.read_text()).get("llm_model")
        except Exception:
            prior = None
    if prior == LLM_HIGH_TIER and not can_use_high:
        warn(f"Previous choice {prior} needs {LLM_HIGH_TIER_MIN_RAM_GB}+ GB RAM; "
             f"this machine has {ram_gb} GB. Falling back to {LLM_DEFAULT}.")
        prior = None

    default_choice = "2" if (prior == LLM_HIGH_TIER and can_use_high) else "1"

    print()
    marker = " [previously selected]" if prior == LLM_DEFAULT else ""
    print(f"  [1] {LLM_DEFAULT:<12}  (~9.3 GB disk, ~12 GB in use)  default, fast, reliable{marker}")
    if can_use_high:
        marker = " [previously selected]" if prior == LLM_HIGH_TIER else ""
        print(f"  [2] {LLM_HIGH_TIER:<12}  (~20 GB disk, ~22 GB in use)  higher quality, ~2x slower{marker}")
    else:
        print(f"  [2] {LLM_HIGH_TIER:<12}  requires {LLM_HIGH_TIER_MIN_RAM_GB}+ GB RAM - UNAVAILABLE on this machine")
    print()

    if not sys.stdin.isatty():
        chosen = prior or LLM_DEFAULT
        info(f"Non-interactive install. Using {chosen}.")
    else:
        try:
            raw = input(f"  Your choice [{default_choice}]: ").strip() or default_choice
        except (EOFError, KeyboardInterrupt):
            raw = default_choice
        if raw == "2" and can_use_high:
            chosen = LLM_HIGH_TIER
        elif raw == "2" and not can_use_high:
            warn(f"{LLM_HIGH_TIER} unavailable on this machine. Using {LLM_DEFAULT}.")
            chosen = LLM_DEFAULT
        else:
            chosen = LLM_DEFAULT

    config = {
        "llm_model": chosen,
        "vision_model": VISION_OLLAMA_MODEL,
        "whisper_model": "large-v3",
        "ram_gb": ram_gb,
        "chip": resources.get("chip", "Unknown"),
    }
    MUNEEM_HOME.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_json.dumps(config, indent=2) + "\n")
    info(f"Selected LLM: {chosen}")
    info(f"Config written: {config_path}")
    return chosen


# ─── Step 7: Python venv + pip ───────────────────────────────────────────────

def setup_python_env():
    header(f"Step 7/10 - Setting up Python environment")

    major, minor, python = _find_supported_python()
    if not python:
        fail(
            f"Python {SUPPORTED_PY_STR} required (ctranslate2/torch wheel availability).\n"
            f"  Install: brew install python@3.12"
        )

    if (major, minor) in RECOMMENDED_PY:
        info(f"Using Python {major}.{minor} - recommended for WhisperX.")
    else:
        info(f"Using Python {major}.{minor} - supported (untested with Muneem).")

    if not VENV_DIR.exists():
        print(f"  Creating virtual environment at {VENV_DIR} with Python {major}.{minor}...")
        run(f'"{python}" -m venv "{VENV_DIR}"')
        info(f"Virtual environment created (Python {major}.{minor}).")
    else:
        venv_py = VENV_DIR / "bin" / "python"
        recreate = False
        reuse_reason = ""
        if venv_py.exists():
            r = run(
                f'"{venv_py}" -c "import sys; print(sys.version_info.major, sys.version_info.minor)"',
                check=False, capture=True,
            )
            if r.returncode == 0:
                try:
                    vmaj, vmin = (int(x) for x in r.stdout.strip().split())
                except (ValueError, IndexError):
                    vmaj, vmin = 0, 0
                if (vmaj, vmin) not in SUPPORTED_PY:
                    warn(f"Existing venv uses Python {vmaj}.{vmin} (unsupported). Recreating with {major}.{minor}...")
                    recreate = True
                elif (vmaj, vmin) == (major, minor):
                    # Selected python matches venv exactly. Reuse as-is (big time saver:
                    # avoids re-downloading torch + whisperx + pyobjc = hundreds of MB).
                    reuse_reason = f"matches selected Python {major}.{minor}"
                else:
                    # Compatible but different minor (e.g. venv=3.11, selected=3.12).
                    # Both in SUPPORTED_PY → reuse; no reason to burn bandwidth.
                    reuse_reason = (
                        f"uses compatible Python {vmaj}.{vmin} "
                        f"(selected {major}.{minor}; keeping existing to avoid re-download)"
                    )
            else:
                warn("Existing venv python is broken. Recreating...")
                recreate = True
        else:
            warn("Venv exists but python binary is missing. Recreating...")
            recreate = True
        if recreate:
            shutil.rmtree(VENV_DIR, ignore_errors=True)
            run(f'"{python}" -m venv "{VENV_DIR}"')
            info(f"Virtual environment recreated (Python {major}.{minor}).")
        else:
            info(f"Reusing existing venv ({reuse_reason}).")
    info("Virtual environment ready.")

    pip = VENV_DIR / "bin" / "pip"
    print("  Installing Python packages (this may take several minutes on first run)...")
    run(f'"{pip}" install --upgrade pip --quiet')
    quoted = " ".join(f'"{pkg}"' for pkg in PIP_PACKAGES)
    run(f'"{pip}" install --upgrade {quoted} --quiet')
    info("All Python packages installed.")

    # ── Exorcise PyAudio ────────────────────────────────────────────────────
    # PyAudio's PortAudio backend crashes in PaMacCore_Initialize on macOS 26
    # Tahoe (null fn ptr in PaMacCore_AudioDeviceGetProperty). Muneem never
    # uses pyaudio directly - audio capture is sox + the Swift native helper.
    #
    # BUT: pyannote.audio (pulled transitively by whisperx) does an optional
    # `import pyaudio` at load time and, if present, wires it into its audio
    # I/O path. Right after that, any call through whisperx (including our
    # alignment-model pre-cache a few lines below) crashes in pyaudio.
    #
    # PyAudio can also linger in the venv as a leftover from older installs
    # that had it in PIP_PACKAGES. Running the uninstall unconditionally is
    # cheap when absent and idempotent when present - and it's the only
    # thing that reliably keeps pyannote from binding to it.
    rp = run(f'"{pip}" uninstall -y PyAudio --quiet', check=False, capture=True)
    if rp.returncode == 0 and "Successfully uninstalled" in (rp.stdout or ""):
        info("Removed leftover PyAudio (crashes on macOS 26 Tahoe; muneem uses sox).")
    # Second invocation in case `pyaudio` is registered lowercase somewhere.
    run(f'"{pip}" uninstall -y pyaudio --quiet', check=False)

    # Pre-cache all WhisperX models so first `muneem start` is fully offline.
    # SSL certs are resolved dynamically via certifi (works for python3.11 and python3.12).
    venv_python = VENV_DIR / "bin" / "python"
    # Resolve certifi path dynamically - works regardless of Python minor version.
    r_cert = run(f'"{venv_python}" -c "import certifi; print(certifi.where())"',
                 check=False, capture=True)
    certifi_pem = r_cert.stdout.strip() if r_cert.returncode == 0 else ""
    ssl_env = (f'SSL_CERT_FILE="{certifi_pem}" REQUESTS_CA_BUNDLE="{certifi_pem}"'
               if certifi_pem else "")

    # Guard string injected into every pre-cache subprocess - blocks any
    # accidental `import pyaudio` path inside whisperx/pyannote from triggering
    # the macOS 26 Tahoe PortAudio crash. Redundant with the uninstall above,
    # but free insurance if the venv ever ends up with pyaudio again (e.g. a
    # user's earlier `pip install` or a future transitive).
    pa_guard = 'import sys; sys.modules[\'pyaudio\'] = None; '

    # 1. WhisperX large-v3 speech model (~3 GB, from HuggingFace)
    print("  Pre-caching WhisperX large-v3 model (~3 GB from HuggingFace, one-time)...")
    r = run(
        f'{ssl_env} "{venv_python}" -c '
        f'"{pa_guard}import warnings; warnings.filterwarnings(\'ignore\'); '
        'import whisperx; whisperx.load_model(\'large-v3\', \'cpu\', compute_type=\'float32\')"',
        check=False, capture=True,
    )
    if r.returncode == 0:
        info("WhisperX large-v3 model cached.")
    else:
        warn("Could not pre-cache large-v3 model (will download on first run). "
             "Check internet connection if transcription fails.")

    # 2. wav2vec2 alignment model (~360 MB, from pytorch.org)
    print("  Pre-caching WhisperX alignment model (wav2vec2, ~360 MB, one-time)...")
    r = run(
        f'{ssl_env} "{venv_python}" -c '
        f'"{pa_guard}import warnings; warnings.filterwarnings(\'ignore\'); '
        'import whisperx; whisperx.load_align_model(language_code=\'en\', device=\'cpu\')"',
        check=False, capture=True,
    )
    if r.returncode == 0:
        info("WhisperX alignment model cached.")
    else:
        warn("Could not pre-cache alignment model (will download on first run). "
             "Check internet connection if transcription fails.")


# ─── Step 7b: Download diarization ONNX models ──────────────────────────────
# Fixes "Speaker 1 and Speaker 2 get swapped between utterances" by replacing
# the per-segment `diarize` package with sherpa-onnx's pyannote-segmentation
# (voice activity + speaker-turn detection) paired with 3D-Speaker's CAM++
# embeddings. Combined with a SpeakerRegistry (cosine-similarity matching of
# embedding centroids) this gives consistent speaker labels across the entire
# meeting - a single Speaker 1 from start to finish.
#
# Both models are Apache-2.0, served from sherpa-onnx's GitHub releases, no
# HuggingFace account required.
SHERPA_SEG_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SHERPA_EMB_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
)
SHERPA_MODELS_DIR_NAME = "sherpa-onnx-pyannote-segmentation-3-0"
SHERPA_EMB_FILENAME   = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"


def download_sherpa_diar_models():
    header("Step 7b/10 - Downloading diarization ONNX models")
    models_dir = MUNEEM_HOME / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    seg_dir    = models_dir / SHERPA_MODELS_DIR_NAME
    seg_model  = seg_dir / "model.onnx"
    emb_model  = models_dir / SHERPA_EMB_FILENAME

    if seg_model.exists() and emb_model.exists():
        info(f"Diarization models present in {models_dir} - skipping download.")
        return True

    import tarfile as _tarfile

    ok = True

    # Segmentation (pyannote-segmentation-3.0) - ~6 MB tarball
    if not seg_model.exists():
        tar_path = models_dir / "_pyannote-seg.tar.bz2"
        print(f"  Downloading pyannote-segmentation-3.0 (~6 MB)...")
        r = run(f'curl -fsSL --retry 3 -o "{tar_path}" "{SHERPA_SEG_URL}"',
                check=False, capture=True)
        if r.returncode != 0 or not tar_path.exists():
            warn("Could not download pyannote-segmentation-3.0. "
                 "Diarization will fall back to the `diarize` package "
                 "(per-segment clustering - less accurate). "
                 f"Manual fetch: {SHERPA_SEG_URL}")
            ok = False
        else:
            try:
                with _tarfile.open(str(tar_path), "r:bz2") as tf:
                    tf.extractall(str(models_dir))
                info(f"Extracted {seg_dir}.")
            except Exception as e:
                warn(f"Failed to extract segmentation tarball: {e}")
                ok = False
            try: tar_path.unlink()
            except OSError: pass

    # Speaker embeddings (CAM++ voxceleb English, ~28 MB)
    if not emb_model.exists():
        print(f"  Downloading 3D-Speaker CAM++ embeddings (~28 MB)...")
        r = run(f'curl -fsSL --retry 3 -o "{emb_model}" "{SHERPA_EMB_URL}"',
                check=False, capture=True)
        if r.returncode != 0 or not emb_model.exists() or emb_model.stat().st_size < 1_000_000:
            warn("Could not download 3D-Speaker CAM++ embeddings. "
                 "Diarization will fall back to the `diarize` package. "
                 f"Manual fetch: {SHERPA_EMB_URL}")
            try: emb_model.unlink()
            except OSError: pass
            ok = False
        else:
            info(f"Cached {emb_model.name} ({emb_model.stat().st_size // (1024*1024)} MB).")

    return ok


# ─── Step 8: Write application modules ──────────────────────────────────────

def write_app_modules():
    header("Step 8/10 - Writing application modules")
    modules = {
        "transcriber.py": _MODULE_TRANSCRIBER,
        "screen_reader.py": _MODULE_SCREEN_READER,
        "enhancer.py": _MODULE_ENHANCER,
        "app.py": _MODULE_APP,
    }
    for name, content in modules.items():
        path = MUNEEM_HOME / name
        path.write_text(content)
        info(f"Wrote {path}")

    # Copy the setup script itself into ~/.muneem/ so re-runs and uninstall
    # work without needing the original source directory.
    setup_src = Path(__file__).resolve()
    setup_dst = MUNEEM_HOME / "muneem-setup.py"
    shutil.copy2(str(setup_src), str(setup_dst))
    info(f"Copied installer to {setup_dst}")


# ─── Step 7: Ollama models ──────────────────────────────────────────────────

def _get_available_models() -> set[str]:
    import json as _json
    r = run("curl -sf http://localhost:11434/api/tags", check=False, capture=True)
    if r.returncode != 0:
        return set()
    try:
        data = _json.loads(r.stdout)
        return {m["name"] for m in data.get("models", [])}
    except Exception:
        return set()


def _model_already_pulled(model: str, available: set[str]) -> bool:
    """Exact match or exact-name-with-quantization-suffix (e.g. qwen3:14b vs qwen3:14b-q4_0).
    Avoids false positives like 'qwen3:8b' matching 'qwen3:18b' (substring containment)."""
    if model in available:
        return True
    for a in available:
        # Allow Ollama's quantization/variant suffixes: "<model>-<variant>"
        if a == model or a.startswith(model + "-"):
            return True
    return False


def pull_ollama_models(llm_model: str = None):
    header("Step 9/10 - Checking / pulling Ollama models")

    if llm_model is None:
        llm_model = LLM_DEFAULT  # backwards-compat if called without explicit choice
    models_to_pull = [llm_model, VISION_OLLAMA_MODEL]
    info(f"Will ensure: {', '.join(models_to_pull)}")

    if not is_installed("ollama"):
        fail("ollama binary not found after brew install. Try: brew install ollama")

    r = run("curl -sf http://localhost:11434/api/tags", check=False, capture=True)
    if r.returncode != 0:
        print("  Starting Ollama service...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time
        for _ in range(30):
            time.sleep(1)
            r = run("curl -sf http://localhost:11434/api/tags", check=False, capture=True)
            if r.returncode == 0:
                break
        else:
            fail("Could not start Ollama after 30 seconds. Run 'ollama serve' manually, then re-run this script.")

    info("Ollama is running.")

    available = _get_available_models()
    if available:
        info(f"Found {len(available)} model(s) already pulled locally.")

    for model in models_to_pull:
        if _model_already_pulled(model, available):
            info(f"{model} already available - skipping pull.")
        else:
            # Retry on transient network flakes. Ollama's server resumes partial
            # downloads server-side, so re-invoking `ollama pull` picks up where
            # the last attempt stopped; no need to clear anything between retries.
            print(f"\n  Pulling {model} (first time only - this is a large download)...")
            import time as _time
            max_attempts = 4
            last_err = ""
            pulled = False
            for attempt in range(1, max_attempts + 1):
                try:
                    result = subprocess.run(
                        f"ollama pull '{model}'",
                        shell=True, check=False, timeout=1800, capture_output=True, text=True,
                    )
                    if result.returncode == 0:
                        pulled = True
                        break
                    last_err = (result.stderr or result.stdout or "").strip().splitlines()[-1:]
                    last_err = last_err[0] if last_err else f"exit {result.returncode}"
                except subprocess.TimeoutExpired:
                    last_err = "exceeded 30-minute window"
                if attempt < max_attempts:
                    wait = 5 * attempt   # 5s, 10s, 15s
                    warn(f"Pull attempt {attempt}/{max_attempts} failed ({last_err}). Retrying in {wait}s...")
                    _time.sleep(wait)
            if not pulled:
                fail(
                    f"Failed to pull {model} after {max_attempts} attempts. Last error: {last_err}\n"
                    f"  Check internet connectivity, then re-run the installer (downloads resume)."
                )
            available.add(model)
            info(f"{model} ready.")


# ─── Step 8: CLI wrapper + symlink ───────────────────────────────────────────

def create_cli_wrapper():
    header("Step 10/10 - Creating `muneem` CLI command")

    wrapper = BIN_DIR / "muneem"
    wrapper.write_text(textwrap.dedent(f"""\
        #!/bin/zsh
        MUNEEM_HOME="$HOME/.muneem"

        # ── uninstall is handled in pure shell (no venv needed) ──
        if [[ "$1" == "uninstall" ]]; then
          echo ""
          echo "  Muneem Uninstall"
          echo "  ════════════════════════════════════════"
          echo ""
          echo "  This will remove:"
          echo "    - $MUNEEM_HOME  (venv, modules, native helper, tmp)"
          echo "    - /usr/local/bin/muneem  (this CLI)"
          echo ""
          echo "  This will NOT remove:"
          echo "    - Ollama (brew service)"
          echo "    - Downloaded Ollama models (~/.ollama/models/)"
          echo "    - Homebrew packages (ffmpeg, sox, portaudio, blackhole-2ch)"
          echo ""
          KEEP_NOTES=1
          if [[ "$2" == "--all" ]]; then
            KEEP_NOTES=0
            echo "  ⚠  --all flag: notes will ALSO be deleted."
          else
            echo "  Your notes will be preserved at $MUNEEM_HOME/notes/"
            echo "  Use 'muneem uninstall --all' to also delete notes."
          fi
          echo ""
          printf "  Proceed? [y/N] "
          read -r confirm
          if [[ "$confirm" != [yY] ]]; then
            echo "  Aborted."
            exit 0
          fi
          echo ""
          if [[ "$KEEP_NOTES" -eq 1 && -d "$MUNEEM_HOME/notes" ]]; then
            BACKUP="$HOME/muneem-notes-backup"
            echo "  → Backing up notes to $BACKUP/ ..."
            rm -rf "$BACKUP"
            cp -R "$MUNEEM_HOME/notes" "$BACKUP"
          fi
          echo "  → Removing $MUNEEM_HOME ..."
          rm -rf "$MUNEEM_HOME"
          if [[ "$KEEP_NOTES" -eq 1 && -d "$BACKUP" ]]; then
            mkdir -p "$MUNEEM_HOME"
            mv "$BACKUP" "$MUNEEM_HOME/notes"
            echo "  → Notes restored to $MUNEEM_HOME/notes/"
          fi
          echo "  → Removing /usr/local/bin/muneem ..."
          if [[ -L "/usr/local/bin/muneem" || -f "/usr/local/bin/muneem" ]]; then
            rm -f "/usr/local/bin/muneem" 2>/dev/null || sudo rm -f "/usr/local/bin/muneem"
          fi
          echo ""
          echo "  ✓  Muneem has been uninstalled."
          echo "     Ollama and all downloaded models are untouched."
          if [[ "$KEEP_NOTES" -eq 1 ]]; then
            echo "     Your notes are still at $MUNEEM_HOME/notes/"
          fi
          echo ""
          echo "  To reinstall:  python3 $MUNEEM_HOME/muneem-setup.py"
          echo ""
          exit 0
        fi

        source "$MUNEEM_HOME/venv/bin/activate"

        # Fix SSL certificate verification for macOS Python.org builds.
        # Python.org's Python 3.x doesn't ship with macOS system certs; certifi provides them.
        # Use 'python' (venv-activated) so the path works regardless of Python version (3.11 or 3.12).
        CERTIFI_CERTS=$(python -c "import certifi; print(certifi.where())" 2>/dev/null)
        if [[ -n "$CERTIFI_CERTS" && -f "$CERTIFI_CERTS" ]]; then
          export SSL_CERT_FILE="$CERTIFI_CERTS"
          export REQUESTS_CA_BUNDLE="$CERTIFI_CERTS"
        fi

        case "$1" in
          start)    shift; python "$MUNEEM_HOME/app.py" start "$@" ;;
          notes)    shift; python "$MUNEEM_HOME/app.py" notes "$@" ;;
          ask)      shift; python "$MUNEEM_HOME/app.py" ask "$@" ;;
          status)   shift; python "$MUNEEM_HOME/app.py" status "$@" ;;
          doctor)   shift; python "$MUNEEM_HOME/app.py" doctor "$@" ;;
          config)   shift; python "$MUNEEM_HOME/app.py" config "$@" ;;
          help|-h|--help|"")  python "$MUNEEM_HOME/app.py" help ;;
          *)        python "$MUNEEM_HOME/app.py" "$@" ;;
        esac
    """))
    wrapper.chmod(0o755)
    info(f"CLI wrapper created at {wrapper}")

    symlink = Path("/usr/local/bin/muneem")
    try:
        symlink.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        run(f'sudo mkdir -p "{symlink.parent}"')

    try:
        if symlink.is_symlink() or symlink.exists():
            symlink.unlink()
        symlink.symlink_to(wrapper)
        info(f"Symlinked {symlink} -> {wrapper}")
    except PermissionError:
        print("  Symlink creation requires elevated privileges.")
        result = subprocess.run(["/usr/bin/sudo", "-n", "true"], capture_output=True)
        if result.returncode != 0:
            print("  This will prompt for your password (one time only).")
        try:
            run(f'sudo ln -sf "{wrapper}" "{symlink}"')
            info(f"Symlinked {symlink} -> {wrapper}")
        except Exception as e:
            warn(f"Symlink creation failed: {e}")
            print(f"  Manual fix: sudo ln -sf {wrapper} {symlink}")
    except Exception as e:
        fail(f"Unexpected error creating symlink: {e}")


# ─── Embedded module: transcriber ────────────────────────────────────────────

_MODULE_TRANSCRIBER = textwrap.dedent('''\
    """
    Muneem - Accuracy-first audio capture and transcription.

    Audio capture: Core Audio Tap (default) with BlackHole fallback.
    Transcription: WhisperX large-v3 with forced alignment.
    Diarization: sherpa-onnx (pyannote-seg + 3D-Speaker CAM++) with `diarize` fallback.
      Session-scoped SpeakerRegistry keeps speaker IDs stable across segments via
      cosine-similarity centroid matching on embeddings. Apache-2.0, fully offline.

    This pipeline prioritises accuracy with short segments for near real-time feedback.
    """

    import sys as _sys
    # Block pyaudio imports BEFORE whisperx/pyannote loads. PyAudio crashes in
    # Pa_Initialize on macOS 26 Tahoe (PaMacCore null fn ptr). Muneem never uses
    # pyaudio - audio is captured via sox and the Swift Core Audio Tap helper.
    # Setting sys.modules["pyaudio"] = None causes `import pyaudio` to raise
    # ImportError, which pyannote handles gracefully (treats it as unavailable).
    _sys.modules["pyaudio"] = None

    import os
    import subprocess
    import signal
    import time
    import tempfile
    import threading
    import warnings
    import wave
    import numpy as np
    from pathlib import Path

    # Suppress known non-critical warnings on macOS
    warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
    warnings.filterwarnings("ignore", message=".*Symbol not found.*")
    warnings.filterwarnings("ignore", message=".*Library not loaded.*")

    MUNEEM_HOME = Path.home() / ".muneem"
    NATIVE_BINARY = MUNEEM_HOME / "native" / "muneem-audio"
    TMP_DIR = MUNEEM_HOME / "tmp"
    RATE = 16000
    CHANNELS = 1
    # 30s aligns with Whisper's training window for maximum accuracy. Shorter
    # segments lose sentence context and force beam search to operate on
    # fragments, which increases hallucinations and word-drop at boundaries.
    # On CPU, large-v3 transcribes 30s of audio in ~10-15s on M1/M2, so the
    # pipeline still keeps up in real-time. If the user explicitly wants
    # lower latency feedback, `muneem start --post-process` records first and
    # transcribes later with zero CPU contention during the call.
    SEGMENT_DURATION = 30
    # Silence detection tuning.
    #
    # SILENCE_THRESHOLD is the RMS gate (below this = skip transcription).
    # Lowered from 150 to 80 so quiet speakers / soft-spoken participants
    # on conference calls don't get dropped entirely. Whisper itself will
    # handle true silence cheaply; the gate is here only to avoid loading
    # the model for a fully dead segment.
    #
    # PEAK_SILENCE_AMP is the peak-amplitude gate - if any sample exceeds
    # this, we treat the segment as non-silent regardless of RMS, so that a
    # short burst of speech in an otherwise quiet window isn't filtered out.
    SILENCE_THRESHOLD = 80
    PEAK_SILENCE_AMP = 500
    WHISPERX_MODEL = "large-v3"

    # Bias Whisper toward meeting vocabulary. Keeps proper nouns, acronyms,
    # and technical terms from being normalised away. Kept short to avoid
    # eating into Whisper's 224-token prompt budget.
    INITIAL_PROMPT = (
        "Business meeting transcript. Preserve names, acronyms, product "
        "names, and technical terms exactly as spoken."
    )


    def _has_native_helper() -> bool:
        return NATIVE_BINARY.exists() and os.access(str(NATIVE_BINARY), os.X_OK)


    def _ffmpeg_audio_devices() -> str:
        """List audio devices via ffmpeg avfoundation. Returns combined stderr+stdout."""
        try:
            r = subprocess.run(
                ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True, text=True, timeout=10,
            )
            return (r.stderr + r.stdout).lower()
        except Exception:
            return ""


    def _has_blackhole() -> bool:
        return "blackhole" in _ffmpeg_audio_devices()


    # ── CoreAudio device enumeration + default-device switching via ctypes ──
    # We use ctypes against CoreAudio.framework instead of depending on
    # pyobjc-framework-CoreAudio (not in our deps) or SwitchAudioSource
    # (extra brew install). The property addresses + selectors match those
    # documented in <CoreAudio/AudioHardware.h>.
    import ctypes as _ct
    import ctypes.util as _ctu
    _CA_LIB = _ctu.find_library("CoreAudio")
    _CA = _ct.CDLL(_CA_LIB) if _CA_LIB else None
    _CF_LIB = _ctu.find_library("CoreFoundation")
    _CF = _ct.CDLL(_CF_LIB) if _CF_LIB else None

    def _fourcc(s: str) -> int:
        return int.from_bytes(s.encode("ascii"), "big")

    # Property selectors (FourCC codes)
    _SEL_DEVICES      = _fourcc("dev#")
    _SEL_DEF_IN       = _fourcc("dIn ")
    _SEL_DEF_OUT      = _fourcc("dOut")
    _SEL_DEV_UID      = _fourcc("uid ")
    _SEL_OBJ_NAME     = _fourcc("lnam")
    _SEL_STREAM_CONF  = _fourcc("slay")
    _SEL_TRANSPORT    = _fourcc("tran")

    _SCOPE_GLOBAL = _fourcc("glob")
    _SCOPE_INPUT  = _fourcc("inpt")
    _SCOPE_OUTPUT = _fourcc("outp")
    _ELEM_MAIN    = 0
    _SYS_OBJ_ID   = 1

    _TRANSPORT_BT     = _fourcc("blte")
    _TRANSPORT_BT_LE  = _fourcc("blea")
    _TRANSPORT_USB    = _fourcc("usb ")
    _TRANSPORT_BUILT  = _fourcc("bltn")
    _TRANSPORT_AGG    = _fourcc("grup")
    _TRANSPORT_VIRT   = _fourcc("virt")
    _TRANSPORT_AIRPLAY = _fourcc("airp")
    _TRANSPORT_HDMI   = _fourcc("hdmi")

    class _AOPropAddr(_ct.Structure):
        _fields_ = [("mSelector", _ct.c_uint32),
                    ("mScope", _ct.c_uint32),
                    ("mElement", _ct.c_uint32)]

    # Explicit argtypes/restype on every CoreAudio/CoreFoundation symbol we
    # call. Without these, ctypes assumes c_int return and auto-coerces args,
    # which on arm64 macOS causes pointer truncation / memory corruption -
    # manifested as silent segfaults during enumeration.
    if _CA is not None:
        _CA.AudioObjectGetPropertyDataSize.argtypes = [
            _ct.c_uint32, _ct.POINTER(_AOPropAddr),
            _ct.c_uint32, _ct.c_void_p, _ct.POINTER(_ct.c_uint32),
        ]
        _CA.AudioObjectGetPropertyDataSize.restype = _ct.c_int32
        _CA.AudioObjectGetPropertyData.argtypes = [
            _ct.c_uint32, _ct.POINTER(_AOPropAddr),
            _ct.c_uint32, _ct.c_void_p,
            _ct.POINTER(_ct.c_uint32), _ct.c_void_p,
        ]
        _CA.AudioObjectGetPropertyData.restype = _ct.c_int32
        _CA.AudioObjectSetPropertyData.argtypes = [
            _ct.c_uint32, _ct.POINTER(_AOPropAddr),
            _ct.c_uint32, _ct.c_void_p,
            _ct.c_uint32, _ct.c_void_p,
        ]
        _CA.AudioObjectSetPropertyData.restype = _ct.c_int32
    if _CF is not None:
        _CF.CFStringGetLength.argtypes = [_ct.c_void_p]
        _CF.CFStringGetLength.restype = _ct.c_long
        _CF.CFStringGetMaximumSizeForEncoding.argtypes = [_ct.c_long, _ct.c_uint32]
        _CF.CFStringGetMaximumSizeForEncoding.restype = _ct.c_long
        _CF.CFStringGetCString.argtypes = [_ct.c_void_p, _ct.c_char_p, _ct.c_long, _ct.c_uint32]
        _CF.CFStringGetCString.restype = _ct.c_bool
        _CF.CFRelease.argtypes = [_ct.c_void_p]
        _CF.CFRelease.restype = None

    def _ca_avail() -> bool:
        return _CA is not None and _CF is not None

    def _ca_get_data_size(obj_id: int, sel: int, scope: int = _SCOPE_GLOBAL, elem: int = _ELEM_MAIN):
        if not _ca_avail(): return 0
        addr = _AOPropAddr(sel, scope, elem)
        size = _ct.c_uint32(0)
        err = _CA.AudioObjectGetPropertyDataSize(
            _ct.c_uint32(obj_id), _ct.byref(addr), 0, None, _ct.byref(size)
        )
        return size.value if err == 0 else 0

    def _ca_get_data(obj_id: int, sel: int, buf_type, count: int = 1,
                     scope: int = _SCOPE_GLOBAL, elem: int = _ELEM_MAIN):
        if not _ca_avail(): return None
        addr = _AOPropAddr(sel, scope, elem)
        ArrType = buf_type * count
        buf = ArrType()
        size = _ct.c_uint32(_ct.sizeof(buf))
        err = _CA.AudioObjectGetPropertyData(
            _ct.c_uint32(obj_id), _ct.byref(addr), 0, None, _ct.byref(size), _ct.byref(buf)
        )
        if err != 0: return None
        if count == 1: return buf[0]
        return list(buf)

    def _ca_cfstring_to_str(cfref) -> str:
        """Convert a CFStringRef (as c_void_p raw address) to Python str.
        Releases the CFStringRef (caller owns it from CoreAudio's 'copy' APIs)."""
        if not cfref or not _CF: return ""
        try:
            length = _CF.CFStringGetLength(cfref)
            if length <= 0: return ""
            maxsz = _CF.CFStringGetMaximumSizeForEncoding(length, 0x08000100) + 1  # kCFStringEncodingUTF8
            buf = _ct.create_string_buffer(int(maxsz))
            ok = _CF.CFStringGetCString(cfref, buf, int(maxsz), 0x08000100)
            return buf.value.decode("utf-8", errors="replace") if ok else ""
        except Exception:
            return ""
        finally:
            try: _CF.CFRelease(cfref)
            except Exception: pass

    def _ca_get_cfstring(obj_id: int, sel: int, scope: int = _SCOPE_GLOBAL) -> str:
        """Get a CFString property (UID, name) as Python str."""
        if not _ca_avail(): return ""
        addr = _AOPropAddr(sel, scope, _ELEM_MAIN)
        cfref = _ct.c_void_p(0)
        size = _ct.c_uint32(_ct.sizeof(cfref))
        err = _CA.AudioObjectGetPropertyData(
            _ct.c_uint32(obj_id), _ct.byref(addr), 0, None, _ct.byref(size), _ct.byref(cfref)
        )
        if err != 0 or not cfref.value: return ""
        return _ca_cfstring_to_str(cfref.value)

    class _AudioBuffer(_ct.Structure):
        _fields_ = [("mNumberChannels", _ct.c_uint32),
                    ("mDataByteSize", _ct.c_uint32),
                    ("mData", _ct.c_void_p)]

    def _ca_has_channels(dev_id: int, scope: int) -> bool:
        """True iff the device has at least one stream channel in the given scope.
        Used to classify a device as input-capable or output-capable.

        AudioBufferList layout (LP64):
          offset 0:  UInt32 mNumberBuffers
          offset 4:  (4 bytes padding - AudioBuffer aligned to 8)
          offset 8+: AudioBuffer mBuffers[n]   (each 16 bytes)
        """
        sz = _ca_get_data_size(dev_id, _SEL_STREAM_CONF, scope)
        if sz == 0: return False
        buf = (_ct.c_ubyte * sz)()
        addr = _AOPropAddr(_SEL_STREAM_CONF, scope, _ELEM_MAIN)
        size = _ct.c_uint32(sz)
        err = _CA.AudioObjectGetPropertyData(
            _ct.c_uint32(dev_id), _ct.byref(addr), 0, None,
            _ct.byref(size), _ct.byref(buf)
        )
        if err != 0: return False
        n = _ct.c_uint32.from_buffer(buf, 0).value
        if n == 0: return False
        ab_sz = _ct.sizeof(_AudioBuffer)
        # First AudioBuffer starts at offset 8 (4 byte header + 4 byte alignment pad).
        first_off = 8
        for i in range(int(n)):
            off = first_off + i * ab_sz
            if off + ab_sz > sz: break
            ab = _AudioBuffer.from_buffer(buf, off)
            if ab.mNumberChannels > 0:
                return True
        return False

    def _list_audio_devices(kind: str) -> list[dict]:
        """Enumerate CoreAudio devices of the given kind ('input' or 'output').

        Returns list of dicts: {id, uid, name, transport, is_bluetooth,
        is_usb, is_builtin, is_virtual, is_aggregate}.
        """
        if not _ca_avail(): return []
        scope = _SCOPE_INPUT if kind == "input" else _SCOPE_OUTPUT
        sz = _ca_get_data_size(_SYS_OBJ_ID, _SEL_DEVICES)
        if sz == 0: return []
        count = sz // _ct.sizeof(_ct.c_uint32)
        ids = _ca_get_data(_SYS_OBJ_ID, _SEL_DEVICES, _ct.c_uint32, count)
        if ids is None: return []
        # Name-based heuristic fallback for devices that report transport=0.
        # Some Bluetooth headsets (AirPods/Buds with certain codecs) return
        # no transport tag on Apple Silicon, so the Bluetooth flag must be
        # inferred from the device name.
        _BT_HINTS  = ("airpods", "buds", "beats", "wh-", "wf-", "jabra",
                      "bose", "sony", "sennheiser", "pixel buds", "surface",
                      "powerbeats", "soundlink", "urbanista", "jbl")
        _USB_HINTS = ("usb", "yeti", "snowball", "shure", "rode", "scarlett",
                      "elgato", "samson", "audio-technica", "at2020", "mv7",
                      "blue ", "podmic", "dac")
        devices = []
        for dev_id in ids:
            if not _ca_has_channels(dev_id, scope):
                continue
            name = _ca_get_cfstring(dev_id, _SEL_OBJ_NAME, scope)
            uid = _ca_get_cfstring(dev_id, _SEL_DEV_UID, scope)
            transport = _ca_get_data(dev_id, _SEL_TRANSPORT, _ct.c_uint32, 1, scope) or 0
            name_l = (name or "").lower()
            uid_l = (uid or "").lower()
            is_bt = transport in (_TRANSPORT_BT, _TRANSPORT_BT_LE) or any(
                h in name_l or h in uid_l for h in _BT_HINTS)
            is_usb = transport == _TRANSPORT_USB or (
                transport == 0 and any(h in name_l for h in _USB_HINTS) and not is_bt)
            is_builtin = transport == _TRANSPORT_BUILT or (
                transport == 0 and ("macbook" in name_l and not is_bt and not is_usb))
            is_virtual = transport == _TRANSPORT_VIRT or uid_l.startswith("~:") or (
                "blackhole" in name_l or "teams audio" in name_l or
                "zoomaudio" in name_l or "soundflower" in name_l or
                "loopback" in name_l or "screenflick" in name_l
            )
            devices.append({
                "id": int(dev_id),
                "uid": uid,
                "name": name,
                "transport": int(transport),
                "is_bluetooth": is_bt and not is_virtual,
                "is_usb":       is_usb and not is_virtual,
                "is_builtin":   is_builtin and not is_virtual,
                "is_aggregate": transport == _TRANSPORT_AGG,
                "is_virtual":   is_virtual,
                "is_hdmi":      transport == _TRANSPORT_HDMI,
                "is_airplay":   transport == _TRANSPORT_AIRPLAY,
            })
        return devices

    def _ca_get_default_device(kind: str) -> int:
        sel = _SEL_DEF_IN if kind == "input" else _SEL_DEF_OUT
        v = _ca_get_data(_SYS_OBJ_ID, sel, _ct.c_uint32, 1)
        return int(v) if v is not None else 0

    def _ca_set_default_device(kind: str, dev_id: int) -> bool:
        """Set system default input/output device. Returns True on success."""
        if not _ca_avail() or not dev_id: return False
        sel = _SEL_DEF_IN if kind == "input" else _SEL_DEF_OUT
        addr = _AOPropAddr(sel, _SCOPE_GLOBAL, _ELEM_MAIN)
        v = _ct.c_uint32(int(dev_id))
        err = _CA.AudioObjectSetPropertyData(
            _ct.c_uint32(_SYS_OBJ_ID), _ct.byref(addr), 0, None,
            _ct.c_uint32(_ct.sizeof(v)), _ct.byref(v)
        )
        return err == 0

    def _find_device_by_name(devices: list[dict], name_or_uid: str) -> dict | None:
        if not name_or_uid: return None
        n = name_or_uid.strip().lower()
        # Exact UID match
        for d in devices:
            if d["uid"].lower() == n: return d
        # Exact name match
        for d in devices:
            if d["name"].lower() == n: return d
        # Substring name match
        for d in devices:
            if n in d["name"].lower(): return d
        return None

    def _rank_output(d: dict) -> int:
        """Auto-detect priority: Bluetooth > USB > built-in > HDMI > other.
        Lower rank = higher priority."""
        # Penalise virtual/aggregate devices - they force 44.1kHz and
        # degrade Bluetooth audio. Per-process tap doesn't need them.
        if d.get("is_virtual") or d.get("is_aggregate"): return 50
        if d.get("is_bluetooth"): return 0
        if d.get("is_usb"):       return 1
        if d.get("is_builtin"):   return 2
        if d.get("is_hdmi"):      return 3
        if d.get("is_airplay"):   return 4
        return 10

    def _rank_input(d: dict) -> int:
        """Mic priority: built-in mic tends to be HIGHEST quality for speech
        on modern MacBooks (three-mic array + voice isolation). Then USB
        (podcast mics), then Bluetooth (compressed, low quality).
        User can override with --mic."""
        if d.get("is_virtual") or d.get("is_aggregate"): return 50
        if d.get("is_builtin"):   return 0
        if d.get("is_usb"):       return 1
        if d.get("is_bluetooth"): return 2
        return 10


    def _current_output_device() -> dict:
        """Return info about the system's current default-output audio device.

        Probe via the muneem-audio helper (it prints the UID to stderr on launch).
        Falls back to parsing `system_profiler SPAudioDataType`.

        Returns: {'uid': str, 'name': str, 'is_virtual': bool, 'is_bluetooth': bool}
        """
        info = {"uid": "", "name": "", "is_virtual": False, "is_bluetooth": False}
        # 1) Authoritative source: the native helper reports the same UID the tap sees.
        if _has_native_helper():
            try:
                r = subprocess.run(
                    [str(NATIVE_BINARY), "system", "/tmp/_muneem_probe.wav", "1"],
                    capture_output=True, text=True, timeout=5,
                )
                stderr = r.stderr or ""
            except subprocess.TimeoutExpired as te:
                stderr = (getattr(te, "stderr", None) or b"")
                if isinstance(stderr, bytes):
                    try: stderr = stderr.decode("utf-8", errors="replace")
                    except Exception: stderr = ""
            except Exception:
                stderr = ""
            try:
                os.remove("/tmp/_muneem_probe.wav")
            except OSError:
                pass
            import re as _re
            # The helper prints: "...for system audio (output UID: <uid>)"
            # Capture up to ) or whitespace so we don't include the trailing ')'.
            m = _re.search(r"output UID:\\s*([^\\s\\)]+)", stderr)
            if m:
                info["uid"] = m.group(1)
                info["is_virtual"] = info["uid"].startswith("~:")
        # 2) Human-readable name via system_profiler (reliable on macOS).
        # system_profiler output lists each audio device as a ": "-terminated
        # header followed by an indented block of properties. We want the name
        # of the device whose block contains "Default Output Device: Yes". A
        # simple state machine is more reliable than backward-looking because
        # several unrelated devices (Teams, Zoom virtual audio, etc.) appear
        # before the true default in the dump.
        try:
            sp = subprocess.run(
                ["/usr/sbin/system_profiler", "SPAudioDataType"],
                capture_output=True, text=True, timeout=10,
            )
            lines = (sp.stdout or "").splitlines()
            current_device = None       # most-recent "Device:" header name
            current_props  = []         # accumulator for its body
            import re as _re
            for ln in lines:
                # Device headers in `system_profiler SPAudioDataType` output
                # are 8-space-indented, end with ":", and have no further
                # ":" before that final one (so "Default Output Device: Yes"
                # is NOT treated as a device header).
                stripped = ln.rstrip()
                indent = len(ln) - len(ln.lstrip())
                is_header = (
                    indent == 8
                    and stripped.endswith(":")
                    and ":" not in stripped[:-1]
                    and stripped.strip() != ":"
                )
                if is_header:
                    # Finalise previous device block.
                    if current_device and any("Default Output Device: Yes" in p for p in current_props):
                        info["name"] = current_device
                        blk = "\\n".join(current_props).lower()
                        if "transport: bluetooth" in blk or "airpod" in blk:
                            info["is_bluetooth"] = True
                        break
                    current_device = stripped.rstrip(":").strip()
                    current_props = []
                else:
                    current_props.append(ln)
            # Handle the very last device in the file.
            if not info["name"] and current_device and any(
                "Default Output Device: Yes" in p for p in current_props
            ):
                info["name"] = current_device
                blk = "\\n".join(current_props).lower()
                if "transport: bluetooth" in blk or "airpod" in blk:
                    info["is_bluetooth"] = True
        except Exception:
            pass
        return info


    def _probe_blackhole_live(duration: float = 0.6) -> tuple[bool, bool]:
        """Quick (<1s) probe: is BlackHole currently receiving audio?

        Returns (sox_ok, captured_audio):
          sox_ok         - True if sox was able to read from BlackHole at all
                           (device present and accessible).
          captured_audio - True if the probe sample was non-zero. False means
                           BlackHole wrote all zeros, which happens when
                           (a) nothing is playing right now, OR
                           (b) the default output path doesn't include BlackHole
                               (Multi-Output Device missing BlackHole 2ch).
                           The preflight can't distinguish (a) from (b) without
                           audio actually playing, so this is advisory.
        """
        import tempfile as _tempfile
        dev = get_device_name("BlackHole") or "BlackHole 2ch"
        tmp = _tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            p = subprocess.run(
                ["sox", "-q", "-t", "coreaudio", dev,
                 "-r", str(RATE), "-c", str(CHANNELS), "-b", "16",
                 tmp, "trim", "0", str(duration)],
                capture_output=True, timeout=int(duration) + 5,
            )
            sox_ok = (p.returncode == 0 and os.path.exists(tmp)
                      and os.path.getsize(tmp) > 44)
            if not sox_ok:
                return (False, False)
            captured = not _is_all_zero_wav(tmp)
            return (True, captured)
        except Exception:
            return (False, False)
        finally:
            try: os.remove(tmp)
            except OSError: pass


    def audio_preflight(verbose: bool = True) -> dict:
        """Run-every-start audio preflight. Reports current route, tests whether
        muneem can actually capture audio, and explains what to do if not.

        Returns a dict that stream_transcribe() uses to pick the backend:
          {
            'backend': 'native' | 'blackhole' | 'mic_only',
            'output_device': {...},
            'blackhole_available': bool,
            'blackhole_live': bool | None,  # live probe: did BH actually capture?
            'native_will_work': bool,
            'warnings': [str, ...],
          }
        """
        result = {
            "backend": "mic_only",
            "output_device": {},
            "blackhole_available": False,
            "blackhole_live": None,
            "native_will_work": False,
            "warnings": [],
        }
        # Current output device
        dev = _current_output_device()
        result["output_device"] = dev
        bh_ok = _has_blackhole()
        result["blackhole_available"] = bh_ok

        if verbose:
            dev_label = dev.get("name") or (dev.get("uid") or "unknown")
            kind = "Bluetooth" if dev.get("is_bluetooth") else (
                "virtual/multi-output" if dev.get("is_virtual") else "physical"
            )
            print(f"  \\033[96m\\u266b\\033[0m Audio route: output = \\033[1m{dev_label}\\033[0m ({kind})")

        # Core Audio Taps work when the default output is a PHYSICAL device.
        # They silently produce zero-valued samples when the default output is
        # a stacked / multi-output / aggregate device - that's how the user ended
        # up with "silent meeting" notes before this preflight existed.
        native_available = _has_native_helper()
        if native_available and not dev.get("is_virtual"):
            result["native_will_work"] = True
        elif native_available and dev.get("is_virtual"):
            result["warnings"].append(
                "Your default output is a stacked/multi-output device, which Core "
                "Audio Taps cannot capture reliably (known macOS limitation). "
                "Falling back to BlackHole capture."
            )

        # Pick the best backend
        if result["native_will_work"]:
            result["backend"] = "native"
        elif bh_ok:
            result["backend"] = "blackhole"
            # If user is on Bluetooth AND has BlackHole, warn if their
            # Multi-Output doesn't include BlackHole (we can't know for sure
            # from Python without AudioToolbox; just flag the risk).
            if dev.get("is_bluetooth"):
                result["warnings"].append(
                    "You're on Bluetooth output. Muneem will capture via BlackHole; "
                    "make sure your audio is routed through a Multi-Output Device "
                    "that includes both BlackHole 2ch AND your Bluetooth headset "
                    "(Audio MIDI Setup > + > Multi-Output Device)."
                )
        elif native_available:
            result["backend"] = "native"  # best-effort; will likely produce zeros
            result["warnings"].append(
                "No BlackHole found. Core Audio Tap may produce silent capture on "
                "this output device. Install BlackHole: brew install --cask blackhole-2ch"
            )

        # Live probe: when we're planning to use BlackHole, do a ~0.6s sox
        # capture RIGHT NOW and check whether BlackHole is actually receiving
        # audio. This catches the most common failure mode: BlackHole is
        # installed, but the user's Multi-Output Device doesn't include it,
        # so every recorded "system" segment is all zeros → Whisper gets
        # silence → "No audio transcript captured".
        # The probe is advisory: a zero result could also mean "no audio is
        # playing right now", so we warn but do not block startup.
        if result["backend"] == "blackhole":
            sox_ok, captured = _probe_blackhole_live()
            result["blackhole_live"] = captured if sox_ok else None
            if not sox_ok:
                result["warnings"].append(
                    "Couldn't probe BlackHole with sox. Verify: sox --version "
                    "and ffmpeg -f avfoundation -list_devices true -i \\"\\""
                )
            elif not captured:
                dev_name = (result["output_device"].get("name") or "your output").strip()
                result["warnings"].append(
                    f"BlackHole probe captured SILENCE. Either nothing is playing "
                    f"right now, OR '{dev_name}' is not routing through BlackHole. "
                    f"If your meeting transcript is empty, open Audio MIDI Setup "
                    f"\\u2192 '{dev_name}' \\u2192 ensure 'BlackHole 2ch' is CHECKED "
                    f"in the device list."
                )

        if verbose:
            label = {
                "native":    "Core Audio Tap (per-output)",
                "blackhole": "BlackHole (via sox)",
                "mic_only":  "Microphone ONLY (no system audio)",
            }.get(result["backend"], result["backend"])
            print(f"  \\033[96m\\u266b\\033[0m Capture method: \\033[1m{label}\\033[0m")
            if result["backend"] == "blackhole" and result["blackhole_live"] is True:
                print(f"  \\033[92m\\u2713\\033[0m  BlackHole live-probe: audio is flowing.")
            elif result["backend"] == "blackhole" and result["blackhole_live"] is False:
                print(f"  \\033[93m!\\033[0m  BlackHole live-probe: NO audio captured "
                      f"(play any sound and re-run to verify routing).")
            for w in result["warnings"]:
                print(f"  \\033[93m!\\033[0m  {w}")
        return result


    def get_device_name(name_substring: str) -> str | None:
        """Return the CoreAudio device name matching name_substring, for use with sox."""
        output = _ffmpeg_audio_devices()
        import re as _re
        for line in output.splitlines():
            if name_substring.lower() in line:
                m = _re.search(r'\\[\\d+\\]\\s+(.+)', line)
                if m:
                    return m.group(1).strip()
        # Fallback: return the canonical name directly
        if name_substring.lower() == "blackhole":
            return "BlackHole 2ch"
        return name_substring


    # Keep old name as alias so app.py imports don't break
    def get_device_index(name_substring: str):
        return get_device_name(name_substring)


    # Track in-flight recorder subprocesses so stream_transcribe can kill them
    # on stop instead of waiting up to SEGMENT_DURATION for sox/native to finish.
    _active_procs: set = set()
    _active_procs_lock = threading.Lock()


    def _register_proc(p):
        with _active_procs_lock:
            _active_procs.add(p)


    def _unregister_proc(p):
        with _active_procs_lock:
            _active_procs.discard(p)


    def _kill_active_procs():
        """Terminate all currently-running recorder subprocesses (sox / native)."""
        with _active_procs_lock:
            procs = list(_active_procs)
            _active_procs.clear()
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        # Give them a beat to flush, then hard-kill any stragglers.
        for p in procs:
            try:
                p.wait(timeout=0.5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


    def _record_segment_native(mode: str, duration: int, output: str,
                                pids: list | None = None) -> bool:
        """Record via the Swift helper.

        mode: "system" (system audio tap) or "mic" (microphone).
        pids: optional list of int PIDs. When set AND mode=="system", the
              Swift helper uses a per-PROCESS Core Audio Tap
              (CATapDescription(stereoMixdownOfProcesses:)) instead of a
              global output tap. This is the ONLY path that captures audio
              reliably when the default output is Bluetooth (where global
              taps silently return zeros) or when the app outputs to a
              virtual device (Zoom Audio Device, Teams Audio, etc.).
              PIDs are passed via MUNEEM_TAP_PIDS env var.

        Launch path:
          macOS 14.4+ added kTCCServiceAudioCapture, a new TCC class
          governing process taps. When a CLI tool asks for it, TCC looks
          at the RESPONSIBLE process (the shell/terminal that started
          the chain) for NSAudioCaptureUsageDescription. Terminal.app,
          iTerm2, and most terminals don't declare that key, so TCC
          silently refuses and the tap returns zero-filled buffers.
          Launching the helper via `open -W -n -a <bundle>` makes launchd
          the parent, and TCC then reads the bundle's Info.plist (which
          declares the key) - permission can be granted and the tap
          delivers real audio. The fallback to direct-binary execution
          is kept for older machines without the .app bundle.
        """
        if not _has_native_helper():
            return False
        env = os.environ.copy()
        if mode == "system" and pids:
            env["MUNEEM_TAP_PIDS"] = ",".join(str(int(p)) for p in pids if p)
        app_bundle = MUNEEM_HOME / "native" / "muneem-audio.app"
        use_bundle = app_bundle.exists() and app_bundle.is_dir()
        try:
            # Remove a stale output WAV so we can detect a silent failure
            # (helper didn't run / never wrote anything).
            if os.path.exists(output):
                try: os.remove(output)
                except Exception: pass

            if use_bundle:
                # `open -W -n -a BUNDLE --env VAR=val --args CLI-args`
                # -W   wait for the launched app to exit
                # -n   force a new instance (otherwise reuses a cached one)
                # --env  forward env vars through launchd (macOS 14+)
                cmd = ["open", "-W", "-n", "-a", str(app_bundle)]
                if mode == "system" and pids:
                    cmd += ["--env", f"MUNEEM_TAP_PIDS={env['MUNEEM_TAP_PIDS']}"]
                cmd += ["--args", mode, output, str(duration)]
                p = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            else:
                p = subprocess.Popen(
                    [str(NATIVE_BINARY), mode, output, str(duration)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                )
            _register_proc(p)
            try:
                # `open` returns as soon as the bundled app exits, so the
                # timeout should only trigger if something is very wrong.
                p.wait(timeout=duration + 15)
            finally:
                _unregister_proc(p)
            # Success = child exited cleanly AND an output WAV of reasonable
            # size actually made it to disk. `open` can return 0 even if the
            # bundled app crashed on startup, so file-existence is load-bearing.
            ok = (p.returncode == 0
                  and os.path.exists(output)
                  and os.path.getsize(output) > 1024)
            return ok
        except subprocess.TimeoutExpired:
            try: p.kill()
            except Exception: pass
            return False
        except Exception:
            return False


    def _record_segment_sox(device_name: str | None, duration: int, output: str) -> bool:
        """Record audio from a CoreAudio device using sox.

        device_name: CoreAudio device name, e.g. 'BlackHole 2ch'. None = default input (mic).
        Uses Popen so stream_transcribe can terminate the process on stop - waiting for
        sox's 'trim' clock to expire naturally costs up to SEGMENT_DURATION seconds on
        shutdown, which makes Ctrl+C feel unresponsive.
        """
        dev = device_name if device_name else "default"
        p = None
        try:
            p = subprocess.Popen(
                [
                    "sox", "-q",
                    "-t", "coreaudio", dev,
                    "-r", str(RATE), "-c", str(CHANNELS), "-b", "16",
                    output,
                    "trim", "0", str(duration),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _register_proc(p)
            try:
                p.wait(timeout=duration + 10)
            finally:
                _unregister_proc(p)
            return p.returncode == 0 and os.path.exists(output)
        except subprocess.TimeoutExpired:
            if p:
                try: p.kill()
                except Exception: pass
            return False
        except Exception as exc:
            print(f"[muneem] sox record error ({dev}): {exc}")
            return False


    # Alias kept for backward-compat with any older call sites; routes to sox.
    # (Historical name; PyAudio was removed - no PortAudio dependency remains.)
    def _record_segment_pyaudio(device_index, duration: int, output: str):
        name = device_index if isinstance(device_index, str) else None
        _record_segment_sox(name, duration, output)


    def is_silent(wav_path: str) -> bool:
        """Classify a WAV as silent using a dual-gate policy:
          - Peak amplitude below PEAK_SILENCE_AMP  AND
          - RMS below SILENCE_THRESHOLD
        A single peak is enough to treat the segment as non-silent - normal
        conversation has peaks even during quiet passages, and we err on the
        side of capturing everything that might have speech.
        """
        try:
            with wave.open(wav_path, "rb") as wf:
                data = wf.readframes(wf.getnframes())
        except Exception:
            return True
        audio = np.frombuffer(data, dtype=np.int16)
        if len(audio) == 0:
            return True
        peak = int(np.max(np.abs(audio))) if len(audio) else 0
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        # If peak is high, treat as speech even if RMS is low (burst + pauses).
        if peak > PEAK_SILENCE_AMP:
            return False
        return rms < SILENCE_THRESHOLD


    def _is_all_zero_wav(wav_path: str) -> bool:
        """True when a captured WAV has frames but every sample is literally 0.
        This is the Core Audio Tap failure mode on stacked/multi-output devices -
        distinct from 'silent' (which can include low-level noise). We use this
        to decide whether to fall back to BlackHole."""
        try:
            with wave.open(wav_path, "rb") as wf:
                n = wf.getnframes()
                if n == 0:
                    return False  # empty file ≠ "all zero" samples; treat as inconclusive
                data = wf.readframes(n)
        except Exception:
            return False
        audio = np.frombuffer(data, dtype=np.int16)
        if len(audio) == 0:
            return False
        return int(np.max(np.abs(audio))) == 0


    # ── Model cache (loaded once per session, reused across segments) ──
    # Language is auto-detected by Whisper on the first segment that contains
    # speech. The detected language pins the alignment model we load: align
    # models are language-specific, so swapping them per segment would be
    # expensive. We keep a small dict keyed by language code so a meeting
    # that legitimately switches languages (rare) still works correctly.
    _model_lock = threading.Lock()
    _whisperx_model = None
    _align_cache = {}  # language_code -> (align_model, align_metadata)
    _session_language = [None]  # pinned after first confident detection

    # sherpa-onnx diarization handles - lazily loaded on first use.
    _sherpa_diar = None     # sherpa_onnx.OfflineSpeakerDiarization instance
    _sherpa_emb  = None     # sherpa_onnx.SpeakerEmbeddingExtractor instance
    _sherpa_disabled = False  # set True once if load fails, to stop retrying

    MODELS_DIR         = MUNEEM_HOME / "models"
    SHERPA_SEG_MODEL   = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    SHERPA_EMB_MODEL   = MODELS_DIR / "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"


    # ── Persistent speaker identity across segments ──
    # sherpa-onnx diarization clusters speakers within each clip it processes,
    # but local Speaker 0/1/2 labels have no cross-clip meaning: Speaker 0 in
    # clip N may be the same human as Speaker 2 in clip N+1. Without this
    # layer, "male and female dialogues get shuffled between Speaker 1 and 2"
    # - the exact symptom the user reported.
    #
    # SpeakerRegistry anchors each local speaker to a GLOBAL label by matching
    # their CAM++ embedding centroid against the registry via cosine similarity.
    # Match → reuse global label + update running-mean centroid.
    # No match → mint a new global label.
    class SpeakerRegistry:
        def __init__(self, threshold: float = 0.55):
            # 0.55 is conservative for CAM++ / 3D-Speaker embeddings. Raise
            # toward 0.7 if distinct speakers are being merged; lower toward
            # 0.45 if the same speaker is being split across multiple labels.
            self.threshold = threshold
            self._lock = threading.Lock()
            self._speakers: list = []   # [{"id": str, "centroid": np.ndarray, "count": int}]

        def reset(self):
            with self._lock:
                self._speakers.clear()

        def assign(self, embedding) -> str:
            """Remap a local speaker's embedding to a stable global label."""
            emb = np.asarray(embedding, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(emb))
            if norm < 1e-8:
                return "Speaker ?"
            emb_n = emb / norm
            with self._lock:
                if not self._speakers:
                    new_id = "Speaker 1"
                    self._speakers.append({"id": new_id, "centroid": emb_n.copy(), "count": 1})
                    return new_id
                # Cosine similarity against every registry centroid.
                best_i = 0
                best_sim = -1.0
                for i, s in enumerate(self._speakers):
                    c = s["centroid"]
                    cn = float(np.linalg.norm(c))
                    if cn < 1e-8: continue
                    sim = float(np.dot(emb_n, c) / cn)
                    if sim > best_sim:
                        best_sim = sim
                        best_i = i
                if best_sim >= self.threshold:
                    s = self._speakers[best_i]
                    w = s["count"] / (s["count"] + 1)
                    s["centroid"] = w * s["centroid"] + (1 - w) * emb_n
                    nc = float(np.linalg.norm(s["centroid"]))
                    if nc > 1e-8:
                        s["centroid"] /= nc
                    s["count"] += 1
                    return s["id"]
                new_id = f"Speaker {len(self._speakers) + 1}"
                self._speakers.append({"id": new_id, "centroid": emb_n.copy(), "count": 1})
                return new_id

        def summary(self) -> str:
            with self._lock:
                return ", ".join(f"{s['id']}({s['count']})" for s in self._speakers)


    _speaker_registry = SpeakerRegistry(threshold=0.55)


    def reset_speaker_registry():
        """Call at the start of every new meeting so speaker IDs don't carry
        over. Different meetings have different people - the embedding space
        isn't meaningful across sessions."""
        _speaker_registry.reset()


    def _load_sherpa_diarization():
        """Lazy-load the sherpa-onnx diarization + embedding pipeline.
        Returns True iff models are loaded and usable. Idempotent."""
        global _sherpa_diar, _sherpa_emb, _sherpa_disabled
        if _sherpa_disabled:
            return False
        if _sherpa_diar is not None and _sherpa_emb is not None:
            return True
        if not SHERPA_SEG_MODEL.exists() or not SHERPA_EMB_MODEL.exists():
            _sherpa_disabled = True
            print(f"[muneem] sherpa-onnx diarization models missing under {MODELS_DIR} - "
                  "falling back to `diarize` package (run muneem-setup.py to fetch them).")
            return False
        try:
            import sherpa_onnx as _sox
        except ImportError as e:
            _sherpa_disabled = True
            print(f"[muneem] sherpa-onnx not importable ({e}); falling back to `diarize` package.")
            return False
        try:
            with _model_lock:
                if _sherpa_diar is not None and _sherpa_emb is not None:
                    return True
                diar_cfg = _sox.OfflineSpeakerDiarizationConfig(
                    segmentation=_sox.OfflineSpeakerSegmentationModelConfig(
                        pyannote=_sox.OfflineSpeakerSegmentationPyannoteModelConfig(
                            model=str(SHERPA_SEG_MODEL),
                        ),
                    ),
                    embedding=_sox.SpeakerEmbeddingExtractorConfig(
                        model=str(SHERPA_EMB_MODEL),
                    ),
                    # num_clusters=-1 → auto-detect speaker count per clip.
                    # Registry takes over for cross-clip identity.
                    clustering=_sox.FastClusteringConfig(num_clusters=-1, threshold=0.5),
                    min_duration_on=0.3,
                    min_duration_off=0.5,
                )
                _sherpa_diar = _sox.OfflineSpeakerDiarization(diar_cfg)
                _sherpa_emb = _sox.SpeakerEmbeddingExtractor(
                    _sox.SpeakerEmbeddingExtractorConfig(model=str(SHERPA_EMB_MODEL))
                )
            return True
        except Exception as e:
            _sherpa_disabled = True
            print(f"[muneem] Failed to load sherpa-onnx models: {e}; falling back to `diarize`.")
            return False


    def _diarize_with_sherpa(audio: np.ndarray, sr: int = 16000) -> dict:
        """Run sherpa-onnx diarization on a 16kHz mono float32 clip.

        Returns {(start_s, end_s): global_speaker_label}. Local sherpa speaker
        IDs are remapped to cross-session global labels via _speaker_registry.

        For clips too short for pyannote segmentation (<~1s of voiced audio),
        falls back to a single-speaker embedding over the whole clip.
        """
        if audio is None or len(audio) == 0:
            return {}
        if sr != 16000:
            # Linear resample (nearest neighbor). Not audiophile-grade but fine
            # for speaker embeddings.
            import numpy as _np
            ratio = 16000 / float(sr)
            n = max(1, int(len(audio) * ratio))
            idx = _np.linspace(0, len(audio) - 1, n).astype(_np.int32)
            audio = audio[idx].astype(_np.float32, copy=False)
            sr = 16000
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        result = _sherpa_diar.process(audio).sort_by_start_time()
        segs = list(result)

        out = {}
        if segs:
            # Group time ranges by local speaker; compute one embedding per
            # speaker over their concatenated audio; map to a global label.
            from collections import defaultdict as _dd
            spans_by_spk: dict = _dd(list)
            for s in segs:
                spans_by_spk[int(s.speaker)].append((float(s.start), float(s.end)))
            local_to_global = {}
            for local_spk, spans in spans_by_spk.items():
                parts = []
                for (s0, s1) in spans:
                    i0 = max(0, int(s0 * sr))
                    i1 = min(len(audio), int(s1 * sr))
                    if i1 > i0:
                        parts.append(audio[i0:i1])
                if not parts:
                    continue
                spk_audio = np.concatenate(parts)
                if len(spk_audio) < int(sr * 0.3):
                    # <300ms of speech - embedding would be unstable; skip.
                    continue
                stream = _sherpa_emb.create_stream()
                stream.accept_waveform(sr, spk_audio)
                stream.input_finished()
                emb = _sherpa_emb.compute(stream)
                local_to_global[local_spk] = _speaker_registry.assign(emb)
            for s in segs:
                g = local_to_global.get(int(s.speaker))
                if g is None:
                    continue
                out[(float(s.start), float(s.end))] = g
        else:
            # No segmentation output (clip too short / mostly silence) -
            # embed the whole clip as ONE speaker if it has enough voice.
            if len(audio) >= int(sr * 0.5):
                stream = _sherpa_emb.create_stream()
                stream.accept_waveform(sr, audio)
                stream.input_finished()
                emb = _sherpa_emb.compute(stream)
                global_lbl = _speaker_registry.assign(emb)
                out[(0.0, float(len(audio) / sr))] = global_lbl
        return out


    def transcribe_and_diarize(wav_path: str, run_diarization: bool = True) -> list[dict]:
        global _whisperx_model
        import warnings
        # Suppress noisy upstream deprecation/config warnings that clutter output.
        warnings.filterwarnings("ignore", category=UserWarning, module="silero_vad")
        warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
        warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
        warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")
        warnings.filterwarnings("ignore", message=".*torchcodec.*")
        warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
        # Silence whisperx INFO/WARNING log lines (VAD, alignment).
        import logging
        for _lg in ("whisperx", "whisperx.vads", "whisperx.vads.pyannote",
                     "whisperx.alignment", "lightning", "lightning.pytorch"):
            logging.getLogger(_lg).setLevel(logging.ERROR)
        import whisperx

        device = "cpu"
        compute_type = "float32"

        with _model_lock:
            if _whisperx_model is None:
                print("[muneem] Loading WhisperX model (first segment only)...")
                # No language= hint: let Whisper auto-detect. Setting this to
                # None at load time keeps the model multilingual; transcribe()
                # will detect per call (cheap - uses the first 30s of mel).
                #
                # initial_prompt must be supplied via asr_options at load time -
                # WhisperX's FasterWhisperPipeline.transcribe() does NOT accept
                # initial_prompt as a per-call kwarg (it raises TypeError).
                # asr_options is merged with faster-whisper's defaults, so we
                # only need to set the keys we want to override.
                _asr_options = {"initial_prompt": INITIAL_PROMPT}
                _whisperx_model = whisperx.load_model(
                    WHISPERX_MODEL, device,
                    compute_type=compute_type,
                    asr_options=_asr_options,
                )

        audio = whisperx.load_audio(wav_path)

        # If the session has already pinned a language from a previous segment,
        # pass it as a hint so Whisper skips re-detection on every segment.
        # Until then, let auto-detect run.
        _lang_hint = _session_language[0]
        _t_kwargs = {"batch_size": 4}
        if _lang_hint:
            _t_kwargs["language"] = _lang_hint
        result = _whisperx_model.transcribe(audio, **_t_kwargs)

        detected_lang = result.get("language") or _lang_hint or "en"
        # Pin the first confidently detected language for the rest of the
        # session. Subsequent segments will reuse the same alignment model.
        if _session_language[0] is None and result.get("language"):
            _session_language[0] = detected_lang

        with _model_lock:
            cached = _align_cache.get(detected_lang)
            if cached is None:
                try:
                    am, amd = whisperx.load_align_model(language_code=detected_lang, device=device)
                    _align_cache[detected_lang] = (am, amd)
                    cached = (am, amd)
                except Exception as e:
                    # Some languages don't have a default wav2vec2 alignment
                    # model in whisperx. Fall through without alignment rather
                    # than crash the whole transcript.
                    print(f"[muneem] Alignment model unavailable for '{detected_lang}' ({e}); "
                          f"continuing with un-aligned word timings.")
                    cached = None
        if cached is not None:
            _align_model, _align_metadata = cached
            result = whisperx.align(result["segments"], _align_model, _align_metadata, audio, device)

        diarize_map = {}
        if run_diarization:
            # Primary path: sherpa-onnx + SpeakerRegistry. The registry keeps
            # speaker identity stable across segments - previously, per-clip
            # clustering produced inconsistent Speaker 1/2 labels (male and
            # female dialogues getting swapped).
            sherpa_ok = False
            if _load_sherpa_diarization():
                try:
                    sh_map = _diarize_with_sherpa(audio, sr=16000)
                    for (ds, de), spk in sh_map.items():
                        diarize_map[(round(ds, 1), round(de, 1))] = spk
                    sherpa_ok = True
                except Exception as e:
                    print(f"[muneem] sherpa-onnx diarization failed ({e}); "
                          "falling back to `diarize` package for this segment.")
            # Fallback: the old per-clip `diarize` package (local labels only -
            # won't be stable across clips, but better than no labels at all).
            if not sherpa_ok:
                try:
                    import logging as _dlog
                    _dlog.getLogger("diarize").setLevel(_dlog.ERROR)
                    from diarize import diarize as run_diar
                    import io, contextlib
                    with contextlib.redirect_stdout(io.StringIO()):
                        diar_result = run_diar(wav_path)
                    for dseg in diar_result.segments:
                        diarize_map[(round(dseg.start, 1), round(dseg.end, 1))] = dseg.speaker
                except Exception as e:
                    print(f"[muneem] Diarization unavailable ({e}); continuing without speaker labels.")

        segments = []
        for seg in result.get("segments", []):
            text = _clean_text(seg.get("text", ""))
            if not text:
                continue
            speaker = "Unknown"
            if diarize_map:
                seg_start = round(seg.get("start", 0), 1)
                seg_end = round(seg.get("end", 0), 1)
                best_overlap = 0
                for (ds, de), spk in diarize_map.items():
                    overlap = max(0, min(seg_end, de) - max(seg_start, ds))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        speaker = spk
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": text,
                "speaker": speaker,
            })
        return _deduplicate_segments(segments)


    import re

    # Only genuine non-lexical disfluencies. Real English words like "so",
    # "like", "right", "actually", "basically", "you know", "I mean" carry
    # meaning and are intentionally NOT stripped - stripping them destroyed
    # accuracy in earlier versions (users reported dropped words).
    _FILLER_PATTERN = re.compile(
        r"\\b(uh+|um+|uhm+|hmm+|hm+|ah+|er+|erm+|mhm+|mm+|mmhm+|uh huh)\\b",
        re.IGNORECASE,
    )
    _MULTI_SPACE = re.compile(r"  +")
    # Only collapse 3+ consecutive identical words. 2x repetition is common
    # emphasis in natural speech ("very very good", "no no", "yes yes") and
    # must be preserved.
    _STUTTER_PATTERN = re.compile(r"\\b(\\w+)(\\s+\\1){2,}\\b", re.IGNORECASE)


    def _clean_text(text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        text = _FILLER_PATTERN.sub("", text)
        text = _STUTTER_PATTERN.sub(r"\\1", text)
        text = _MULTI_SPACE.sub(" ", text).strip()
        text = text.strip(" ,.-")
        return text


    def _deduplicate_segments(segments: list[dict]) -> list[dict]:
        """Conservative dedup: only merges segments when they genuinely overlap
        in time AND share the same speaker AND share the same text. The old
        prefix-based heuristic (if next.startswith(prev)) was unsafe - two
        different speakers both starting with 'I think...' would lose one of
        them entirely. Timestamp-based merging cannot drop real content.
        """
        if not segments:
            return segments
        deduped = [segments[0]]
        for seg in segments[1:]:
            prev = deduped[-1]
            same_speaker = seg.get("speaker") == prev.get("speaker")
            same_text = seg.get("text", "").strip() == prev.get("text", "").strip()
            time_overlap = float(seg.get("start", 0)) < float(prev.get("end", 0))
            if same_speaker and same_text and time_overlap:
                # Same words, same speaker, overlapping timestamps = one
                # utterance split across segment boundaries. Extend prev.
                prev["end"] = max(float(prev.get("end", 0)), float(seg.get("end", 0)))
                continue
            deduped.append(seg)
        return deduped


    def _default_output_is_stacked() -> bool:
        """Return True when the system's default output is a stacked/aggregate/
        multi-output device. Core Audio Process Taps CANNOT capture real audio
        from such devices - the tap runs and exits 0, but every sample is zero.
        Detected by the macOS convention that virtual/stacked UIDs start with
        '~:' (e.g. '~:AMS2_StackedOutput:1', '~:AMSAggregateOutput:').

        When True, we must fall back to BlackHole-based capture even if the
        native tap binary exists - otherwise the user gets a 'silent meeting'.

        Probe strategy: run muneem-audio with a very short duration (1s). The
        helper logs the resolved default-output UID to stderr during startup,
        so we get the authoritative UID the tap would use. We bound the call
        with a subprocess timeout as a backstop.
        """
        if not _has_native_helper():
            return False
        import re as _re
        probe_path = "/tmp/_muneem_probe.wav"
        try:
            r = subprocess.run(
                [str(NATIVE_BINARY), "system", probe_path, "1"],
                capture_output=True, text=True, timeout=5,
            )
            stderr = r.stderr or ""
        except subprocess.TimeoutExpired as te:
            # Older Python versions may not populate .stderr on timeout; fallback to "".
            stderr = getattr(te, "stderr", None) or ""
            if isinstance(stderr, bytes):
                try: stderr = stderr.decode("utf-8", errors="replace")
                except Exception: stderr = ""
        except Exception:
            stderr = ""
        finally:
            try: os.remove(probe_path)
            except OSError: pass
        m = _re.search(r"output UID:\\s*(\\S+)", stderr)
        if not m:
            return False
        uid = m.group(1)
        return uid.startswith("~:")


    def choose_backend() -> str:
        """Pick the audio capture backend.

        Order of preference:
          1. BlackHole (via sox) when the default output is stacked/multi-output,
             OR when the user explicitly set MUNEEM_FORCE_BLACKHOLE=1.
             Reason: Core Audio Process Tap silently returns all-zero samples
             when the default output is a stacked/aggregate device - meetings
             look 'silent' even though audio is playing. BlackHole is unaffected.
          2. Core Audio Tap (native helper) when the output is a physical device
             and the tap binary is present.
          3. BlackHole (direct loopback) as a general-purpose fallback.
          4. Mic-only if nothing else is available.
        """
        force_bh = os.environ.get("MUNEEM_FORCE_BLACKHOLE") == "1"
        if force_bh and _has_blackhole():
            return "blackhole"
        if _has_native_helper() and not _default_output_is_stacked():
            return "native"
        if _has_blackhole():
            return "blackhole"
        if _has_native_helper():
            # No BlackHole and output is stacked - native tap will silently fail,
            # but it's still better than nothing (mic will still capture).
            return "native"
        return "mic_only"


    def _record_one_round(backend, bh_index, seg_id, stop_event, follow_pids=None):
        """Record one segment pair (sys + mic). Returns (sys_wav, mic_wav, has_two_tracks).

        follow_pids: optional list of int PIDs. When set and backend=="native",
        the Swift helper does a PER-PROCESS tap instead of a global output
        tap. This is the only path that works on Bluetooth output or when the
        target app uses its own virtual audio device.
        """
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        sys_wav = str(TMP_DIR / f"_sys_{seg_id}.wav")
        mic_wav = str(TMP_DIR / f"_mic_{seg_id}.wav")

        if backend == "native":
            sys_ok = [False]
            mic_ok = [False]
            # Pass follow_pids only on the SYSTEM tap (mic tap uses mic regardless).
            def _rs(): sys_ok[0] = _record_segment_native(
                "system", SEGMENT_DURATION, sys_wav, pids=follow_pids
            )
            def _rm(): mic_ok[0] = _record_segment_native("mic", SEGMENT_DURATION, mic_wav)
            ts = threading.Thread(target=_rs)
            tm = threading.Thread(target=_rm)
            ts.start(); tm.start(); ts.join(); tm.join()
            if not sys_ok[0]:
                return sys_wav, mic_wav, True, "blackhole"
            # Live validation: Core Audio Process Tap silently returns
            # all-zero samples when the default output is a stacked device
            # or a Bluetooth device (known macOS limitation).
            # If per-process tap (follow_pids) is active, zeros mean the
            # target process genuinely isn't producing audio right now - do
            # NOT fall back, because BlackHole won't help either.
            if not getattr(_record_one_round, "_native_validated", False):
                if _is_all_zero_wav(sys_wav):
                    _record_one_round._native_all_zero_count = (
                        getattr(_record_one_round, "_native_all_zero_count", 0) + 1
                    )
                    # 2 consecutive all-zero captures ⇒ tap is dead on this output.
                    if _record_one_round._native_all_zero_count >= 2:
                        if follow_pids:
                            # Per-process tap returning zeros means the process
                            # isn't playing audio; don't blow away the setup.
                            # Just let it keep trying - audio may start shortly.
                            pass
                        else:
                            print("[muneem] Core Audio Tap is producing silence "
                                  "(Bluetooth or stacked/multi-output device detected). "
                                  "Falling back to BlackHole.")
                            return sys_wav, mic_wav, True, "blackhole"
                else:
                    _record_one_round._native_validated = True
            return sys_wav, mic_wav, True, None
        elif backend == "blackhole":
            # bh_index is now a device name string from get_device_name()
            ts = threading.Thread(target=_record_segment_sox, args=(bh_index, SEGMENT_DURATION, sys_wav))
            tm = threading.Thread(target=_record_segment_sox, args=(None, SEGMENT_DURATION, mic_wav))
            ts.start(); tm.start(); ts.join(); tm.join()
            # Detect if BlackHole is actually capturing audio. If the system
            # track is consistently silent (BlackHole not in audio routing),
            # auto-degrade to mic_only so we don't waste resources recording
            # an empty track.  The per-segment transcriber already diarizes
            # mic when sys is silent, so this just skips the dead recording.
            if not getattr(_record_one_round, "_bh_validated", False):
                if os.path.exists(sys_wav) and is_silent(sys_wav):
                    _record_one_round._bh_silent_count = (
                        getattr(_record_one_round, "_bh_silent_count", 0) + 1
                    )
                    if _record_one_round._bh_silent_count >= 3:
                        print("[muneem] BlackHole is silent (not in audio routing). "
                              "Switching to mic-only with diarization.")
                        return sys_wav, mic_wav, True, "mic_only"
                elif os.path.exists(sys_wav):
                    _record_one_round._bh_validated = True
            return sys_wav, mic_wav, True, None
        else:
            _record_segment_sox(None, SEGMENT_DURATION, mic_wav)
            return mic_wav, mic_wav, False, None


    # Track silent-segment streak so we can surface "we're listening but nobody
    # is talking" to the user instead of eerie silence for 20 minutes.
    _silent_streak = [0]
    _SILENT_NOTICE_EVERY = 6   # ~6 * 5s = ~30s between notices
    # Track BlackHole-silent-but-mic-active streaks to warn about audio routing.
    _bh_silent_streak = [0]
    _bh_warned = [False]
    # Track acoustic-echo suppression. When the user listens on speakers instead
    # of headphones, the built-in mic re-records the remote speakers' voices, and
    # the same words get transcribed twice - once as [Speaker N] (from the tap)
    # and once as [You] (from the mic). We drop the mic copy.
    _echo_suppressed_total = [0]
    _echo_warned = [False]


    def _normalize_for_echo_match(text: str) -> str:
        import re as _re
        t = (text or "").lower()
        t = _re.sub(r"[^a-z0-9 ]+", " ", t)
        t = _re.sub(r"\\s+", " ", t).strip()
        return t


    def _mic_segment_is_echo(mic_seg: dict, sys_segs: list[dict],
                             time_tol: float = 3.0,
                             sim_thresh: float = 0.72) -> bool:
        """True if a mic segment looks like an echo of one of the sys segments
        from the same round (rough time overlap + high text similarity)."""
        import difflib
        m_text = _normalize_for_echo_match(mic_seg.get("text", ""))
        if not m_text:
            return False
        m_start = float(mic_seg.get("start", 0) or 0)
        m_end = float(mic_seg.get("end", 0) or 0)
        for ss in sys_segs:
            s_text = _normalize_for_echo_match(ss.get("text", ""))
            if not s_text:
                continue
            s_start = float(ss.get("start", 0) or 0)
            s_end = float(ss.get("end", 0) or 0)
            if s_end + time_tol < m_start or s_start - time_tol > m_end:
                continue
            if m_text == s_text:
                return True
            if len(m_text) >= 12 and (m_text in s_text or s_text in m_text):
                return True
            if difflib.SequenceMatcher(None, m_text, s_text).ratio() >= sim_thresh:
                return True
        return False


    def _cleanup_wavs(*paths):
        """Unlink all given WAV paths, ignoring missing/errored files."""
        for f in set(p for p in paths if p):
            try:
                os.remove(f)
            except OSError:
                pass


    def _transcribe_segment(sys_wav, mic_wav, has_two_tracks, callback, recv_epoch=None):
        """Transcribe a recorded segment pair and fire callbacks.

        `recv_epoch`: if provided (typically in batch/post-process mode), it
        is stamped onto every segment's seg_data so the final transcript
        carries the ORIGINAL recording wall-clock time, not the post-process
        time. The callback (MeetingSession.on_transcript) honours this.
        """
        sys_exists = os.path.exists(sys_wav)
        mic_exists = os.path.exists(mic_wav)
        sys_silent = is_silent(sys_wav) if sys_exists else True
        mic_silent = is_silent(mic_wav) if mic_exists else True

        if sys_silent and mic_silent:
            _silent_streak[0] += 1
            if _silent_streak[0] % _SILENT_NOTICE_EVERY == 0:
                mins = (_silent_streak[0] * SEGMENT_DURATION) // 60
                secs = (_silent_streak[0] * SEGMENT_DURATION) % 60
                # Differentiated messaging: report which tracks are dead.
                if has_two_tracks and sys_exists and mic_exists:
                    detail = ("both mic AND system audio are silent \\u2014 "
                              "verify mic isn't muted and output routes through BlackHole")
                elif has_two_tracks and sys_exists and not mic_exists:
                    detail = ("no mic segment captured \\u2014 sox/coreaudio may not have "
                              "been able to open the default input")
                elif not has_two_tracks:
                    detail = ("mic is silent \\u2014 check System Settings \\u2192 Privacy "
                              "& Security \\u2192 Microphone, and that your mic isn't muted")
                else:
                    detail = "no audio on any captured track"
                print(
                    f"  \\033[90m\\u00b7\\033[0m Still listening \\u2014 {detail} "
                    f"({mins}m{secs}s elapsed).",
                    flush=True,
                )
            # CRITICAL: clean up both WAVs even on the all-silent path.
            # The old code returned early here, leaking every silent segment
            # to disk (observed: 286 leaked mic files in one session).
            _cleanup_wavs(sys_wav, mic_wav)
            return
        _silent_streak[0] = 0

        all_segments = []

        if has_two_tracks and not sys_silent and not mic_silent:
            # Both tracks have audio - ideal case.
            # Mic = "You", system audio = diarized remote speakers.
            mic_segs = transcribe_and_diarize(mic_wav, run_diarization=False)
            sys_segs = transcribe_and_diarize(sys_wav, run_diarization=True)
            for seg in sys_segs:
                if seg["speaker"] == "Unknown":
                    seg["speaker"] = "Other"

            # Acoustic-echo suppression: when the user is on speakers (not
            # headphones), the built-in mic picks up the remote speakers'
            # voices, Whisper re-transcribes them, and the mic copy gets
            # mis-labelled [You]. Drop mic segments that duplicate sys audio.
            kept_mic = []
            dropped_echo = 0
            for m in mic_segs:
                if _mic_segment_is_echo(m, sys_segs):
                    dropped_echo += 1
                    continue
                m["speaker"] = "You"
                kept_mic.append(m)
            if dropped_echo:
                _echo_suppressed_total[0] += dropped_echo
                if _echo_suppressed_total[0] >= 8 and not _echo_warned[0]:
                    _echo_warned[0] = True
                    print(
                        "\\n  \\033[93m\\u26a0\\033[0m  Acoustic echo detected: your mic is "
                        "picking up meeting audio from your speakers.\\n"
                        "      Muneem is suppressing the duplicated [You] lines.\\n"
                        "      For cleaner attribution, use headphones or route output\\n"
                        "      through a Multi-Output Device that includes BlackHole 2ch.\\n",
                        flush=True,
                    )
            all_segments.extend(kept_mic)
            all_segments.extend(sys_segs)

        elif has_two_tracks and sys_silent and not mic_silent:
            # System audio silent (BlackHole not routing properly) but mic
            # picks up everyone (speakers + you).  Run diarization on the
            # mixed mic track so we get at least Speaker 0 / Speaker 1 labels
            # instead of labeling everything "You" incorrectly.
            _bh_silent_streak[0] += 1
            if _bh_silent_streak[0] >= 3 and not _bh_warned[0]:
                _bh_warned[0] = True
                print(
                    "\\n  \\033[93m\\u26a0\\033[0m  System audio (BlackHole) is silent but mic is active.\\n"
                    "      Speaker diarization is running on mic audio (all participants mixed).\\n"
                    "      For better separation, route audio through a Multi-Output Device\\n"
                    "      that includes BlackHole 2ch (Audio MIDI Setup \\u2192 + \\u2192 Multi-Output Device).\\n",
                    flush=True,
                )
            mic_segs = transcribe_and_diarize(mic_wav, run_diarization=True)
            all_segments.extend(mic_segs)

        elif has_two_tracks and not sys_silent and mic_silent:
            # Only system audio has content (rare but possible).
            sys_segs = transcribe_and_diarize(sys_wav, run_diarization=True)
            for seg in sys_segs:
                if seg["speaker"] == "Unknown":
                    seg["speaker"] = "Other"
            all_segments.extend(sys_segs)

        elif not has_two_tracks:
            # Mic-only mode: diarize to distinguish speakers.
            segs = transcribe_and_diarize(mic_wav, run_diarization=True)
            all_segments.extend(segs)

        all_segments.sort(key=lambda s: s.get("start", 0))

        for seg in all_segments:
            # In batch/post-process mode, stamp each segment with the original
            # recording epoch so the transcript timestamps reflect meeting
            # time (not when WhisperX ran on it).
            if recv_epoch is not None:
                seg["_recv_epoch"] = recv_epoch + float(seg.get("start", 0) or 0)
            label = f"[{seg['speaker']}]"
            text = f"{label} {seg['text']}".strip()
            if seg["text"] and seg["text"] not in ("[BLANK_AUDIO]", "(silence)"):
                if callback:
                    callback(text, seg)
                else:
                    print(f"[transcript] {text}")

        _cleanup_wavs(sys_wav, mic_wav)


    def stream_transcribe(
        backend: str = "auto",
        callback=None,
        stop_event: threading.Event | None = None,
        follow_pids: list | None = None,
        defer_transcription: bool = False,
        deferred_out: list | None = None,
    ):
        """
        Pipelined recording + transcription: while segment N is being
        transcribed, segment N+1 is already being recorded.
        No audio gaps between segments.

        follow_pids: optional list of int PIDs. When set and backend is
        "native", the Swift helper runs a per-PROCESS Core Audio Tap that
        captures those processes' audio directly - bypassing the Bluetooth
        output-tap limitation and per-app virtual audio devices (Zoom Audio
        Device, Teams Audio, etc.). This is the primary fix for Bluetooth
        output + web-based meetings (Arc running Zoom in the browser).

        defer_transcription: if True, ONLY record raw audio segments to disk
        during the meeting - skip WhisperX + diarization entirely while the
        user is on the call. This keeps CPU/GPU load minimal (important for
        demanding video calls). The recorded segment tuples are appended to
        `deferred_out` (if provided) so the caller can run them through
        `_transcribe_segment()` after the meeting ends.
        """
        if backend == "auto":
            backend = choose_backend()

        import queue

        # Startup sweep: delete stale _sys_*.wav / _mic_*.wav segments from a
        # previous crashed/killed session (>5 min old). Previously the all-
        # silent-segment early-return leaked these files, piling up megabytes
        # of dead audio under ~/.muneem/tmp/ across sessions.
        try:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            now = time.time()
            swept = 0
            for p in TMP_DIR.glob("_sys_*.wav"):
                if now - p.stat().st_mtime > 300:
                    try: p.unlink(); swept += 1
                    except OSError: pass
            for p in TMP_DIR.glob("_mic_*.wav"):
                if now - p.stat().st_mtime > 300:
                    try: p.unlink(); swept += 1
                    except OSError: pass
            if swept:
                print(f"[muneem] Swept {swept} stale audio segment(s) from previous session.")
        except Exception:
            pass

        backend_labels = {
            "native": "Core Audio Tap (default)",
            "blackhole": "BlackHole (fallback)",
            "mic_only": "Microphone only (no system audio)",
        }
        print(f"[muneem] Audio backend: {backend_labels.get(backend, backend)}")
        print(f"[muneem] Segment duration: {SEGMENT_DURATION}s (pipelined, no gaps)")
        print(f"[muneem] Listening... (first transcript in ~{SEGMENT_DURATION}s)")

        bh_index = get_device_name("BlackHole") if backend == "blackhole" else None
        _stop = stop_event or threading.Event()
        # Unbounded queue - recording NEVER blocks or drops segments.
        # A high-water-mark warns the user when transcription falls behind.
        seg_queue = queue.Queue(maxsize=0)
        seg_counter = [0]
        _Q_WARN_THRESHOLD = 8        # warn once when backlog reaches this
        _Q_WARN_INTERVAL  = 16       # warn again every N additional segments
        _q_warned_at = [0]           # last queue depth at which we warned

        def _recorder_thread():
            nonlocal backend, bh_index
            _reprobe_every = 2   # re-check output device every ~60s (2 * 30s)
            while not _stop.is_set():
                seg_id = seg_counter[0]
                seg_counter[0] += 1

                # Periodically re-probe the output device. If the user switched
                # from Bluetooth to speakers (or vice versa) mid-meeting, the
                # optimal backend may have changed.
                if seg_id > 0 and seg_id % _reprobe_every == 0 and backend == "mic_only":
                    new_be = choose_backend()
                    if new_be != "mic_only":
                        # Reset validation flags so the new backend gets a fair trial.
                        for attr in ("_native_validated", "_native_all_zero_count",
                                     "_bh_validated", "_bh_silent_count"):
                            try: delattr(_record_one_round, attr)
                            except AttributeError: pass
                        backend = new_be
                        if new_be == "blackhole":
                            bh_index = get_device_name("BlackHole")
                        print(f"[muneem] Output device changed - trying {backend} backend.")

                sys_wav, mic_wav, two_tracks, fallback = _record_one_round(
                    backend, bh_index, seg_id, _stop, follow_pids=follow_pids
                )
                if fallback == "blackhole":
                    print("[muneem] Native capture failed, falling back to BlackHole")
                    backend = "blackhole"
                    bh_index = get_device_name("BlackHole")
                    if not _has_blackhole():
                        print("[muneem] BlackHole not available - using mic with diarization.")
                        backend = "mic_only"
                    continue
                if fallback == "mic_only":
                    backend = "mic_only"
                    # Don't continue - the mic_wav from this round is still valid.
                    # Re-tag as single-track so the transcriber diarizes it.
                    two_tracks = False
                if _stop.is_set():
                    for f in set([sys_wav, mic_wav]):
                        try: os.remove(f)
                        except OSError: pass
                    break
                # Non-blocking put - queue is unbounded so this never blocks.
                seg_queue.put((sys_wav, mic_wav, two_tracks))
                depth = seg_queue.qsize()
                if depth >= _Q_WARN_THRESHOLD and depth - _q_warned_at[0] >= _Q_WARN_INTERVAL:
                    _q_warned_at[0] = depth
                    lag_s = depth * SEGMENT_DURATION
                    print(f"[muneem] Transcription backlog: {depth} segments (~{lag_s}s behind). "
                          f"All audio is being recorded - transcripts will catch up.")
                    # Only nudge toward post-process mode once, on the first warn.
                    if _q_warned_at[0] == depth and depth == _Q_WARN_THRESHOLD:
                        print("[muneem] Tip: for heavy calls, re-run with "
                              "`muneem start --post-process` to record only "
                              "during the call and transcribe after stop.")

        # Daemon thread so a stuck sox/native child doesn't block process exit.
        # Combined with _kill_active_procs() below, Ctrl+C is responsive.
        recorder = threading.Thread(target=_recorder_thread, daemon=True)
        recorder.start()

        _deferred_banner_shown = [False]
        _deferred_notice_every = 2    # every ~60s (2 * 30s)
        try:
            while not _stop.is_set():
                try:
                    sys_wav, mic_wav, two_tracks = seg_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if defer_transcription:
                    # Post-process mode: DO NOT transcribe now. Just hand the WAV
                    # paths to the caller for post-meeting processing.
                    # Attach a recording-time epoch so timestamps in the
                    # final transcript reflect when the audio was captured
                    # (not when WhisperX ran on it).
                    if deferred_out is not None:
                        deferred_out.append({
                            "sys_wav": sys_wav,
                            "mic_wav": mic_wav,
                            "two_tracks": two_tracks,
                            "recv_epoch": time.time(),
                        })
                    if not _deferred_banner_shown[0]:
                        _deferred_banner_shown[0] = True
                        print("[muneem] Post-process mode: recording only \\u2014 transcription deferred until stop.")
                    # Periodic reassurance every ~60s so the user knows audio is still flowing.
                    n = (deferred_out and len(deferred_out)) or 0
                    if n and n % _deferred_notice_every == 0:
                        mins = (n * SEGMENT_DURATION) // 60
                        secs = (n * SEGMENT_DURATION) % 60
                        print(f"  \\033[90m\\u00b7\\033[0m Recording... "
                              f"{n} segment(s) buffered (\\u2248{mins}m{secs}s).", flush=True)
                    continue
                _transcribe_segment(sys_wav, mic_wav, two_tracks, callback)
        except KeyboardInterrupt:
            print("\\n[muneem] Transcription stopped.")
        finally:
            _stop.set()
            # Kill any in-flight sox / native child so the recorder thread unblocks
            # immediately rather than waiting up to SEGMENT_DURATION for it to finish.
            _kill_active_procs()
            recorder.join(timeout=3)


    if __name__ == "__main__":
        backend = choose_backend()
        print(f"Detected backend: {backend}")
        stream_transcribe(backend=backend)
''')

# ─── Embedded module: screen_reader ──────────────────────────────────────────

_MODULE_SCREEN_READER = textwrap.dedent('''\
    """
    Muneem - Periodic screen capture + vision model OCR/context extraction via Ollama.

    Supports capturing: primary screen, a specific display, or a specific app window
    (e.g. Zoom, Webex, Teams, Slack). Screenshots run every CAPTURE_INTERVAL seconds.
    Capture and analysis run on separate threads with rate-limited vision calls.
    """

    import subprocess
    import base64
    import os
    import sys
    import time
    import threading
    import requests
    from pathlib import Path

    # ── Cadence ───────────────────────────────────────────────────────────────
    # CAPTURE_INTERVAL = how often screencapture writes a frame (cheap - sub-100ms).
    # ANALYSIS_INTERVAL = target cadence for the vision model. In practice the
    #   model (qwen3-vl:8b on Apple Silicon) takes 3-8s per inference, so the
    #   real cadence floats between 3 and 8 seconds - we NEVER stack requests.
    #   If the model is slower than the target, we skip to the latest frame
    #   instead of queueing stale ones (drop-when-behind).
    CAPTURE_INTERVAL = 1      # snapshot the follow target every 1s
    ANALYSIS_INTERVAL = 2     # ask the vision model ideally every 2s
    OLLAMA_URL = "http://localhost:11434"
    # Read vision model from ~/.muneem/config.json; fall back to qwen3-vl:8b for
    # older installs that predate config.json.
    try:
        import json as _json
        _cfg = _json.loads((Path.home() / ".muneem" / "config.json").read_text())
        VISION_MODEL = _cfg.get("vision_model", "qwen3-vl:8b")
    except Exception:
        VISION_MODEL = "qwen3-vl:8b"
    SCREENSHOT_DIR = str(Path.home() / ".muneem" / "tmp" / "screens")
    LATEST_FRAME = os.path.join(SCREENSHOT_DIR, "latest.png")
    _TMP_FRAME = os.path.join(SCREENSHOT_DIR, "latest_tmp.png")

    _SYSTEM_APPS = {
        "Window Server", "SystemUIServer", "Control Center",
        "Notification Center", "Dock", "Spotlight", "WindowManager",
    }
    _MIN_WINDOW_SIZE = 200


    def _ensure_dir():
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)


    def get_displays() -> list[dict]:
        """Return list of displays using Quartz. Each dict: {index: 1-based, name: str}."""
        try:
            from Quartz import CGGetActiveDisplayList, CGDisplayBounds
            err, display_ids, count = CGGetActiveDisplayList(16, None, None)
            displays = []
            for i, did in enumerate(display_ids[:count]):
                bounds = CGDisplayBounds(did)
                w, h = int(bounds.size.width), int(bounds.size.height)
                displays.append({"index": i + 1, "name": f"Screen {i + 1} ({w}x{h})"})
            return displays
        except Exception:
            return [{"index": 1, "name": "Screen 1 (primary)"}]


    # ── Call-app detection ────────────────────────────────────────────────────
    #
    # We restrict window enumeration to known meeting/call applications so the
    # picker doesn't drown the user in Finder/Cursor/Activity-Monitor windows.
    # The left column is the macOS process name (what Quartz reports as
    # kCGWindowOwnerName); the right column is a friendly label.
    #
    # Browsers are treated as call apps because Google Meet / Webex web /
    # Microsoft Teams web / WhatsApp Web run inside them. We'll still only
    # show a browser entry if the window title hints at a call (Meet URL,
    # meeting room name, "Huddle", etc.).
    _CALL_APP_OWNERS = {
        # Native call apps
        "zoom.us":                 "Zoom",
        "Microsoft Teams":         "Microsoft Teams",
        "Microsoft Teams (work or school)": "Microsoft Teams",
        "Microsoft Teams classic": "Microsoft Teams (classic)",
        "Webex":                   "Cisco Webex",
        "Cisco Webex Meetings":    "Cisco Webex",
        "WebexMeetingsHost":       "Cisco Webex",
        "Meet":                    "Google Meet",            # macOS PWA
        "Slack":                   "Slack",
        "WhatsApp":                "WhatsApp",
        "\u200eWhatsApp":          "WhatsApp",
        "Discord":                 "Discord",
        "FaceTime":                "FaceTime",
        "Skype":                   "Skype",
        "Around":                  "Around",
        "Tuple":                   "Tuple",
        # Browsers (only included if window title hints at a call - filtered below)
        "Google Chrome":           "Google Chrome",
        "Google Chrome Beta":      "Google Chrome",
        "Google Chrome Canary":    "Google Chrome",
        "Chromium":                "Chromium",
        "Brave Browser":           "Brave",
        "Arc":                     "Arc",
        "Safari":                  "Safari",
        "Firefox":                 "Firefox",
        "Microsoft Edge":          "Microsoft Edge",
        "Vivaldi":                 "Vivaldi",
    }
    _BROWSER_OWNERS = {
        "Google Chrome", "Google Chrome Beta", "Google Chrome Canary",
        "Chromium", "Brave Browser", "Arc", "Safari", "Firefox",
        "Microsoft Edge", "Vivaldi",
    }
    # Title hints that indicate an active call in a browser.
    _BROWSER_CALL_HINTS = (
        "meet.google.com", "google meet", "teams.microsoft.com", "ms teams",
        "webex.com", "cisco webex", "zoom.us", "zoom meeting",
        "whatsapp web", "slack huddle", "huddle", "meeting", "call",
        "- huddle", "- meeting", "- call with",
    )


    def get_call_windows() -> list[dict]:
        """Return windows belonging to known call/meeting applications.

        For browsers, we only include the window if the title suggests an
        ongoing call (meet.google.com in title, "Huddle", "- Meeting", etc.)
        so we don't surface random tabs.
        """
        try:
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGWindowListOptionOnScreenOnly,
                kCGWindowListExcludeDesktopElements,
                kCGNullWindowID,
            )
            raw = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
                kCGNullWindowID,
            )
        except ImportError:
            print("[muneem] Warning: Quartz framework not available.")
            return []
        except Exception as e:
            print(f"[muneem] Warning: Could not enumerate windows ({type(e).__name__}: {e}).")
            return []

        out = []
        seen = set()
        for w in raw:
            if w.get("kCGWindowLayer", -1) != 0:
                continue
            owner = (w.get("kCGWindowOwnerName") or "").strip()
            title = (w.get("kCGWindowName") or "").strip()
            wid = w.get("kCGWindowNumber")
            pid = w.get("kCGWindowOwnerPID")
            bounds = w.get("kCGWindowBounds", {})
            width = int(bounds.get("Width", 0))
            height = int(bounds.get("Height", 0))
            if wid is None or not owner:
                continue
            if owner not in _CALL_APP_OWNERS:
                continue
            if width < _MIN_WINDOW_SIZE or height < _MIN_WINDOW_SIZE:
                continue
            # For browsers: only include if title hints at a meeting.
            if owner in _BROWSER_OWNERS:
                tl = title.lower()
                if not any(h in tl for h in _BROWSER_CALL_HINTS):
                    continue
            display_title = title or "(no title)"
            label = _CALL_APP_OWNERS.get(owner, owner)
            key = (label, display_title)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "id": wid, "pid": pid,
                "owner": owner, "label": label,
                "title": display_title,
                "width": width, "height": height,
            })
        # Sort: native apps first (non-browser), then by app label.
        return sorted(
            out,
            key=lambda x: (x["owner"] in _BROWSER_OWNERS, x["label"].lower(), x["title"].lower()),
        )


    def get_windows() -> list[dict]:
        """Return ALL visible on-screen windows (not just call apps).
        Filters out system UI and tiny windows. Used by the interactive
        picker as a fallback section so the user can select any open window.
        """
        try:
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGWindowListOptionOnScreenOnly,
                kCGWindowListExcludeDesktopElements,
                kCGNullWindowID,
            )
            raw = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
                kCGNullWindowID,
            )
        except ImportError:
            return []
        except Exception:
            return []

        out = []
        seen = set()
        for w in raw:
            if w.get("kCGWindowLayer", -1) != 0:
                continue
            owner = (w.get("kCGWindowOwnerName") or "").strip()
            title = (w.get("kCGWindowName") or "").strip()
            wid = w.get("kCGWindowNumber")
            pid = w.get("kCGWindowOwnerPID")
            bounds = w.get("kCGWindowBounds", {})
            width = int(bounds.get("Width", 0))
            height = int(bounds.get("Height", 0))
            if wid is None or not owner:
                continue
            if owner in _SYSTEM_APPS:
                continue
            if width < _MIN_WINDOW_SIZE or height < _MIN_WINDOW_SIZE:
                continue
            display_title = title or "(no title)"
            key = (owner, display_title)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "id": wid, "pid": pid,
                "owner": owner, "label": owner,
                "title": display_title,
                "width": width, "height": height,
            })
        return sorted(out, key=lambda x: (x["label"].lower(), x["title"].lower()))


    def capture_screen(display_index: int | None = None, window_id: int | None = None) -> str:
        _ensure_dir()
        cmd = ["screencapture", "-x"]
        if window_id is not None:
            cmd.extend(["-l", str(window_id)])
        elif display_index is not None:
            cmd.extend(["-D", str(display_index)])
        else:
            cmd.append("-C")
        cmd.append(LATEST_FRAME)
        subprocess.run(cmd, check=True, capture_output=True)
        return LATEST_FRAME


    # Max dimension (longest edge) for images sent to the vision model.
    # Retina screens produce 3024-wide images that cost 50-100s in inference;
    # downscaling to 1280 cuts latency ~4× while preserving enough detail
    # for the model to read participant names and UI elements.
    _VISION_MAX_DIM = 1280

    def image_to_base64(path: str, max_dim: int = _VISION_MAX_DIM) -> str:
        """Read an image, optionally downscale it, and return base64-encoded PNG."""
        try:
            from PIL import Image
            import io
            img = Image.open(path)
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except ImportError:
            # Pillow not available - fall back to raw file.
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")


    def read_screen(prompt: str = "In 2-3 sentences, describe what app and content is visible on screen. Note any key text like meeting titles, chat messages, or document headings.", display_index: int | None = None, window_id: int | None = None) -> str:
        path = capture_screen(display_index=display_index, window_id=window_id)
        img_b64 = image_to_base64(path)
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": VISION_MODEL, "prompt": prompt, "images": [img_b64], "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]


    def _capture_loop(interval: int, stop_event: threading.Event, display_index: int | None = None, window_id: int | None = None):
        """Capture screenshots at regular intervals. Falls back to full screen if target window closes."""
        _ensure_dir()
        fail_count = 0
        active_window_id = window_id
        while not stop_event.is_set():
            cmd = ["screencapture", "-x"]
            if active_window_id is not None:
                cmd.extend(["-l", str(active_window_id)])
            elif display_index is not None:
                cmd.extend(["-D", str(display_index)])
            else:
                cmd.append("-C")
            cmd.append(_TMP_FRAME)
            subprocess.run(cmd, capture_output=True)
            # Atomic rename to avoid analysis thread reading partial file
            if os.path.exists(_TMP_FRAME) and os.path.getsize(_TMP_FRAME) > 0:
                os.replace(_TMP_FRAME, LATEST_FRAME)
                fail_count = 0
            else:
                fail_count += 1
                if fail_count == 3 and active_window_id is not None:
                    print("[muneem] Target window may have closed. Falling back to full screen.")
                    active_window_id = None
            stop_event.wait(interval)


    # ── Vision prompt ─────────────────────────────────────────────────────────
    # Produces BOTH narrative context (for the "Screen Context" section of notes)
    # AND structured speaker signals (names, active speaker) so the transcriber
    # can rewrite diarize's generic "Speaker 1/2/3" labels with real names.
    # JSON output keeps parsing deterministic.
    _VISION_PROMPT = (
        "You are watching a video-conferencing or collaboration app screen "
        "(Zoom, Teams, Webex, Google Meet, Slack huddle, Discord, etc.). "
        "Respond with ONLY a JSON object, no prose, matching exactly:\\n"
        "{\\n"
        '  "participants": ["<name>", ...],\\n'
        '  "active_speaker": "<name or empty string>",\\n'
        '  "shared_content": "<one short phrase or empty string>",\\n'
        '  "description": "<2 short sentences describing what is visible>"\\n'
        "}\\n"
        "Rules:\\n"
        "- `participants`: names visible next to video tiles; omit labels like \\"You\\" "
        "unless the real name is also shown.\\n"
        "- `active_speaker`: the participant whose tile is highlighted "
        "(yellow/green/blue border, raised hand, 'speaking' indicator). "
        "Empty string if none is clearly highlighted.\\n"
        "- `shared_content`: e.g. 'Figma design', 'VS Code - repo.py', "
        "'slides - Q3 roadmap'. Empty string if no screen is being shared.\\n"
        "- `description`: short, specific. No speculation.\\n"
    )


    def _parse_vision_response(raw: str) -> dict:
        """Parse the vision model's JSON response robustly."""
        import json as _json
        import re as _re
        if not raw:
            return {"participants": [], "active_speaker": "", "shared_content": "", "description": ""}
        # Strip common markdown fences.
        raw = raw.strip()
        if raw.startswith("```"):
            raw = _re.sub(r"^```[a-zA-Z0-9]*\\n?", "", raw)
            raw = _re.sub(r"\\n?```$", "", raw)
        # Find the first {...} block in the output.
        m = _re.search(r"\\{.*\\}", raw, flags=_re.DOTALL)
        if m:
            try:
                obj = _json.loads(m.group(0))
                return {
                    "participants": [p for p in obj.get("participants", []) if isinstance(p, str) and p.strip()],
                    "active_speaker": (obj.get("active_speaker") or "").strip(),
                    "shared_content": (obj.get("shared_content") or "").strip(),
                    "description": (obj.get("description") or "").strip(),
                }
            except Exception:
                pass
        # Fallback: treat whole text as description.
        return {"participants": [], "active_speaker": "", "shared_content": "", "description": raw[:400]}


    def _analysis_loop(callback, stop_event: threading.Event, analysis_interval: int = ANALYSIS_INTERVAL):
        """Run vision model as fast as it can (drop-when-behind).

        The callback receives TWO arguments now:
          - description (str)   - narrative for the Screen Context timeline
          - signals (dict)      - {participants, active_speaker, shared_content,
                                    ts_recorded, inference_ms}
        Old single-arg callbacks still work - we detect arity and fall back.
        """
        import inspect
        try:
            cb_argcount = len(inspect.signature(callback).parameters) if callback else 0
        except Exception:
            cb_argcount = 1

        def call_vision(max_retries=2):
            for attempt in range(max_retries):
                try:
                    t0 = time.time()
                    response = requests.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={
                            "model": VISION_MODEL,
                            "prompt": _VISION_PROMPT,
                            "images": [image_to_base64(LATEST_FRAME)],
                            "stream": False,
                            # NB: do NOT pass "format": "json" here - qwen3-vl:8b
                            # silently returns an empty string when format-json is
                            # enforced. The prompt already asks for JSON and the
                            # parser (_parse_vision_response) handles both fenced
                            # and bare JSON reliably.
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    elapsed_ms = int((time.time() - t0) * 1000)
                    return response.json()["response"], elapsed_ms
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        stop_event.wait(2 ** attempt)
                    else:
                        raise
                except requests.exceptions.ConnectionError:
                    if attempt < max_retries - 1:
                        stop_event.wait(2 ** attempt)
                    else:
                        raise
            return None, 0

        last_end = 0.0
        while not stop_event.is_set():
            try:
                if not os.path.exists(LATEST_FRAME):
                    stop_event.wait(0.5)
                    continue

                # Drop-when-behind: if we finished the last inference recently,
                # wait until the target cadence is met. If the model was slower
                # than the target, skip the wait and go again immediately with
                # the NEWEST frame on disk (not a queued stale one).
                elapsed_since = time.time() - last_end
                if elapsed_since < analysis_interval:
                    stop_event.wait(analysis_interval - elapsed_since)
                    if stop_event.is_set():
                        break

                raw, infer_ms = call_vision(max_retries=2)
                last_end = time.time()
                if not raw:
                    continue
                parsed = _parse_vision_response(raw)
                signals = {
                    "participants": parsed["participants"],
                    "active_speaker": parsed["active_speaker"],
                    "shared_content": parsed["shared_content"],
                    "ts_recorded": last_end,
                    "inference_ms": infer_ms,
                }
                description = parsed["description"] or ""
                if callback:
                    if cb_argcount >= 2:
                        callback(description, signals)
                    else:
                        callback(description)
                else:
                    print(f"[screen] {description[:200]}")
            except requests.exceptions.Timeout:
                print("[muneem] Vision model timeout (skipping frame).")
            except requests.exceptions.ConnectionError:
                print("[muneem] Ollama unreachable. Waiting for connection...")
                stop_event.wait(2)
            except Exception as e:
                print(f"[muneem] Screen analysis error ({type(e).__name__}): {e}")
                stop_event.wait(1)


    if __name__ == "__main__":
        print(read_screen())
''')

# ─── Embedded module: enhancer ───────────────────────────────────────────────

_MODULE_ENHANCER = textwrap.dedent('''\
    """
    Muneem - Combine transcript, screen context, and user notes into structured
    meeting notes using a local LLM via Ollama.
    """

    import requests
    from datetime import datetime

    OLLAMA_URL = "http://localhost:11434"
    # Read LLM from ~/.muneem/config.json. Falls back to 14B default if the
    # file is missing (older install) or malformed.
    try:
        import json as _json
        from pathlib import Path as _P
        _cfg = _json.loads((_P.home() / ".muneem" / "config.json").read_text())
        ENHANCE_MODEL = _cfg.get("llm_model", "qwen3:14b")
    except Exception:
        ENHANCE_MODEL = "qwen3:14b"

    # Default structured-notes template.
    #
    # Design principles:
    #   - TL;DR first - a 2-3 sentence "if you only read one paragraph".
    #   - Concrete decisions, with who agreed, not hedged "we might" phrasing.
    #   - Action items as a checklist, with owner + when (explicit or inferred).
    #   - Bulleted topics attributed to speakers, noise-stripped but faithful.
    #   - Open questions surfaced clearly - things that need follow-up.
    #   - No invented content. When the transcript is ambiguous, say so.
    #   - Screen context is supporting evidence only; if the transcript is empty
    #     the model must say so and NOT invent discussion that wasn't transcribed.
    DEFAULT_TEMPLATE = """You produce high-fidelity structured meeting notes. Work ONLY from the provided transcript, screen context, and user notes. Do not invent content. If the transcript is empty or the audio was silent, say so explicitly in the Summary and do not fabricate discussion.

    Hard rules:
    1. Every decision, action item, and open question must be traceable to a specific statement in the transcript OR a clear signal in the user's own notes. If you cannot trace it, do not include it.
    2. Attribute speakers accurately. Speaker labels in the transcript are: "You" (the user's own mic), "Other" (other participants from system audio), "Speaker 0/1/2" (diarized speakers), or named speakers from vision/diarization. Preserve these.
    3. Remove filler words, stutters, and repeated phrases. Keep substantive content verbatim-adjacent.
    4. Action items must include WHO (owner) and WHEN (explicit deadline if stated; otherwise "unspecified").
    5. If the transcript is truly empty, write a one-line Summary noting the session produced no audio transcript, then fill sections from screen context only - and label each section "(inferred from on-screen context)".

    Output format (markdown, follow exactly):
    ## Meeting Notes -- {date}

    ### Participants
    (List all unique speakers/participants identified in the transcript and screen context. For each: name or label, and brief role if apparent from context.)

    ### Executive Summary
    (3-5 sentences. High-level overview suitable for a skip-level manager or stakeholder who needs the outcome without technical detail.)

    ### TL;DR
    (2-3 sentences - the single most important takeaway a teammate who missed this meeting needs.)

    ### Decisions
    (Bulleted. Each decision: **Decision** - who agreed. If none, write "No explicit decisions recorded.")

    ### Action Items
    (Markdown checklist. Each item: `- [ ] **Owner**: action - due: <date or "unspecified">`. If none, write "No action items recorded.")

    ### Discussion
    (Bulleted, grouped by topic. Each bullet attributes the speaker(s). Stay faithful to what was actually said.)

    ### Technical Summary
    (If the meeting involved technical content - architecture, code, APIs, infrastructure, debugging, etc. - summarize those details here with specifics: system names, APIs, error codes, configuration changes, etc. Omit this section entirely if the meeting had no technical content.)

    ### Open Questions
    (Bulleted. Anything raised but unresolved, including who raised it.)

    ### Context / Artifacts
    (Anything from screen context that was visibly referenced during the meeting - e.g. docs, code, dashboards. Omit this section if nothing notable.)
    """

    TEMPLATES = {
        "default": DEFAULT_TEMPLATE,

        "standup": """Produce standup meeting notes from the transcript and context.
    The transcript includes speaker labels. Remove all filler words, stutters, and repeated phrases - output only clean sentences.

    Output format:
    ## Standup -- {date}

    ### What was done (yesterday)
    (Bullet points per speaker)

    ### What is planned (today)
    (Bullet points per speaker)

    ### Blockers
    (Any blockers mentioned, with who raised them)
    """,

        "one_on_one": """Produce 1:1 meeting notes.
    The transcript includes speaker labels. Remove all filler words, stutters, and repeated phrases - output only clean sentences.

    Output format:
    ## 1:1 Notes -- {date}

    ### Topics Discussed
    (Main themes, attributed to speakers)

    ### Feedback Given
    (Any feedback exchanged)

    ### Career / Growth
    (Development topics if discussed)

    ### Action Items
    (Next steps for each person)
    """,

        "discovery": """Produce customer/user discovery call notes.
    The transcript includes speaker labels. Remove all filler words, stutters, and repeated phrases - output only clean sentences.

    Output format:
    ## Discovery Call -- {date}

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


    # Context budget: 14B handles ~8k tokens comfortably; 32B can take more.
    # ~4 chars per token, so 12000 chars ≈ 3000 tokens (safe for 14B) and
    # 24000 chars ≈ 6000 tokens (fits 32B's larger effective window).
    MAX_TRANSCRIPT_CHARS = 24000 if ENHANCE_MODEL.endswith(":32b") else 12000
    MAX_SCREEN_CONTEXT_CHARS = 4000

    # When the transcript is too long for a single prompt, we do a map-reduce:
    # chunk the transcript, summarize each chunk into structured
    # notes, then merge those summaries into the final meeting notes. This keeps
    # information from the first hour of a long meeting from being dropped.
    MAP_CHUNK_CHARS = MAX_TRANSCRIPT_CHARS  # one chunk per model-context budget
    # Overlap between chunks so topics spanning the boundary don't get split.
    MAP_CHUNK_OVERLAP_CHARS = 800


    def _chat(system_prompt: str, user_message: str, timeout: int = 300) -> str:
        """Low-level Ollama call. Returns assistant text or raises."""
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": ENHANCE_MODEL, "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ], "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


    def _chunk_transcript(transcript: str) -> list[str]:
        """Split transcript into overlapping chunks that fit one context window."""
        if len(transcript) <= MAP_CHUNK_CHARS:
            return [transcript]
        chunks = []
        start = 0
        n = len(transcript)
        while start < n:
            end = min(start + MAP_CHUNK_CHARS, n)
            # Try to break on a line boundary within the last 1k chars so we
            # don't split a speaker's sentence mid-word.
            if end < n:
                nl = transcript.rfind("\\n", max(end - 1000, start), end)
                if nl > start + MAP_CHUNK_CHARS // 2:
                    end = nl
            chunks.append(transcript[start:end])
            if end >= n:
                break
            start = max(end - MAP_CHUNK_CHARS_OVERLAP, start + 1)
        return chunks


    # Alias name to avoid a forward-reference issue in _chunk_transcript.
    MAP_CHUNK_CHARS_OVERLAP = MAP_CHUNK_OVERLAP_CHARS


    _MAP_SYSTEM = (
        "You are summarizing ONE SEGMENT of a longer meeting transcript. Produce a "
        "compact structured summary of JUST this segment. Follow the output schema "
        "exactly. Do not invent content. If nothing in this segment matches a section, "
        "write '(none)' for that section."
    )
    _MAP_SCHEMA = """
    ### Segment summary
    (2-3 sentences)

    ### Decisions in this segment
    (Bulleted. Each decision: **Decision** - who agreed. Or '(none)'.)

    ### Action items in this segment
    (Bulleted. `- [ ] **Owner**: action - due: <when or unspecified>`. Or '(none)'.)

    ### Topics discussed
    (Bulleted, with speaker attribution.)

    ### Open questions
    (Bulleted, with who raised them. Or '(none)'.)
    """


    def _map_chunk(chunk: str, idx: int, total: int) -> str:
        user = (
            f"Segment {idx+1} of {total}. Summarize ONLY what appears below; "
            f"do not speculate beyond the text.\\n\\n"
            f"Schema:\\n{_MAP_SCHEMA}\\n\\n"
            f"--- SEGMENT TRANSCRIPT ---\\n{chunk}"
        )
        try:
            return _chat(_MAP_SYSTEM, user, timeout=180)
        except Exception as e:
            return f"(Segment {idx+1} summarization failed: {e})"


    def enhance_notes(transcript: str, screen_context: str = "", user_notes: str = "", template: str = "default") -> str:
        system_prompt = TEMPLATES.get(template, TEMPLATES["default"]).replace(
            "{date}", datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        if len(screen_context) > MAX_SCREEN_CONTEXT_CHARS:
            screen_context = screen_context[-MAX_SCREEN_CONTEXT_CHARS:]

        # ── Short transcript: one-shot ──
        if len(transcript) <= MAX_TRANSCRIPT_CHARS:
            user_message = f"""Here is everything from the meeting. Produce the structured notes.

    --- RAW TRANSCRIPT (with speaker labels) ---
    {transcript if transcript else "(no audio transcript was captured)"}

    --- SCREEN CONTEXT (what was visible on screen during the meeting) ---
    {screen_context if screen_context else "(no screen context captured)"}

    --- USER'S OWN NOTES ---
    {user_notes if user_notes else "(no manual notes taken)"}

    Now produce the structured meeting notes following the template exactly."""
            try:
                return _chat(system_prompt, user_message, timeout=300)
            except Exception as e:
                return f"(Enhancement failed: {e}. Raw transcript has been saved.)"

        # ── Long transcript: map-reduce ──
        chunks = _chunk_transcript(transcript)
        print(f"  [muneem] Long meeting ({len(transcript):,} chars). Summarizing {len(chunks)} segments...", flush=True)
        partial_summaries = []
        for i, c in enumerate(chunks):
            print(f"  [muneem] Segment {i+1}/{len(chunks)}...", flush=True)
            partial_summaries.append(_map_chunk(c, i, len(chunks)))

        merged_partials = "\\n\\n---\\n\\n".join(
            f"### Segment {i+1}\\n{s}" for i, s in enumerate(partial_summaries)
        )
        reduce_user = f"""You are merging per-segment summaries of a long meeting into final structured notes.

    Each segment below is already structured. Your job:
      - Deduplicate decisions and action items across segments.
      - Consolidate recurring topics under a single bullet, preserving attribution.
      - Preserve chronological order of decisions where that matters.
      - If the same action item appears in multiple segments, keep only the most specific/complete version.
      - Do NOT invent items that weren't in any segment.

    Follow the final output schema from the system prompt exactly.

    --- PER-SEGMENT SUMMARIES ---
    {merged_partials}

    --- SCREEN CONTEXT ---
    {screen_context if screen_context else "(no screen context captured)"}

    --- USER'S OWN NOTES ---
    {user_notes if user_notes else "(no manual notes taken)"}

    Now produce the final structured meeting notes for the ENTIRE meeting."""
        try:
            print(f"  [muneem] Merging segments into final notes...", flush=True)
            return _chat(system_prompt, reduce_user, timeout=300)
        except Exception as e:
            # Fall back to returning the partial summaries so we don't lose them.
            return (f"(Final merge failed: {e}. Raw per-segment summaries below.)\\n\\n"
                    f"{merged_partials}")


    def chat_with_transcript(transcript: str, question: str) -> str:
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": ENHANCE_MODEL, "messages": [
                    {"role": "system", "content": "You have access to a meeting transcript with speaker labels. Answer the user's question based only on what was said. Be specific and quote relevant parts, noting which speaker said what."},
                    {"role": "user", "content": f"Transcript:\\n{transcript}\\n\\nQuestion: {question}"},
                ], "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"(Could not get answer: {e}. Is Ollama running?)"
''')

# ─── Embedded module: app ────────────────────────────────────────────────────

_MODULE_APP = textwrap.dedent('''\
    """
    Muneem - Offline AI notepad. Main CLI entry point.

    Accuracy-first design: transcription uses WhisperX with forced alignment
    and offline diarization for speaker attribution. Audio is captured via
    Core Audio Tap (default) with BlackHole fallback.

    Subcommands:
        start   -- Start a meeting recording session (Ctrl+C to stop and generate notes)
        notes   -- List saved notes or open the latest one
        ask     -- Ask a question about the last meeting transcript
        status  -- Check Ollama and model availability
        doctor  -- Verify all dependencies
        help    -- Show usage
    """

    import argparse
    import os
    import platform
    import shutil
    import subprocess
    import sys
    import time
    import threading
    import json
    from datetime import datetime
    from pathlib import Path

    MUNEEM_HOME = Path.home() / ".muneem"
    OLLAMA_URL = "http://localhost:11434"
    # Runtime model selection lives in ~/.muneem/config.json. Falls back to the
    # historical defaults if the file is missing (e.g. very old install).
    CONFIG_PATH = MUNEEM_HOME / "config.json"
    try:
        _cfg = json.loads(CONFIG_PATH.read_text())
        ENHANCE_MODEL = _cfg.get("llm_model", "qwen3:14b")
        VISION_MODEL = _cfg.get("vision_model", "qwen3-vl:8b")
        WHISPER_MODEL = _cfg.get("whisper_model", "large-v3")
    except Exception:
        ENHANCE_MODEL = "qwen3:14b"
        VISION_MODEL = "qwen3-vl:8b"
        WHISPER_MODEL = "large-v3"
    NOTES_DIR = MUNEEM_HOME / "notes"
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR = MUNEEM_HOME / "tmp"
    TMP_DIR.mkdir(parents=True, exist_ok=True)


    sys.path.insert(0, str(MUNEEM_HOME))
    from transcriber import (
        stream_transcribe, choose_backend, get_device_index, get_device_name,
        _has_native_helper, _has_blackhole, audio_preflight, _current_output_device,
        _list_audio_devices, _ca_get_default_device, _ca_set_default_device,
        _find_device_by_name, _rank_input, _rank_output,
        reset_speaker_registry,
    )
    from screen_reader import (
        read_screen, get_call_windows,
        _capture_loop,
        _analysis_loop,
        LATEST_FRAME,
        _ensure_dir,
        get_displays,
        get_windows,
    )
    from enhancer import enhance_notes, chat_with_transcript, TEMPLATES


    def _get_available_models() -> list[str]:
        import requests as req
        r = req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]


    def _has_model(model: str, available: list[str]) -> bool:
        """Exact match or exact-name-with-quantization-suffix. Avoids false positives
        like 'qwen3:8b' matching 'qwen3:18b' via naive substring containment."""
        if model in available:
            return True
        return any(a == model or a.startswith(model + "-") for a in available)


    def _preflight(require_vision: bool) -> bool:
        ok = True

        backend = choose_backend()
        if backend == "mic_only":
            print("  \\033[92m✓\\033[0m  Microphone capture ready (diarization will distinguish speakers).")
            print("  \\033[90m   Tip: for cleaner speaker separation, install BlackHole:\\033[0m")
            print("  \\033[90m   brew install --cask blackhole-2ch, then set up a Multi-Output Device.\\033[0m")
            print("")
        elif backend == "blackhole":
            print("  \\033[92m✓\\033[0m  BlackHole audio capture available.")
            print("      \\033[90m(Will auto-fallback to mic + diarization if BlackHole is not in routing.)\\033[0m")
        else:
            print("  \\033[92m✓\\033[0m  Core Audio Tap available (default backend).")
            print("      \\033[90m(Will auto-fallback to mic + diarization if output device changes.)\\033[0m")

        required = [ENHANCE_MODEL]
        if require_vision:
            required.append(VISION_MODEL)
        try:
            available = _get_available_models()
        except Exception:
            print("  \\033[91m✗\\033[0m  Ollama is not reachable. Start it with: ollama serve")
            return False
        missing = [m for m in required if not _has_model(m, available)]
        if missing:
            print("  \\033[93m!\\033[0m  Required Ollama models are missing:")
            for model in missing:
                print(f"      - {model}")
            # Offer to pull them now instead of making the user paste commands.
            if sys.stdin.isatty():
                try:
                    ans = input("\\n  Pull them now? [Y/n]: ").strip().lower() or "y"
                except (EOFError, KeyboardInterrupt):
                    ans = "n"
                if ans in ("y", "yes"):
                    for model in missing:
                        print(f"\\n  Pulling {model} ...")
                        pr = subprocess.run(["ollama", "pull", model])
                        if pr.returncode != 0:
                            print(f"  \\033[91m✗\\033[0m  Pull failed for {model}. Aborting.")
                            return False
                    print("  \\033[92m\\u2713\\033[0m  All models pulled.")
                    return ok
            # Non-interactive or user declined: print exact commands and bail.
            print("\\n  To pull manually:")
            for model in missing:
                print(f"      ollama pull {model}")
            return False

        return ok


    def prompt_capture_target():
        """
        Interactive picker:
          - Running call apps (Zoom, Teams, Webex, Google Meet, Slack, browser calls, ...)
          - Other open windows (browsers, any app - useful for browser-based calls)
          - Displays/screens
          - Full screen fallback

        Returns (display_index, window_id, description_str, record_share_display).
        """
        call_windows = get_call_windows()
        displays = get_displays()
        # All windows minus the ones already shown as call apps.
        call_wids = {w["id"] for w in call_windows}
        all_windows = [w for w in get_windows() if w["id"] not in call_wids]

        print()
        print("  \\033[1mWhat should Muneem capture?\\033[0m")
        print()

        options = []  # list of (display_index, window_id, description)
        idx = 1

        # ── Section 1: running call apps ──
        if call_windows:
            print("  \\033[1m\\033[96m  Call apps currently running:\\033[0m")
            for w in call_windows:
                title = (w["title"] or "")[:60]
                if len(w.get("title", "")) > 60:
                    title += "\\u2026"
                print(f"    {idx})  \\033[1m{w['label']}\\033[0m \\u2014 {title}")
                options.append((None, w["id"], f"{w['label']} \\u2014 {w['title']}"))
                idx += 1
        else:
            print("  \\033[90m  No call apps detected currently running.\\033[0m")

        # ── Section 2: other open windows (browsers, any app) ──
        if all_windows:
            print()
            print("  \\033[1m\\033[96m  Other open windows:\\033[0m")
            for w in all_windows:
                title = (w["title"] or "")[:55]
                if len(w.get("title", "")) > 55:
                    title += "\\u2026"
                print(f"    {idx})  {w['label']} \\u2014 {title}")
                options.append((None, w["id"], f"{w['label']} \\u2014 {w['title']}"))
                idx += 1

        # ── Section 2: displays (for screen-share recording) ──
        print()
        print("  \\033[1m  Displays \\u2014 for recording YOUR screen share:\\033[0m")
        for d in displays:
            print(f"    {idx})  {d['name']}")
            options.append((d["index"], None, d["name"]))
            idx += 1

        # ── Section 3: full screen fallback ──
        print()
        print(f"    {idx})  Full screen (auto-picks primary display)")
        options.append((None, None, "Full screen"))
        idx += 1

        # Default: first call app if any, else full screen.
        default_choice = "1" if call_windows else str(len(options))

        print()
        print(f"  \\033[90mTip: pick your call app. If you plan to share your screen during\\033[0m")
        print(f"  \\033[90mthe call, muneem will offer to record that share separately.\\033[0m")
        print()

        while True:
            try:
                choice = input(f"  Enter number [{default_choice}]: ").strip() or default_choice
                n = int(choice)
                if 1 <= n <= len(options):
                    di, wi, desc = options[n - 1]
                    print(f"  \\033[92m\\u2713\\033[0m  Selected: {desc}")
                    # If user picked a call-app window, ask if they'll be screen-sharing.
                    share_idx = None
                    if wi is not None and sys.stdin.isatty():
                        print()
                        if len(displays) > 1:
                            print("  Will you share your screen in this call? If yes, which display?")
                            print(f"    0)  No, don't record my screen share")
                            for di2 in displays:
                                print(f"    {di2['index']})  {di2['name']}")
                            try:
                                s = input(f"  Choice [0]: ").strip() or "0"
                                si = int(s)
                                if si >= 1 and si <= len(displays):
                                    share_idx = si
                            except (ValueError, EOFError, KeyboardInterrupt):
                                share_idx = None
                        else:
                            try:
                                ans = input("  Will you share your screen? [y/N]: ").strip().lower()
                            except (EOFError, KeyboardInterrupt):
                                ans = "n"
                            if ans in ("y", "yes"):
                                share_idx = 1
                        if share_idx:
                            print(f"  \\033[92m\\u2713\\033[0m  Will also record display {share_idx} when you share.")
                    return di, wi, desc, share_idx
                print(f"  Invalid choice. Enter 1-{len(options)}.")
            except ValueError:
                print(f"  Invalid input. Enter a number 1-{len(options)}.")
            except (EOFError, KeyboardInterrupt):
                print()
                return None, None, "Full screen", None


    # ── Mic + output source selection ─────────────────────────────────────────
    # Helpers (_list_audio_devices, _ca_*, _rank_input/output) are imported
    # from the transcriber module at the top of this file.

    def _device_label(d: dict) -> str:
        tags = []
        if d.get("is_bluetooth"): tags.append("Bluetooth")
        if d.get("is_usb"):       tags.append("USB")
        if d.get("is_builtin"):   tags.append("built-in")
        if d.get("is_aggregate"): tags.append("aggregate")
        if d.get("is_virtual"):   tags.append("virtual")
        if d.get("is_hdmi"):      tags.append("HDMI")
        if d.get("is_airplay"):   tags.append("AirPlay")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return f"{d['name']}{tag_str}"

    def _resolve_device_arg(kind: str, name_or_uid: str | None):
        """Resolve a --mic/--output CLI arg to a device dict. Returns None if
        empty arg, raises SystemExit with a helpful list if no match."""
        if not name_or_uid:
            return None
        devs = _list_audio_devices(kind)
        match = _find_device_by_name(devs, name_or_uid)
        if match:
            return match
        print(f"  \\033[91m\\u2717\\033[0m  No {kind} device matching '{name_or_uid}'.")
        print(f"      Available {kind} devices:")
        for d in devs:
            print(f"        - {_device_label(d)}")
        raise SystemExit(2)

    def prompt_mic_source(preselected: dict | None = None):
        """Show input devices, let user pick. Sets default input to the choice.
        Returns (dev_dict, original_default_id) so caller can restore on exit.
        """
        inputs = _list_audio_devices("input")
        if not inputs:
            return None, 0
        cur_id = _ca_get_default_device("input")
        # Sort by priority for display
        inputs_sorted = sorted(inputs, key=lambda d: (d["id"] != cur_id, _rank_input(d), d["name"]))

        if preselected is not None:
            chosen = preselected
        elif not sys.stdin.isatty():
            # Non-interactive: keep current default.
            chosen = next((d for d in inputs if d["id"] == cur_id), inputs_sorted[0])
        else:
            print()
            print(f"  \\033[1m\\U0001F3A4 Microphone source:\\033[0m")
            default_idx = 1
            for i, d in enumerate(inputs_sorted, start=1):
                marker = " \\033[92m(current default)\\033[0m" if d["id"] == cur_id else ""
                print(f"    {i})  {_device_label(d)}{marker}")
                if d["id"] == cur_id:
                    default_idx = i
            print()
            try:
                choice = input(f"  Enter number [{default_idx}]: ").strip() or str(default_idx)
                n = int(choice)
                if not (1 <= n <= len(inputs_sorted)):
                    raise ValueError
                chosen = inputs_sorted[n - 1]
            except (ValueError, EOFError, KeyboardInterrupt):
                chosen = next((d for d in inputs_sorted if d["id"] == cur_id), inputs_sorted[0])
                print(f"  \\033[90mKeeping current mic: {chosen['name']}\\033[0m")

        if chosen["id"] != cur_id:
            if _ca_set_default_device("input", chosen["id"]):
                print(f"  \\033[92m\\u2713\\033[0m  Mic set to: \\033[1m{_device_label(chosen)}\\033[0m "
                      f"\\033[90m(restored on exit)\\033[0m")
            else:
                print(f"  \\033[93m!\\033[0m  Could not switch mic to {chosen['name']}; keeping current default.")
                chosen = next((d for d in inputs if d["id"] == cur_id), chosen)
        else:
            print(f"  \\033[92m\\u2713\\033[0m  Mic: \\033[1m{_device_label(chosen)}\\033[0m")
        return chosen, cur_id

    def prompt_output_source(preselected: dict | None = None, auto: bool = True):
        """Show output devices, let user pick (or auto-pick best active device).

        auto=True: pick the highest-priority physical device automatically if the
        current default is virtual/aggregate (user has a Multi-Output or stacked
        output set but a better physical option is available). Still informs
        the user and lets them override.
        """
        outputs = _list_audio_devices("output")
        if not outputs:
            return None, 0
        cur_id = _ca_get_default_device("output")
        cur = next((d for d in outputs if d["id"] == cur_id), None)
        physical = [d for d in outputs if not d.get("is_virtual") and not d.get("is_aggregate")]
        best_physical = sorted(physical, key=lambda d: (_rank_output(d), d["name"]))[0] if physical else None

        if preselected is not None:
            chosen = preselected
        elif not sys.stdin.isatty():
            chosen = cur or (best_physical if best_physical else outputs[0])
        else:
            # Recommendation note
            if cur and (cur.get("is_virtual") or cur.get("is_aggregate")) and best_physical:
                print()
                print(f"  \\033[93m!\\033[0m  Current output is "
                      f"\\033[1m{_device_label(cur)}\\033[0m - virtual/aggregate devices")
                print(f"      force 44.1kHz and degrade Bluetooth audio. "
                      f"\\033[90m(Per-process tap makes this unnecessary.)\\033[0m")
                print(f"      Recommended: \\033[1m{_device_label(best_physical)}\\033[0m")

            outputs_sorted = sorted(outputs, key=lambda d: (d["id"] != cur_id, _rank_output(d), d["name"]))
            print()
            print(f"  \\033[1m\\U0001F50A Speaker / output source:\\033[0m")
            default_idx = 1
            for i, d in enumerate(outputs_sorted, start=1):
                tags = []
                if d["id"] == cur_id: tags.append("\\033[92mcurrent default\\033[0m")
                if best_physical and d["id"] == best_physical["id"]: tags.append("\\033[96mrecommended\\033[0m")
                tag_str = f" ({', '.join(tags)})" if tags else ""
                print(f"    {i})  {_device_label(d)}{tag_str}")
                # Default to recommended physical if current is virtual; else current.
                if best_physical and (cur is None or cur.get("is_virtual") or cur.get("is_aggregate")):
                    if d["id"] == best_physical["id"]:
                        default_idx = i
                elif d["id"] == cur_id:
                    default_idx = i
            print(f"    {len(outputs_sorted) + 1})  Keep as-is (no change)")
            print()
            try:
                choice = input(f"  Enter number [{default_idx}]: ").strip() or str(default_idx)
                n = int(choice)
                if n == len(outputs_sorted) + 1:
                    chosen = cur or outputs_sorted[0]
                elif 1 <= n <= len(outputs_sorted):
                    chosen = outputs_sorted[n - 1]
                else:
                    raise ValueError
            except (ValueError, EOFError, KeyboardInterrupt):
                chosen = cur or outputs_sorted[0]
                print(f"  \\033[90mKeeping current output: {chosen['name']}\\033[0m")

        if chosen and cur and chosen["id"] != cur_id:
            if _ca_set_default_device("output", chosen["id"]):
                print(f"  \\033[92m\\u2713\\033[0m  Output set to: \\033[1m{_device_label(chosen)}\\033[0m "
                      f"\\033[90m(restored on exit)\\033[0m")
            else:
                print(f"  \\033[93m!\\033[0m  Could not switch output to {chosen['name']}; keeping current.")
                chosen = cur
        elif chosen:
            print(f"  \\033[92m\\u2713\\033[0m  Output: \\033[1m{_device_label(chosen)}\\033[0m")
        return chosen, cur_id


    # ── MeetingSession ────────────────────────────────────────────────────────

    class MeetingSession:
        def __init__(self, template="default", enable_screen=True, defer_transcription=False):
            self.template = template
            self.enable_screen = enable_screen
            # Post-process mode: record raw segments during the call, transcribe
            # them after the user stops. Lower CPU load during demanding
            # video calls at the cost of post-meeting processing time.
            self.defer_transcription = defer_transcription
            self._deferred_segments: list[dict] = []
            self.transcript_segments: list[str] = []
            self.raw_segments: list[dict] = []
            self.screen_contexts: list[str] = []
            self.user_notes = ""
            self.start_time = datetime.now()
            self._stop_event = threading.Event()
            # Time-indexed signals from the vision model - used to rewrite
            # diarize's generic "Speaker 1/2/3" labels with real participant
            # names, and to detect screen-sharing events.
            # Each entry: (epoch_seconds, active_speaker_str, participants_list, shared_content_str)
            self.speaker_signals: list[tuple] = []
            # Set of distinct names the vision model has EVER seen. Used during
            # post-processing as the pool of candidate real names.
            self.all_participants: set = set()
            # Performance tracking (for the summary line at end of session).
            self.vision_latencies_ms: list[int] = []

        def on_transcript(self, text, seg_data):
            # Wall-clock HH:MM:SS for display + an epoch timestamp for merging
            # with speaker signals. In batch/post-process mode, seg_data
            # arrives with _recv_epoch already stamped (at recording time),
            # so we preserve it - otherwise fall back to now().
            pre_epoch = seg_data.get("_recv_epoch")
            if pre_epoch:
                ts_epoch = float(pre_epoch)
                ts_str = datetime.fromtimestamp(ts_epoch).strftime("%H:%M:%S")
            else:
                now = datetime.now()
                ts_str = now.strftime("%H:%M:%S")
                ts_epoch = now.timestamp()
                seg_data["_recv_epoch"] = ts_epoch
            speaker = seg_data.get("speaker", "Unknown")
            entry = f"[{ts_str}] [{speaker}] {seg_data.get('text', text)}"
            self.transcript_segments.append(entry)
            self.raw_segments.append(seg_data)
            if speaker == "You":
                color = "\\033[92m"
            elif speaker == "Unknown":
                color = "\\033[90m"
            else:
                color = "\\033[96m"
            print(f"  {color}\\u25b8\\033[0m {entry}", flush=True)
            try:
                live_path = TMP_DIR / "_live_transcript.txt"
                with open(live_path, "a") as f:
                    f.write(entry + "\\n")
            except OSError:
                pass

        def on_screen_context(self, context, signals=None):
            """Screen-analysis callback. New two-arg form: context text + signals dict.

            Old one-arg callers still work (signals=None). When signals are
            present we stash the active-speaker info with its timestamp so the
            final transcript can rewrite generic diarize labels ("Speaker 2")
            with real names ("Moneesh Reddy") via nearest-time matching.
            """
            ts_str = datetime.now().strftime("%H:%M:%S")
            if signals:
                self.speaker_signals.append((
                    signals.get("ts_recorded", datetime.now().timestamp()),
                    signals.get("active_speaker", "") or "",
                    list(signals.get("participants", []) or []),
                    signals.get("shared_content", "") or "",
                ))
                for p in signals.get("participants", []) or []:
                    self.all_participants.add(p)
                ms = signals.get("inference_ms", 0)
                if ms:
                    self.vision_latencies_ms.append(ms)
                # More-useful progress line: cadence + who is speaking if known.
                active = signals.get("active_speaker") or ""
                shared = signals.get("shared_content") or ""
                tail = (f" \\u2014 active: {active}" if active else "")
                if shared:
                    tail += f" \\u2014 share: {shared[:30]}"
                print(f"  \\033[94m\\u25c9\\033[0m {ts_str}  vision {ms}ms{tail}", flush=True)
            else:
                print(f"  \\033[94m\\u25c9\\033[0m Screen analysed at {ts_str}", flush=True)
            self.screen_contexts.append(f"[{ts_str}] {context}")


        # ── Speaker-name post-processing ──────────────────────────────────────
        def _rewrite_speaker_labels(self):
            """Walk raw_segments and rewrite generic diarize labels to real names.

            Generic labels: 'Speaker 0', 'SPEAKER_00', 'Unknown', 'Other', etc.
            Real names come from the vision model's active-speaker + participant
            signals, matched by nearest timestamp with a vote-based majority.

            When the vision model detects participants but no clear active_speaker,
            we still associate the participants list with nearby segments - if only
            one non-"You" participant is identified across the meeting, all generic
            labels map to that person.

            Returns the number of labels that were actually rewritten.
            """
            if not self.speaker_signals or not self.raw_segments:
                return 0
            import bisect
            import re as _re

            # Index signals by epoch for binary search.
            sigs = sorted(self.speaker_signals, key=lambda s: s[0])
            sig_ts = [s[0] for s in sigs]
            sig_active = [s[1] for s in sigs]       # active_speaker string
            sig_parts  = [s[2] for s in sigs]       # participants list

            # Collect ALL unique participant names seen across the meeting.
            all_names = set()
            for parts in sig_parts:
                for p in (parts or []):
                    if p and p.strip():
                        all_names.add(p.strip())

            def is_generic(label):
                if not label:
                    return True
                l = label.strip()
                if _re.match(r"(?i)^speaker[_ ]?\\d+$", l):
                    return True
                if _re.match(r"(?i)^SPEAKER_\\d+$", l):
                    return True
                if l in ("Unknown", "Other", ""):
                    return True
                return False

            # Build a vote table: for each generic speaker ID, count which real
            # names the vision model reported as active at the same time.
            votes = {}   # generic_id -> dict[real_name] -> count
            for seg in self.raw_segments:
                sp = seg.get("speaker", "")
                if not is_generic(sp):
                    continue
                t = seg.get("_recv_epoch", 0)
                if not t or not sig_ts:
                    continue
                # Find the nearest vision signal.
                i = bisect.bisect_left(sig_ts, t)
                cand = []
                if i > 0:
                    cand.append(i - 1)
                if i < len(sig_ts):
                    cand.append(i)
                nearest = min(cand, key=lambda j: abs(sig_ts[j] - t))
                # Only match if within 30s - beyond that the signal is stale.
                if abs(sig_ts[nearest] - t) > 30:
                    continue
                name = sig_active[nearest]
                gid = sp or "Unknown"
                if name:
                    votes.setdefault(gid, {})
                    votes[gid][name] = votes[gid].get(name, 0) + 1
                elif sig_parts[nearest]:
                    # No clear active speaker, but we know who's in the call.
                    # Give each participant a fractional vote.
                    for p in sig_parts[nearest]:
                        if p and p.strip():
                            votes.setdefault(gid, {})
                            votes[gid][p.strip()] = votes[gid].get(p.strip(), 0) + 0.5

            # Collapse votes to a single best name per generic_id.
            mapping = {}
            for gid, tally in votes.items():
                if not tally:
                    continue
                sorted_names = sorted(tally.items(), key=lambda kv: -kv[1])
                if len(sorted_names) == 1:
                    mapping[gid] = sorted_names[0][0]
                elif sorted_names[0][1] >= 1.5 * sorted_names[1][1]:
                    mapping[gid] = sorted_names[0][0]

            # Fallback: if we have exactly one non-"You" participant from vision
            # and some generic labels remain unmapped, assign them all to that person.
            mapped_names = set(mapping.values())
            unmapped_gids = [gid for gid in votes if gid not in mapping]
            if unmapped_gids and len(all_names - mapped_names) == 1:
                remaining = (all_names - mapped_names).pop()
                for gid in unmapped_gids:
                    mapping[gid] = remaining

            rewritten = 0
            for idx, seg in enumerate(self.raw_segments):
                sp = seg.get("speaker", "")
                if sp in mapping:
                    new = mapping[sp]
                    seg["speaker"] = new
                    # Update the displayed transcript_segments entry too.
                    line = self.transcript_segments[idx] if idx < len(self.transcript_segments) else ""
                    if line:
                        self.transcript_segments[idx] = line.replace(f"[{sp}]", f"[{new}]", 1)
                    rewritten += 1
            if mapping:
                print(f"  \\033[92m\\u2713\\033[0m  Speaker mapping: {mapping}")
            return rewritten

        def start_follow_watchdog(self, pid, label="selected app", poll_s=2.0, debounce=2):
            """Stop the session when the selected app's PID exits.

            If the user quits the selected call window (e.g. Zoom) without
            hitting Ctrl+C, the recorder otherwise keeps running and produces
            a transcript full of silence. This polls the owner PID with
            os.kill(pid, 0) every poll_s seconds and, after `debounce`
            consecutive ESRCH hits, sets _stop_event so the transcription
            loop exits cleanly and notes are generated as if Ctrl+C was hit.

            Debounce absorbs the brief window where a process has exited but
            its PID has not yet been reaped (zombie) on macOS. EPERM means
            the process exists but we can't signal it - treat as alive.
            """
            try:
                pid = int(pid) if pid is not None else None
            except (TypeError, ValueError):
                pid = None
            if not pid or pid <= 1:
                return False

            import errno as _errno

            def _alive(p: int) -> bool:
                try:
                    os.kill(p, 0)
                    return True
                except ProcessLookupError:
                    return False
                except PermissionError:
                    return True  # exists, we just can't signal it
                except OSError as e:
                    # ESRCH = no such process; anything else: assume alive to
                    # avoid stopping the session on a transient error.
                    return getattr(e, "errno", None) != _errno.ESRCH

            if not _alive(pid):
                print(f"  \\033[93m!\\033[0m  {label} (pid {pid}) is not running; "
                      f"watchdog not started.")
                return False

            def _watch():
                misses = 0
                while not self._stop_event.is_set():
                    if _alive(pid):
                        misses = 0
                    else:
                        misses += 1
                        if misses >= debounce:
                            print(f"\\n  \\033[93m\\u25c6\\033[0m  {label} (pid {pid}) "
                                  f"has closed. Stopping transcription and "
                                  f"generating notes...", flush=True)
                            self._stop_event.set()
                            return
                    # Wait with interruptible sleep so Ctrl+C still feels snappy.
                    if self._stop_event.wait(timeout=poll_s):
                        return

            t = threading.Thread(target=_watch, name="follow-watchdog", daemon=True)
            t.start()
            self._follow_watchdog = t
            return True

        def start_transcription(self, backend, follow_pids=None):
            stream_transcribe(
                backend=backend,
                callback=self.on_transcript,
                stop_event=self._stop_event,
                follow_pids=follow_pids,
                defer_transcription=self.defer_transcription,
                deferred_out=self._deferred_segments if self.defer_transcription else None,
            )

        def run_deferred_transcription(self):
            """Post-process: transcribe + diarize all segments recorded during a
            post-process session. Each segment fires `on_transcript` with the
            original recording epoch so timestamps in the final transcript
            reflect meeting time, not post-process time.
            """
            from transcriber import _transcribe_segment
            pending = list(self._deferred_segments)
            n = len(pending)
            if n == 0:
                return 0
            print(f"  Transcribing {n} buffered segment(s) "
                  f"(\\u2248{(n * 5) // 60}m{(n * 5) % 60}s of audio)...")
            done = 0
            for rec in pending:
                try:
                    _transcribe_segment(
                        rec["sys_wav"], rec["mic_wav"], rec["two_tracks"],
                        self.on_transcript, recv_epoch=rec.get("recv_epoch"),
                    )
                except Exception as e:
                    print(f"  \\033[93m!\\033[0m  Segment failed ({e}); continuing.")
                done += 1
                # Progress line every 5 segments or on the last one.
                if done % 5 == 0 or done == n:
                    pct = int(done * 100 / n)
                    print(f"  \\033[90m\\u00b7\\033[0m Post-process: {done}/{n} ({pct}%)", flush=True)
            self._deferred_segments.clear()
            return done

        def start_screen_capture(self, interval=2, display_index=None, window_id=None, share_display_index=None):
            """Start screen capture.

              - `display_index` / `window_id`: what muneem FOLLOWS (the call window
                usually) - this feeds the vision model so screen context shows up
                alongside the transcript.
              - `share_display_index`: if set, muneem ALSO records frames from
                that display every CAPTURE_INTERVAL seconds into
                ~/.muneem/tmp/share/ - representing what the user is sharing
                to the call. Those frames are passed through the same vision
                analysis pipeline so the notes include the shared content.

            When both are set, muneem interleaves the two streams into a single
            screen-context timeline, labelled "(follow)" and "(share)".
            """
            _ensure_dir()
            # Primary follow capture
            cap = threading.Thread(
                target=_capture_loop,
                args=(interval, self._stop_event),
                kwargs={"display_index": display_index, "window_id": window_id},
                daemon=True,
            )
            ana = threading.Thread(target=_analysis_loop, args=(self.on_screen_context, self._stop_event), daemon=True)
            cap.start()
            ana.start()

            # Share capture - user's display that they're mirroring to the call.
            if share_display_index is not None:
                # Use a separate latest-frame file so the two streams don't race.
                share_dir = str(Path(SCREENSHOT_DIR).parent / "share")
                os.makedirs(share_dir, exist_ok=True)
                share_frame = os.path.join(share_dir, "latest.png")
                share_tmp = os.path.join(share_dir, "latest_tmp.png")

                def _share_capture_loop():
                    fail_count = 0
                    while not self._stop_event.is_set():
                        cmd = ["screencapture", "-x", "-D", str(share_display_index), share_tmp]
                        try:
                            subprocess.run(cmd, capture_output=True, timeout=5)
                        except Exception:
                            fail_count += 1
                            self._stop_event.wait(interval)
                            continue
                        if os.path.exists(share_tmp) and os.path.getsize(share_tmp) > 0:
                            os.replace(share_tmp, share_frame)
                            fail_count = 0
                        self._stop_event.wait(interval)

                def _share_analysis_loop():
                    """Analyze the share display at ANALYSIS_INTERVAL cadence, tag
                    the result as '(share)' so notes can tell the two apart."""
                    prompt = (
                        "This is the content the user is SHARING to the meeting. "
                        "In 2-3 sentences, describe what app, document, or screen "
                        "content is visible. Note any key text, chart labels, code "
                        "identifiers, or headings that will matter for meeting notes."
                    )
                    import requests as _rq, base64 as _b64
                    while not self._stop_event.is_set():
                        if not os.path.exists(share_frame):
                            self._stop_event.wait(1)
                            continue
                        try:
                            with open(share_frame, "rb") as f:
                                img_b64 = _b64.b64encode(f.read()).decode("utf-8")
                            r = _rq.post(
                                f"{OLLAMA_URL}/api/generate",
                                json={
                                    "model": VISION_MODEL,
                                    "prompt": prompt,
                                    "images": [img_b64],
                                    "stream": False,
                                },
                                timeout=120,
                            )
                            r.raise_for_status()
                            text = r.json().get("response", "")
                            if text:
                                ts = datetime.now().strftime("%H:%M:%S")
                                self.screen_contexts.append(f"[{ts}] (share) {text}")
                                print(f"  \\033[94m\\u25c9\\033[0m Share display analysed at {ts}", flush=True)
                        except Exception as e:
                            pass
                        self._stop_event.wait(ANALYSIS_INTERVAL)

                scap = threading.Thread(target=_share_capture_loop, daemon=True)
                sana = threading.Thread(target=_share_analysis_loop, daemon=True)
                scap.start()
                sana.start()

        def get_full_transcript(self):
            return "\\n".join(self.transcript_segments)

        def get_screen_summary(self):
            return "\\n\\n".join(self.screen_contexts)

        def save_raw(self):
            ts = self.start_time.strftime("%Y%m%d_%H%M%S")
            raw = NOTES_DIR / f"{ts}_raw_transcript.md"
            raw.write_text(
                f"# Raw Transcript -- {self.start_time.strftime('%Y-%m-%d %H:%M')}\\n\\n"
                + self.get_full_transcript()
                + "\\n\\n---\\n\\n## Screen Context\\n\\n"
                + self.get_screen_summary()
            )
            return raw


        def save_transcript(self):
            """Full speaker-grouped transcript: one file with every spoken line,
            grouped by speaker, with a header block showing participants +
            session metadata. This is the "play-by-play" artifact a reader can
            scrub through to find exact quotes.
            """
            ts = self.start_time.strftime("%Y%m%d_%H%M%S")
            end_time = datetime.now()
            duration_sec = int((end_time - self.start_time).total_seconds())
            mins = duration_sec // 60
            secs = duration_sec % 60

            participants = sorted(self.all_participants) if self.all_participants else []
            speakers_in_text = sorted({(s.get("speaker") or "Unknown") for s in self.raw_segments})
            header = (
                f"# Transcript \\u2014 {self.start_time.strftime('%Y-%m-%d %H:%M')}\\n\\n"
                f"**Duration:** {mins}m {secs}s  \\n"
                f"**Audio segments:** {len(self.raw_segments)}  \\n"
                f"**Vision analyses:** {len(self.speaker_signals)}  "
            )
            if self.vision_latencies_ms:
                avg_ms = sum(self.vision_latencies_ms) // len(self.vision_latencies_ms)
                header += f"(avg latency {avg_ms}ms)  \\n"
            else:
                header += "\\n"
            if participants:
                header += f"**Participants (detected from screen):** {', '.join(participants)}  \\n"
            if speakers_in_text:
                header += f"**Speakers in transcript:** {', '.join(speakers_in_text)}  \\n"
            header += "\\n---\\n\\n"

            # Body: group consecutive segments from the same speaker into one
            # paragraph - makes the transcript skimmable.
            body_lines = []
            last_speaker = None
            pending_parts = []
            def flush_pending():
                if pending_parts and last_speaker is not None:
                    text = " ".join(p.strip() for p in pending_parts if p.strip())
                    if text:
                        body_lines.append(f"**{last_speaker}** \\u2014 `{pending_first_ts}`")
                        body_lines.append("")
                        body_lines.append(text)
                        body_lines.append("")

            pending_first_ts = ""
            for seg in self.raw_segments:
                sp = seg.get("speaker", "Unknown") or "Unknown"
                txt = (seg.get("text") or "").strip()
                ts_epoch = seg.get("_recv_epoch")
                ts_display = ""
                if ts_epoch:
                    ts_display = datetime.fromtimestamp(ts_epoch).strftime("%H:%M:%S")
                if sp != last_speaker:
                    flush_pending()
                    last_speaker = sp
                    pending_parts = []
                    pending_first_ts = ts_display
                pending_parts.append(txt)
            flush_pending()

            transcript_body = "\\n".join(body_lines) if body_lines else "*(no spoken content captured)*"

            # Screen-context timeline appended - chronological record of what was
            # visible. Useful for jumping back to "what slide was showing when X said Y".
            screen_block = ""
            if self.screen_contexts:
                screen_block = "\\n\\n---\\n\\n## Screen Timeline\\n\\n" + "\\n\\n".join(self.screen_contexts)

            provenance = (
                f"<!-- muneem provenance: "
                f"whisper={WHISPER_MODEL} llm={ENHANCE_MODEL} vision={VISION_MODEL} "
                f"generated={ts} -->\\n"
            )
            p = NOTES_DIR / f"{ts}_transcript.md"
            p.write_text(provenance + header + transcript_body + screen_block + "\\n")
            return p


        def save_enhanced(self, notes):
            ts = self.start_time.strftime("%Y%m%d_%H%M%S")
            is_screen_only = (not self.transcript_segments) and bool(self.screen_contexts)
            fname_stem = f"{ts}_screen_only_notes" if is_screen_only else f"{ts}_meeting_notes"
            p = NOTES_DIR / f"{fname_stem}.md"
            provenance = (
                f"<!-- muneem provenance: "
                f"whisper={WHISPER_MODEL} llm={ENHANCE_MODEL} vision={VISION_MODEL} "
                f"generated={ts} -->\\n"
            )
            banner = ""
            if is_screen_only:
                banner = (
                    "> \\u26a0\\ufe0f  **SCREEN-ONLY SESSION** - No audio transcript captured.\\n"
                    "> These notes were inferred purely from on-screen context.\\n"
                    "> Verify details before relying on them.\\n\\n"
                )
            p.write_text(provenance + banner + notes)
            return p


    # ── Subcommand: start ─────────────────────────────────────────────────────

    def cmd_start(args):
        # ── Step 1: auto-configure audio (every start) ──
        print()
        print("  \\033[1m\\u266b Audio configuration\\033[0m")

        # ── Step 1a: mic + output selection BEFORE preflight ──
        # Resolve --mic / --output args first (fail fast with device list on typo),
        # then prompt interactively if not supplied. Any device-default changes
        # made here are tracked in _audio_restore so we can revert on exit.
        _audio_restore = {"input": 0, "output": 0}
        try:
            preselected_mic = _resolve_device_arg("input", getattr(args, "mic", None))
            preselected_out = _resolve_device_arg("output", getattr(args, "output", None))
        except SystemExit:
            raise

        skip_audio_prompt = getattr(args, "no_audio_prompt", False)
        if skip_audio_prompt and preselected_mic is None and preselected_out is None:
            # User wants zero prompts: use whatever the system defaults are.
            pass
        else:
            _mic_dev, _mic_orig = prompt_mic_source(preselected=preselected_mic)
            _out_dev, _out_orig = prompt_output_source(preselected=preselected_out)
            _audio_restore["input"]  = _mic_orig if (_mic_dev and _mic_dev.get("id") != _mic_orig) else 0
            _audio_restore["output"] = _out_orig if (_out_dev and _out_dev.get("id") != _out_orig) else 0

        # Ensure originals get restored on any exit path (Ctrl+C, exception,
        # normal termination). atexit runs after sys.exit and KeyboardInterrupt.
        import atexit as _atexit
        def _restore_audio_defaults():
            try:
                if _audio_restore.get("input"):
                    _ca_set_default_device("input", _audio_restore["input"])
                if _audio_restore.get("output"):
                    _ca_set_default_device("output", _audio_restore["output"])
            except Exception:
                pass
        _atexit.register(_restore_audio_defaults)

        # ── Step 1b: preflight (output-dependent; runs AFTER selection) ──
        pf = audio_preflight(verbose=True)
        backend = pf["backend"]
        backend_label = {
            "native":    "Core Audio Tap",
            "blackhole": "BlackHole",
            "mic_only":  "Microphone only",
        }.get(backend, backend)

        # Screen capture: ON by default when a call app is detected.
        # Identifies speaker names via vision model → maps to diarized audio.
        # --no-screen explicitly disables.
        explicit_target = (getattr(args, "display", None) is not None
                           or getattr(args, "window", None) is not None
                           or getattr(args, "window_id", None) is not None)
        if args.no_screen:
            enable_screen = False
        elif args.screen or explicit_target:
            enable_screen = True
        else:
            # Auto-detect: enable screen if a call app is running.
            try:
                call_apps = get_call_windows()
                enable_screen = bool(call_apps)
                if enable_screen:
                    app_names = ", ".join(sorted({w["label"] for w in call_apps[:5]}))
                    print(f"  \\033[92m\\u2713\\033[0m  Auto-detected call app(s): {app_names}")
                    print(f"      Screen capture ON (speaker name identification via vision model).")
                    print(f"      Use \\033[1m--no-screen\\033[0m to disable.\\n")
            except Exception:
                enable_screen = False

        if not _preflight(require_vision=enable_screen):
            return

        # ── Step 2: determine capture target (call apps + displays only) ──
        display_index = getattr(args, "display", None)
        window_id = getattr(args, "window_id", None)
        window_name = getattr(args, "window", None)
        share_display_index = getattr(args, "record_share", None)   # --record-share=N
        screen_desc = "OFF"

        if enable_screen:
            if window_name and not window_id:
                available = get_call_windows()
                for w in available:
                    if window_name.lower() in w["label"].lower() or window_name.lower() in w["owner"].lower():
                        window_id = w["id"]
                        screen_desc = f"{w['label']} \\u2014 {w['title']}"
                        break
                if not window_id:
                    for w in available:
                        if window_name.lower() in (w.get("title") or "").lower():
                            window_id = w["id"]
                            screen_desc = f"{w['label']} \\u2014 {w['title']}"
                            break
                if window_id:
                    print(f"  \\033[92m\\u2713\\033[0m  Matched call app: {screen_desc}")
                else:
                    print(f"  \\033[93m!\\033[0m  No call app matching \\"{window_name}\\" is running.")
                    if available:
                        print("      Running call apps:")
                        for w in available[:10]:
                            print(f"        - {w['label']} \\u2014 {w['title']}")
                    print("      Falling back to interactive selection.")
            if display_index is not None and window_id is None:
                screen_desc = f"Screen {display_index}"
            if display_index is None and window_id is None:
                display_index, window_id, screen_desc, picked_share = prompt_capture_target()
                if share_display_index is None:
                    share_display_index = picked_share
            if enable_screen and screen_desc == "OFF":
                screen_desc = "Full screen"

        # ── Step 2b: resolve follow-window PID for per-process Core Audio Tap ──
        # This is the critical fix for Bluetooth output + browser-based calls.
        # Global Core Audio Taps return zeros when output is Bluetooth (macOS
        # limitation); per-process taps bypass that by intercepting the target
        # app's audio stream directly. For Chromium-based browsers (Arc,
        # Chrome, Edge), audio is produced by a renderer subprocess, so we
        # also include ALL descendant PIDs of the window-owner.
        follow_pids = None
        # Hoisted so the follow-app watchdog (set up below) can see the owner
        # PID after this try/except block. Without this the name would only
        # exist inside the try and be unreachable later.
        follow_owner_pid = None
        follow_owner_label = None
        if window_id is not None:
            try:
                # Find the window owner's PID.
                all_wins = get_call_windows() + get_windows()
                owner_pid = None
                for w in all_wins:
                    if w.get("id") == window_id:
                        owner_pid = w.get("pid")
                        follow_owner_label = (w.get("label") or w.get("owner") or "selected app")
                        break
                if owner_pid:
                    follow_owner_pid = int(owner_pid)
                    # Collect descendants (Chromium renderers are children of
                    # the browser main process and produce the actual audio).
                    pids = {int(owner_pid)}
                    try:
                        import subprocess as _sp
                        r = _sp.run(
                            ["ps", "-o", "pid,ppid", "-eA"],
                            capture_output=True, text=True, timeout=5,
                        )
                        children = {}
                        for ln in (r.stdout or "").splitlines()[1:]:
                            parts = ln.split()
                            if len(parts) < 2: continue
                            try:
                                cp, pp = int(parts[0]), int(parts[1])
                            except ValueError:
                                continue
                            children.setdefault(pp, []).append(cp)
                        stack = [int(owner_pid)]
                        while stack:
                            p = stack.pop()
                            for c in children.get(p, []):
                                if c not in pids:
                                    pids.add(c); stack.append(c)
                    except Exception:
                        pass
                    follow_pids = sorted(pids)
                    print(f"  \\033[92m\\u2713\\033[0m  Per-process tap targets "
                          f"{len(follow_pids)} PID(s) (owner={owner_pid}): "
                          f"{follow_pids[:6]}{'\\u2026' if len(follow_pids) > 6 else ''}")
            except Exception as exc:
                print(f"  \\033[93m!\\033[0m  Could not resolve PIDs for per-process tap: {exc}")

        # ── Step 2c: override backend when per-process tap is available ──
        # CRITICAL: per-process taps BYPASS output device routing entirely,
        # so the stacked/multi-output / Bluetooth limitations that force the
        # BlackHole fallback don't apply. If follow_pids are resolved AND
        # the native helper is present, ALWAYS use native - even when the
        # default output is a Multi-Output Device.
        # Without this override, users who set up a Multi-Output Device
        # (BlackHole + headphones) get silent transcripts because the
        # preflight picks BlackHole and BlackHole isn't actually in the
        # routing path for the target app.
        if follow_pids and _has_native_helper() and backend != "native":
            prev_backend = backend
            backend = "native"
            backend_label = "Core Audio Tap (per-process)"
            print(f"  \\033[92m\\u2713\\033[0m  Switching backend: "
                  f"\\033[90m{prev_backend}\\033[0m \\u2192 "
                  f"\\033[1mper-process Core Audio Tap\\033[0m "
                  f"(bypasses output routing).")
            dev_name_tmp = (pf.get("output_device", {}).get("name") or "").strip()
            if pf.get("output_device", {}).get("is_virtual"):
                print(f"      \\033[90mYou do NOT need a Multi-Output Device for this to work.\\033[0m")
                print(f"      \\033[90mFor best audio quality, set your output directly to\\033[0m")
                print(f"      \\033[90mBluetooth earbuds / headphones (avoids forced 44.1kHz).\\033[0m")

        # ── Step 3: session banner ──
        scr = f"{screen_desc} (every {args.screen_interval}s)" if enable_screen else "OFF"
        share_line = (f"Screen-share: display {share_display_index} (recorded separately)"
                      if share_display_index else "Screen-share: not recorded")
        dev_name = (pf.get("output_device", {}).get("name") or "").strip()
        audio_line = f"{backend_label}"
        if dev_name:
            audio_line += f" (output: {dev_name[:28]})"
        print()
        print("\\u2554" + "\\u2550" * 58 + "\\u2557")
        print("\\u2551              MUNEEM -- Offline AI Notepad                \\u2551")
        print("\\u2551      Offline local capture (macOS, Apple Silicon)        \\u2551")
        print("\\u2560" + "\\u2550" * 58 + "\\u2563")
        print(f"\\u2551  Template:     {args.template:<41}\\u2551")
        print(f"\\u2551  Follow:       {scr[:41]:<41}\\u2551")
        print(f"\\u2551  {share_line[:55]:<55} \\u2551")
        print(f"\\u2551  Audio:        {audio_line[:41]:<41}\\u2551")
        print(f"\\u2551  LLM:          {ENHANCE_MODEL:<41}\\u2551")
        print(f"\\u2551  Diarization:  {'ON (sherpa-onnx, stable speaker IDs)':<41}\\u2551")
        mode_line = ("Post-process (transcribe after stop)"
                     if getattr(args, "post_process", False) else "Real-time (transcribe during call)")
        print(f"\\u2551  Mode:         {mode_line[:41]:<41}\\u2551")
        print("\\u2551                                                          \\u2551")
        print("\\u2551  Press Ctrl+C to stop and generate notes.                \\u2551")
        if follow_owner_pid:
            _stop_line = (f"  (or close {(follow_owner_label or 'the call')[:38]} - auto-stops)")
            print(f"\\u2551{_stop_line[:58]:<58}\\u2551")
        print("\\u255a" + "\\u2550" * 58 + "\\u255d")
        print()

        # Clear any speaker identities carried over from a previous session -
        # the CAM++ embedding space only has meaning within a single meeting
        # (different rooms, different people, different acoustics).
        reset_speaker_registry()

        session = MeetingSession(
            template=args.template,
            enable_screen=enable_screen,
            defer_transcription=getattr(args, "post_process", False),
        )

        # Auto-stop transcription when the selected call app/window exits.
        # Only armed when the user picked a specific window (so the watchdog
        # has a concrete PID to follow) - omitted for whole-display capture.
        if follow_owner_pid:
            started = session.start_follow_watchdog(
                follow_owner_pid,
                label=follow_owner_label or "selected app",
            )
            if started:
                print(f"  \\033[92m\\u2713\\033[0m  Follow-app watchdog armed: "
                      f"transcription will stop automatically if "
                      f"{follow_owner_label or 'the selected app'} "
                      f"(pid {follow_owner_pid}) exits.")

        if enable_screen:
            session.start_screen_capture(
                interval=args.screen_interval,
                display_index=display_index,
                window_id=window_id,
                share_display_index=share_display_index,
            )
            print(f"  \\033[92m\\u2713\\033[0m  Screen capture started: {screen_desc} (every {args.screen_interval}s).")
            if share_display_index:
                print(f"  \\033[92m\\u2713\\033[0m  Screen-share recording: display {share_display_index} (every {args.screen_interval}s).")

        if session.defer_transcription:
            print("  \\033[92m\\u2713\\033[0m  Recording started (batch mode - transcription runs after stop).\\n")
        else:
            print("  \\033[92m\\u2713\\033[0m  Transcription starting (pipelined, ~30s segments, no gaps)...\\n")

        try:
            session.start_transcription(backend=backend, follow_pids=follow_pids)
        except KeyboardInterrupt:
            pass

        session._stop_event.set()

        # Keep _live_transcript.txt in place until the enhanced note is saved,
        # so it survives as crash-recovery if enhancement fails or is interrupted.
        live_path = TMP_DIR / "_live_transcript.txt"

        print("\\n")
        print("\\u2550" * 59)
        print("  Meeting ended. Processing notes...")
        print("\\u2550" * 59)

        # Batch-mode post-processing: transcribe + diarize everything we
        # buffered during the call. This is the CPU-heavy part the user
        # explicitly asked to defer - runs AFTER the call ends.
        if session.defer_transcription and session._deferred_segments:
            print()
            print(f"  \\033[1mPost-process mode:\\033[0m transcribing buffered audio now "
                  f"({len(session._deferred_segments)} segment(s)).")
            try:
                session.run_deferred_transcription()
            except KeyboardInterrupt:
                print("\\n  \\033[93m!\\033[0m  Post-process interrupted - partial transcript preserved.")
            print()

        # ── Post-process: map diarize's "Speaker 1/2/3" to real names
        # using signals from the vision model. Done BEFORE saving so the
        # transcript, raw dump, and summary all use consistent names.
        if session.speaker_signals and session.raw_segments:
            n_rewritten = session._rewrite_speaker_labels()
            if n_rewritten:
                print(f"  \\033[92m\\u2713\\033[0m  Speaker names: rewrote {n_rewritten} segment labels "
                      f"using vision-model active-speaker signals.")

        raw_path = session.save_raw()
        print(f"  \\033[92m\\u2713\\033[0m  Raw transcript saved: {raw_path}")

        # Full speaker-grouped transcript (separate from the summary).
        transcript_path = session.save_transcript()
        print(f"  \\033[92m\\u2713\\033[0m  Full transcript saved: {transcript_path}")

        if not session.transcript_segments and not session.screen_contexts:
            print("  No transcript or screen context captured. Exiting.")
            try:
                live_path.unlink(missing_ok=True)
            except OSError:
                pass
            return

        if not session.transcript_segments:
            print("  \\033[93m!\\033[0m  No audio transcript captured (silent meeting). Using screen context only.")

        print(f"  Generating enhanced notes (template: {args.template})...")
        print("  This may take 30-60 seconds...\\n")

        enhanced = enhance_notes(
            transcript=session.get_full_transcript(),
            screen_context=session.get_screen_summary(),
            user_notes=session.user_notes,
            template=args.template,
        )
        note_path = session.save_enhanced(enhanced)
        print(enhanced)
        print()
        print(f"  \\033[92m\\u2713\\033[0m  Summary:    {note_path}")
        print(f"  \\033[92m\\u2713\\033[0m  Transcript: {transcript_path}")
        print(f"  \\033[90m    (raw: {raw_path.name})\\033[0m")

        # Enhanced note is now safely persisted. Remove the live transcript temp file.
        try:
            live_path.unlink(missing_ok=True)
        except OSError:
            pass

        print("\\n  --- Ask questions about this meeting (type 'quit' to exit) ---\\n")
        transcript = session.get_full_transcript()
        while True:
            try:
                q = input("  You: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if not q:
                    continue
                ans = chat_with_transcript(transcript, q)
                print(f"\\n  AI: {ans}\\n")
            except (KeyboardInterrupt, EOFError):
                break
        print(f"\\n  Done. Notes saved to {NOTES_DIR}/")


    # ── Subcommand: notes ─────────────────────────────────────────────────────

    def cmd_notes(args):
        summaries = sorted(
            list(NOTES_DIR.glob("*_meeting_notes.md"))
            + list(NOTES_DIR.glob("*_screen_only_notes.md")),
            reverse=True,
        )
        transcripts = sorted(NOTES_DIR.glob("*_transcript.md"), reverse=True)
        if not summaries and not transcripts:
            print("  No meeting notes found yet. Run 'muneem start' first.")
            return

        sub = args.sub

        if sub == "last":
            if summaries:
                latest = summaries[0]
                print(f"  Opening summary: {latest.name}")
                subprocess.run(["open", str(latest)])
            else:
                print("  No summary found.")
            return

        if sub == "transcript":
            if transcripts:
                latest = transcripts[0]
                print(f"  Opening transcript: {latest.name}")
                subprocess.run(["open", str(latest)])
            else:
                print("  No transcripts found.")
            return

        print(f"\\n  Notes in {NOTES_DIR}/:\\n")
        print(f"  \\033[1mSummaries:\\033[0m")
        for i, n in enumerate(summaries[:20], 1):
            size = n.stat().st_size
            print(f"  {i:>3}.  {n.name}  ({size:,} bytes)")
        if transcripts:
            print()
            print(f"  \\033[1mTranscripts (speaker-grouped):\\033[0m")
            for i, t in enumerate(transcripts[:20], 1):
                size = t.stat().st_size
                print(f"  {i:>3}.  {t.name}  ({size:,} bytes)")
        print()
        print(f"  \\033[90m  muneem notes last        \\u2192 open most recent summary\\033[0m")
        print(f"  \\033[90m  muneem notes transcript  \\u2192 open most recent transcript\\033[0m")


    # ── Subcommand: ask ───────────────────────────────────────────────────────

    def cmd_ask(args):
        raws = sorted(NOTES_DIR.glob("*_raw_transcript.md"), reverse=True)
        if not raws:
            print("  No transcripts found. Run 'muneem start' first.")
            return

        latest = raws[0]
        transcript = latest.read_text()
        question = " ".join(args.question)
        if not question:
            print("  Usage: muneem ask \\"your question here\\"")
            return

        # Warn when transcript is likely to be silently truncated by Ollama's
        # default context window. 14B/32B ship with ~8k-16k token windows;
        # ~4 chars per token => ~32k-64k chars. Use 30k as a conservative
        # threshold so we nudge the user before answers start missing detail.
        ASK_CONTEXT_CHAR_WARN = 30000
        char_count = len(transcript)
        if char_count > ASK_CONTEXT_CHAR_WARN:
            print(
                f"  \\033[93m!\\033[0m  Transcript is large ({char_count:,} chars "
                f"\\u2248 {char_count // 4:,} tokens). The model may only see a "
                f"portion; recent content is prioritised. Consider narrowing your "
                f"question or asking about a specific section."
            )

        print(f"  Searching transcript: {latest.name}...")
        answer = chat_with_transcript(transcript, question)
        print(f"\\n  {answer}")


    # ── Subcommand: status ────────────────────────────────────────────────────

    def cmd_status(args):
        print("\\n  Muneem Status")
        print("  " + "\\u2500" * 50)

        try:
            models = _get_available_models()
            print(f"  Ollama:         \\033[92mrunning\\033[0m ({len(models)} models)")
            for required in [ENHANCE_MODEL, VISION_MODEL]:
                st = "\\033[92mready\\033[0m" if _has_model(required, models) else "\\033[91mmissing\\033[0m"
                print(f"    {required:<20} {st}")
        except Exception:
            print("  Ollama:         \\033[91mnot running\\033[0m  (start with: ollama serve)")

        native = _has_native_helper()
        bh = _has_blackhole()
        if native:
            print(f"  Audio backend:  \\033[92mCore Audio Tap\\033[0m (native)")
        elif bh:
            print(f"  Audio backend:  \\033[93mBlackHole\\033[0m (fallback)")
        else:
            print(f"  Audio backend:  \\033[91mNone\\033[0m (mic only)")

        if bh:
            print(f"  BlackHole:      \\033[92minstalled\\033[0m")
        else:
            print(f"  BlackHole:      \\033[93mnot found\\033[0m")

        # Diarization: prefer sherpa-onnx (stable speaker IDs across segments
        # via CAM++ embeddings); fall back to `diarize` package if models or
        # the sherpa_onnx package are missing.
        import importlib
        _so_ok = False
        try:
            importlib.import_module("sherpa_onnx")
            _so_ok = True
        except ImportError:
            pass
        _seg  = MUNEEM_HOME / "models" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
        _emb  = MUNEEM_HOME / "models" / "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
        _models_ok = _seg.exists() and _emb.exists()
        if _so_ok and _models_ok:
            print(f"  Diarization:    \\033[92msherpa-onnx\\033[0m (pyannote-seg + 3D-Speaker CAM++, Apache-2.0)")
        else:
            # Fallback diagnostic
            try:
                importlib.import_module("diarize")
                missing = []
                if not _so_ok: missing.append("sherpa-onnx (pip)")
                if not _models_ok: missing.append("ONNX models (re-run muneem-setup.py)")
                print(f"  Diarization:    \\033[93m`diarize` fallback\\033[0m "
                      f"(missing: {', '.join(missing) or '-'})")
            except ImportError:
                print(f"  Diarization:    \\033[91mnot available\\033[0m "
                      "(run muneem-setup.py to install sherpa-onnx)")

        notes = list(NOTES_DIR.glob("*_meeting_notes.md")) + list(NOTES_DIR.glob("*_screen_only_notes.md"))
        print(f"  Notes:          {len(notes)} saved in {NOTES_DIR}/")
        print()


    # ── Subcommand: doctor ────────────────────────────────────────────────────

    def cmd_doctor(args):
        print("\\n  Muneem Doctor -- dependency check")
        print("  " + "\\u2500" * 50)
        ok = True

        mac_ver_str = platform.mac_ver()[0] or "0.0"
        parts = mac_ver_str.split(".")
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            major, minor = 0, 0
        if major >= 15:
            print(f"  \\033[92m\\u2713\\033[0m  macOS {major}.{minor}")
        else:
            print(f"  \\033[91m\\u2717\\033[0m  macOS {major}.{minor} -- macOS 15+ required")
            ok = False

        for cmd in ["brew", "ffmpeg", "sox", "ollama"]:
            if shutil.which(cmd):
                print(f"  \\033[92m\\u2713\\033[0m  {cmd}")
            else:
                print(f"  \\033[91m\\u2717\\033[0m  {cmd} -- NOT FOUND")
                ok = False

        native_bin = MUNEEM_HOME / "native" / "muneem-audio"
        if native_bin.exists():
            print(f"  \\033[92m\\u2713\\033[0m  muneem-audio (Core Audio Tap helper)")
        else:
            print(f"  \\033[93m!\\033[0m  muneem-audio -- not compiled (BlackHole fallback will be used)")

        bh_ok = _has_blackhole()
        if bh_ok:
            print(f"  \\033[92m\\u2713\\033[0m  BlackHole (fallback audio)")
        else:
            if not native_bin.exists():
                print(f"  \\033[91m\\u2717\\033[0m  BlackHole -- NOT FOUND and no native helper either")
                ok = False
                print("")
                print("  \\033[1mFix system audio (pick one):\\033[0m")
                print("  \\033[1mBlackHole:\\033[0m  brew install --cask blackhole-2ch")
                print("    Then: Audio MIDI Setup → + → Create Multi-Output Device")
                print("    → Check BlackHole 2ch + your speakers → Use For Sound Output.")
                print("  \\033[1mCore Audio Tap (macOS 15+):\\033[0m")
                setup_py = MUNEEM_HOME / "muneem-setup.py"
                if setup_py.exists():
                    print(f"    python3 {setup_py}")
                else:
                    print("    Re-run the muneem-setup.py installer you originally used.")
                print("")
            else:
                print(f"  \\033[93m!\\033[0m  BlackHole -- not found (native helper available, fallback not needed)")

        venv_python = MUNEEM_HOME / "venv" / "bin" / "python"
        if venv_python.exists():
            print(f"  \\033[92m\\u2713\\033[0m  python venv")
        else:
            print(f"  \\033[91m\\u2717\\033[0m  python venv -- NOT FOUND at {venv_python}")
            ok = False

        for pkg in ["whisperx", "sherpa_onnx", "diarize", "numpy", "requests", "PIL", "rich", "torch", "torchaudio", "Quartz"]:
            import_name = pkg.replace(".", "_")
            if pkg == "Quartz":
                import_name = "Quartz"
            r = subprocess.run(
                [str(venv_python), "-c", f"import {import_name}"],
                capture_output=True,
            )
            if r.returncode == 0:
                print(f"  \\033[92m\\u2713\\033[0m  pip: {pkg}")
            else:
                # sherpa_onnx is primary diarizer but we have a fallback - warn, don't fail
                if pkg == "sherpa_onnx":
                    print(f"  \\033[93m!\\033[0m  pip: {pkg} -- NOT INSTALLED "
                          "(diarization will use `diarize` fallback; re-run muneem-setup.py)")
                else:
                    print(f"  \\033[91m\\u2717\\033[0m  pip: {pkg} -- NOT INSTALLED")
                    ok = False

        # Sherpa-onnx ONNX model files (for primary diarization pipeline)
        _seg_onnx = MUNEEM_HOME / "models" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
        _emb_onnx = MUNEEM_HOME / "models" / "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
        if _seg_onnx.exists():
            print(f"  \\033[92m\\u2713\\033[0m  onnx: pyannote-segmentation-3.0")
        else:
            print(f"  \\033[93m!\\033[0m  onnx: pyannote-segmentation-3.0 -- NOT FOUND "
                  "(diarize fallback will be used; re-run muneem-setup.py)")
        if _emb_onnx.exists():
            print(f"  \\033[92m\\u2713\\033[0m  onnx: 3D-Speaker CAM++ (speaker embeddings)")
        else:
            print(f"  \\033[93m!\\033[0m  onnx: 3D-Speaker CAM++ -- NOT FOUND "
                  "(diarize fallback will be used; re-run muneem-setup.py)")

        required_models = [ENHANCE_MODEL, VISION_MODEL]
        try:
            available = _get_available_models()
            for m in required_models:
                found = _has_model(m, available)
                if found:
                    print(f"  \\033[92m\\u2713\\033[0m  model: {m}")
                else:
                    print(f"  \\033[91m\\u2717\\033[0m  model: {m} -- NOT PULLED (ollama pull {m})")
                    ok = False
        except Exception:
            print(f"  \\033[91m\\u2717\\033[0m  Ollama not reachable -- cannot check models")
            ok = False

        print()
        if ok:
            print("  \\033[92mAll checks passed.\\033[0m Muneem is ready.")
        else:
            setup_py = MUNEEM_HOME / "muneem-setup.py"
            if setup_py.exists():
                print(f"  \\033[93mSome checks failed.\\033[0m Fix: python3 {setup_py}")
            else:
                print("  \\033[93mSome checks failed.\\033[0m Re-run the muneem-setup.py installer to fix.")
        print()


    # ── Subcommand: config ────────────────────────────────────────────────────

    def _ram_gb_sysctl() -> int:
        try:
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            return int(r.stdout.strip()) // (1024 ** 3)
        except Exception:
            return 0


    def _load_config() -> dict:
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return {
                "llm_model": "qwen3:14b",
                "vision_model": "qwen3-vl:8b",
                "whisper_model": "large-v3",
            }


    def _save_config(cfg: dict):
        MUNEEM_HOME.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\\n")


    def cmd_config(args):
        """muneem config show | muneem config llm <qwen3:14b|qwen3:32b>"""
        sub = args[0] if args else "show"

        if sub == "show":
            cfg = _load_config()
            print()
            print(f"  LLM model:      {cfg.get('llm_model')}")
            print(f"  Vision model:   {cfg.get('vision_model')} (fixed)")
            print(f"  Whisper model:  {cfg.get('whisper_model')} (fixed)")
            if cfg.get("ram_gb"):
                print(f"  Detected RAM:   {cfg.get('ram_gb')} GB")
            if cfg.get("chip"):
                print(f"  Detected chip:  {cfg.get('chip')}")
            print(f"  Config file:    {CONFIG_PATH}")
            print()
            return

        if sub == "llm" and len(args) >= 2:
            chosen = args[1]
            if chosen not in ("qwen3:14b", "qwen3:32b"):
                print(f"  \\033[91m\\u2717\\033[0m  Unsupported LLM: {chosen}")
                print("      Valid options: qwen3:14b, qwen3:32b")
                return
            ram = _ram_gb_sysctl()
            if chosen == "qwen3:32b" and ram and ram < 48:
                print(f"  \\033[91m\\u2717\\033[0m  {chosen} requires 48+ GB RAM (detected {ram} GB).")
                print(f"      Aborting - sticking with current selection.")
                return
            # Pull the model if not already local.
            try:
                r = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
                already_local = bool(r.stdout) and any(
                    line.startswith(chosen) or line.startswith(chosen + "-")
                    for line in (r.stdout or "").splitlines()
                )
            except Exception:
                already_local = False
            if not already_local:
                print(f"  Pulling {chosen} (first time only)...")
                pr = subprocess.run(["ollama", "pull", chosen])
                if pr.returncode != 0:
                    print(f"  \\033[91m\\u2717\\033[0m  Pull failed. Config not changed.")
                    return
            cfg = _load_config()
            cfg["llm_model"] = chosen
            _save_config(cfg)
            print(f"  \\033[92m\\u2713\\033[0m  LLM set to {chosen}. Next `muneem start` will use it.")
            return

        print("  Usage:")
        print("    muneem config show")
        print("    muneem config llm qwen3:14b | qwen3:32b")


    # ── Subcommand: help ──────────────────────────────────────────────────────

    def cmd_help(args=None):
        print("""
      Muneem -- Offline AI Notepad (accuracy-first)

      Usage:
        muneem start                     Start a meeting (auto-detects call apps for screen)
        muneem start --no-screen         Audio only, no screen capture
        muneem start --screen            Force screen capture even without detected call app
        muneem start --template standup  Use a specific template (standup|one_on_one|discovery)
        muneem start --display 2         Follow screen 2
        muneem start --window Zoom       Follow Zoom window
        muneem start --window-id 12345   Follow by window ID
        muneem start --record-share 1    Also record display 1 as your screen share
        muneem start --mic "Galaxy Buds2"  Pick specific mic (name/UID substring)
        muneem start --output "MacBook Pro Speakers"  Pick specific output (name/UID substring)
        muneem start --no-audio-prompt   Skip mic/output prompts (use current defaults)
        muneem start --post-process      Record raw audio only; transcribe after stop (low CPU)
        muneem notes                     List all saved meeting notes
        muneem notes last                Open the most recent summary
        muneem notes transcript          Open the most recent full transcript
        muneem ask "your question"       Ask about the last meeting transcript
        muneem status                    Check Ollama, models, audio, and diarization
        muneem doctor                    Verify all dependencies are installed
        muneem config show               Show currently configured LLM / vision / whisper models
        muneem config llm <model>        Switch notes LLM (qwen3:14b | qwen3:32b)
        muneem uninstall                 Remove muneem (keeps Ollama + models + notes)
        muneem uninstall --all           Remove muneem including saved notes
        muneem help                      Show this message

      Output (saved to ~/.muneem/notes/):
        - Summary:    Structured notes (TL;DR, decisions, action items)
        - Transcript: Full speaker-attributed transcript with timestamps
        - Raw:        Unprocessed transcript + screen context

      Screen: ON by default when a call app (Zoom, Teams, etc.) is detected.
             Vision model reads participant names and maps them to diarized audio.
             Use --no-screen to disable.
      Audio: Core Audio Tap (default) with BlackHole fallback.
      Transcription: WhisperX large-v3 with forced alignment (~30s pipelined segments, auto-detect language).
      Diarization: sherpa-onnx (pyannote-seg + 3D-Speaker CAM++) with `diarize` fallback.
                   Session-scoped SpeakerRegistry keeps Speaker 0/1/2 labels stable
                   across segments. Fully offline, Apache-2.0, no tokens needed.
      Speaker names: vision model identifies names on screen → maps to diarized Speaker 0/1/2 labels.
      macOS 15+ (Sequoia/Tahoe) and Apple Silicon required.
        """)


    # ── Main dispatcher ───────────────────────────────────────────────────────

    def main():
        if len(sys.argv) < 2 or sys.argv[1] in ("help", "-h", "--help"):
            cmd_help()
            return

        command = sys.argv[1]

        if command == "start":
            parser = argparse.ArgumentParser(prog="muneem start")
            parser.add_argument("--template", choices=list(TEMPLATES.keys()), default="default")
            parser.add_argument("--screen", action="store_true", help="Force screen capture ON even if no call app is detected.")
            parser.add_argument("--no-screen", action="store_true", help="Disable screen capture (audio only, no speaker name identification)")
            parser.add_argument("--screen-interval", type=int, default=1)
            parser.add_argument("--display", type=int, default=None, help="Capture a specific screen (1=first, 2=second, ...). Skips interactive prompt.")
            parser.add_argument("--window", type=str, default=None, help="Capture a specific app window by name (e.g. Zoom, Webex, Teams, Slack). Skips interactive prompt.")
            parser.add_argument("--window-id", type=int, default=None, help="Capture by window ID. Skips interactive prompt.")
            parser.add_argument("--record-share", type=int, default=None, metavar="N", help="ALSO record display N as the screen you're sharing in the call (parallel capture, analysed separately). Example: --record-share 1")
            parser.add_argument("--mic", type=str, default=None, help="Microphone source by name/UID (e.g. 'MacBook Pro Microphone', 'Galaxy Buds2'). Skips interactive prompt.")
            parser.add_argument("--output", type=str, default=None, help="Output/speaker source by name/UID (e.g. 'Galaxy Buds2', 'MacBook Pro Speakers'). Skips interactive prompt.")
            parser.add_argument("--no-audio-prompt", action="store_true", help="Skip mic/output selection prompts entirely; use current system defaults.")
            parser.add_argument("--post-process", action="store_true",
                                help="Post-process mode: record raw audio only during the call, transcribe + "
                                     "diarize after you stop. Minimises CPU/GPU load during demanding video "
                                     "calls. Transcription runs on Ctrl+C. Default: off (real-time transcription).")
            cmd_start(parser.parse_args(sys.argv[2:]))

        elif command == "notes":
            class NS: sub = sys.argv[2] if len(sys.argv) > 2 else None
            cmd_notes(NS())

        elif command == "ask":
            class NS: question = sys.argv[2:]
            cmd_ask(NS())

        elif command == "status":
            cmd_status(None)

        elif command == "doctor":
            cmd_doctor(None)

        elif command == "config":
            cmd_config(sys.argv[2:])

        else:
            print(f"  Unknown command: {command}")
            cmd_help()


    if __name__ == "__main__":
        main()
''')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print()
    print(f"{BOLD}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║            MUNEEM - Setup & Installation                ║{RESET}")
    print(f"{BOLD}║    Accuracy-First Offline AI Notepad (Apple Silicon)    ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════════╝{RESET}")

    # Step sequence (10 steps total):
    #   1. Prereqs (macOS / arm64 / Python 3.10-3.13 discovery / brew / Xcode CLT)
    #   2. Brew packages
    #   3. Project directories
    #   4. Core Audio Tap (Swift) native helper
    #   5. System resource check (RAM, chip)
    #   6. Notes LLM selection (14B default; 32B if 48+ GB RAM)
    #   7. Python venv + pip packages + WhisperX model pre-cache
    #   8. Write embedded application modules
    #   9. Ollama: verify daemon + pull required models
    #  10. CLI wrapper + symlink
    check_prerequisites()
    install_brew_packages()
    create_directories()
    core_audio_tap_ok = compile_native_helper()
    resources = check_system_resources()
    llm_model = select_llm(resources)
    setup_python_env()
    download_sherpa_diar_models()
    write_app_modules()
    pull_ollama_models(llm_model)
    create_cli_wrapper()

    header("Setup complete!")
    audio_default_note = (
        f"  {BOLD}System audio:{RESET} Core Audio Tap is configured as the default backend.\n"
        if core_audio_tap_ok
        else f"  {BOLD}System audio:{RESET} Core Audio Tap could not be built; BlackHole is the fallback.\n"
    )
    print(f"""  Muneem has been installed to {MUNEEM_HOME}/
  CLI available at /usr/local/bin/muneem (ready immediately).

{audio_default_note}
  {BOLD}Get started:{RESET}

    muneem start                     # Start a meeting session
    muneem start --template standup  # Standup template
    muneem start --no-screen         # Audio only
    muneem start --post-process      # Record raw only; transcribe post-meeting (low CPU)
    muneem notes                     # List saved notes
    muneem notes last                # Open the latest note
    muneem ask "what was decided?"   # Ask about last meeting
    muneem status                    # Check everything
    muneem doctor                    # Verify all dependencies
    muneem uninstall                 # Remove muneem (keeps notes)
    muneem uninstall --all           # Remove muneem including saved notes

  {BOLD}Alternate (without the muneem CLI):{RESET}
    python3 {MUNEEM_HOME}/muneem-setup.py uninstall
    python3 {MUNEEM_HOME}/muneem-setup.py uninstall --all

  {BOLD}Speaker diarization:{RESET}
    Already included and fully offline. No accounts, tokens, or
    internet needed. Primary: sherpa-onnx (pyannote-segmentation + 3D-Speaker
    CAM++ embeddings) with a session-scoped SpeakerRegistry that keeps
    Speaker 0/1/2 labels stable across segments. Falls back to the `diarize`
    package if the ONNX models aren't present.

  {BOLD}BlackHole (fallback if Core Audio Tap unavailable):{RESET}
    1. Open Audio MIDI Setup (Spotlight > "Audio MIDI Setup")
    2. Click "+", select "Create Multi-Output Device"
    3. Check both "BlackHole 2ch" and your speakers/headphones
    4. Right-click the Multi-Output > "Use This Device For Sound Output"

  {GREEN}Muneem is ready.{RESET}
""")


def do_uninstall(delete_all=False):
    """Uninstall muneem - mirrors the shell wrapper's uninstall logic."""
    print()
    print(f"  {BOLD}Muneem Uninstall{RESET}")
    print("  " + "═" * 40)
    print()

    # Safety check: verify this looks like a valid muneem directory
    if not MUNEEM_HOME.exists():
        print(f"  ⚠  {MUNEEM_HOME} does not exist. Nothing to uninstall.")
        return

    # Check for marker files to ensure we're not deleting wrong directory
    has_muneem_structure = any([
        (MUNEEM_HOME / "venv").exists(),
        (MUNEEM_HOME / "app.py").exists(),
        (MUNEEM_HOME / "muneem-setup.py").exists(),
    ])
    if not has_muneem_structure:
        print(f"  ✗ Safety check failed: {MUNEEM_HOME} doesn't look like a Muneem installation.")
        print(f"     Expected to find: venv/, app.py, or muneem-setup.py")
        print(f"     Aborting uninstall to prevent data loss.")
        return

    print(f"  This will remove:")
    print(f"    - {MUNEEM_HOME}  (venv, modules, native helper, tmp)")
    print(f"    - /usr/local/bin/muneem  (CLI symlink)")
    print()
    print(f"  This will NOT remove:")
    print(f"    - Ollama (brew service)")
    print(f"    - Downloaded Ollama models (~/.ollama/models/)")
    print(f"    - Homebrew packages (ffmpeg, sox, portaudio, blackhole-2ch)")
    print()
    if delete_all:
        print(f"  {YELLOW}⚠{RESET}  --all flag: notes will ALSO be deleted.")
    else:
        print(f"  Your notes will be preserved at {MUNEEM_HOME}/notes/")
        print(f"  Use --all to also delete notes.")
    print()
    confirm = input("  Proceed? [y/N] ").strip()
    if confirm.lower() != "y":
        print("  Aborted.")
        return
    print()

    backup = None
    if not delete_all and (MUNEEM_HOME / "notes").is_dir():
        backup = Path.home() / "muneem-notes-backup"
        print(f"  → Backing up notes to {backup}/ ...")
        try:
            if backup.exists():
                shutil.rmtree(backup)
            shutil.copytree(MUNEEM_HOME / "notes", backup)
        except Exception as e:
            fail(f"Failed to backup notes: {e}. Aborting uninstall to prevent data loss.")

    print(f"  → Removing {MUNEEM_HOME} ...")
    try:
        shutil.rmtree(MUNEEM_HOME, ignore_errors=False)
    except Exception as e:
        fail(f"Failed to remove {MUNEEM_HOME}: {e}")

    if backup and backup.exists():
        try:
            MUNEEM_HOME.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup), str(MUNEEM_HOME / "notes"))
            print(f"  → Notes restored to {MUNEEM_HOME}/notes/")
        except Exception as e:
            warn(f"Failed to restore notes: {e}")
            print(f"  Your notes are backed up at {backup}/")

    symlink = Path("/usr/local/bin/muneem")
    if symlink.is_symlink() or symlink.exists():
        print(f"  → Removing {symlink} ...")
        try:
            symlink.unlink()
        except PermissionError:
            try:
                subprocess.run(["sudo", "rm", "-f", str(symlink)], check=True)
            except Exception as e:
                warn(f"Could not remove symlink (may require manual 'sudo rm {symlink}'): {e}")
        except Exception as e:
            warn(f"Could not remove symlink: {e}")

    print()
    print(f"  {GREEN}✓{RESET}  Muneem has been uninstalled.")
    print(f"     Ollama and all downloaded models are untouched.")
    if not delete_all:
        print(f"     Your notes are still at {MUNEEM_HOME}/notes/")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        do_uninstall(delete_all="--all" in sys.argv)
    else:
        main()
