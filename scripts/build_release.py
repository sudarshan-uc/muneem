#!/usr/bin/env python3
"""
scripts/build_release.py - Produce a closed-source single-file installer.

Everything release-related lives in this file. ``muneem-setup.py`` is a clean
user-facing installer with no release hooks; this script does all the
heavy lifting externally:

  1. Extract the four embedded modules (_MODULE_TRANSCRIBER/SCREEN_READER/
     ENHANCER/APP) out of muneem-setup.py into build/src/modules/*.py.
  2. Nuitka-compile each module into a .so extension (obfuscated; no
     plaintext source).
  3. Compile + ad-hoc-codesign the Swift Core Audio Tap .app bundle.
  4. Generate build/src/muneem-installer-release.py - a standalone COPY of
     muneem-setup.py rewritten so write_app_modules() copies the compiled
     .so files, compile_native_helper() copies the pre-built .app, and the
     four _MODULE_* string literals are stripped. The committed
     muneem-setup.py is never modified.
  5. Nuitka --standalone --onefile the rewritten copy, bundling the .so
     files and the .app as data files.
  6. Ad-hoc-codesign the final binary.

Usage (from a macos-14 runner or local Apple Silicon Mac with Nuitka):

    python scripts/build_release.py --version 0.1.0

Produces: dist/muneem-installer (single-file binary, arm64, ad-hoc signed).
"""
from __future__ import annotations

import argparse
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import textwrap

REPO = pathlib.Path(__file__).resolve().parent.parent
SETUP = REPO / "muneem-setup.py"
BUILD = REPO / "build"
DIST = REPO / "dist"

# Modules extracted out of muneem-setup.py. Order is informational only -
# Nuitka compiles each .py to .so independently.
MODULES = ["transcriber", "screen_reader", "enhancer", "app"]

# Imports Nuitka should NOT try to bundle into the compiled per-module .so.
# They live in the user's Python venv at install time; the compiled module
# resolves them at runtime via normal Python import, so bundling them here
# would just bloat the artefact (or fail outright for compiled extensions).
NOFOLLOW_MODULE = [
    "whisperx",
    "torch",
    "torchaudio",
    "numpy",
    "PIL",
    "requests",
    "rich",
    "sherpa_onnx",
    "diarize",
    "pyannote",
    "silero_vad",
    "certifi",
    "ollama",
    "Quartz",
    "objc",
    "AppKit",
]

# Mirror of _MUNEEM_AUDIO_INFO_PLIST in muneem-setup.py. Kept here so the
# release build can assemble the .app without importing the installer.
INFO_PLIST = """\
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
"""

# ─── Snippets injected into the rewritten installer ──────────────────────────
#
# The generated build/src/muneem-installer-release.py is a drop-in
# replacement for muneem-setup.py that does NOT write plaintext Python
# modules or invoke swiftc. Instead, it copies pre-built artefacts out of
# the Nuitka onefile extraction directory (exposed via ``__compiled__`` on
# the running binary, with a fallback for plain-Python test runs).

INJECTED_RELEASE_HEADER = r'''
# ─── Release-mode runtime helpers (injected by scripts/build_release.py) ────
# These helpers exist only in the compiled single-file installer. They
# replace the plaintext-module writer and the swiftc invocation of the
# source installer.

def _release_bundle_dir():
    """Return the dir that holds the Nuitka onefile's extracted data."""
    try:
        compiled = globals().get("__compiled__")
        if compiled is not None and hasattr(compiled, "containing_dir"):
            return Path(compiled.containing_dir)
    except Exception:
        pass
    return Path(sys.executable).resolve().parent


_RELEASE_MODULE_FILES = [
    ("modules/transcriber.so",   "transcriber.so"),
    ("modules/screen_reader.so", "screen_reader.so"),
    ("modules/enhancer.so",      "enhancer.so"),
    ("modules/app.so",           "app.so"),
]
_RELEASE_APP_BUNDLE_REL = "native/muneem-audio.app"


def _release_copy_modules():
    bundle = _release_bundle_dir()
    for rel, dest_name in _RELEASE_MODULE_FILES:
        src = bundle / rel
        dst = MUNEEM_HOME / dest_name
        if not src.exists():
            fail(f"Release bundle missing compiled module: {rel}")
        shutil.copy2(str(src), str(dst))
        dst.chmod(0o644)
        info(f"Installed compiled module {dst}")


def _release_copy_app_bundle():
    bundle = _release_bundle_dir()
    src = bundle / _RELEASE_APP_BUNDLE_REL
    dst = NATIVE_DIR / "muneem-audio.app"
    if not src.exists():
        warn(f"Release bundle missing {_RELEASE_APP_BUNDLE_REL}. BlackHole fallback will be used.")
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(str(src), str(dst))
    macos_bin = dst / "Contents" / "MacOS" / "muneem-audio"
    if macos_bin.exists():
        macos_bin.chmod(0o755)
        try:
            flat = NATIVE_DIR / "muneem-audio"
            if flat.exists() or flat.is_symlink():
                flat.unlink()
            shutil.copy2(str(macos_bin), str(flat))
            flat.chmod(0o755)
        except Exception as e:
            warn(f"Could not stage flat muneem-audio binary: {e}")
    info(f"Installed pre-built {dst}")
    return True

'''

# Replacement body for write_app_modules() in the release variant. The
# compiled binary cannot re-emit plaintext Python source, so it copies
# the prebuilt .so files out of the onefile bundle instead.
RELEASE_WRITE_APP_MODULES = '''\
def write_app_modules():
    header("Step 8/10 - Installing application modules (pre-compiled)")
    _release_copy_modules()
    # Stage the installer binary inside ~/.muneem/ so `muneem uninstall` and
    # re-runs work without the original download.
    try:
        src = Path(sys.executable).resolve()
        dst = MUNEEM_HOME / "muneem-installer"
        shutil.copy2(str(src), str(dst))
        dst.chmod(0o755)
        info(f"Copied installer binary to {dst}")
    except Exception as e:
        warn(f"Could not copy installer binary: {e}")
'''

# Replacement body for compile_native_helper() - copies a pre-built,
# pre-signed .app out of the onefile bundle instead of invoking swiftc.
RELEASE_COMPILE_NATIVE_HELPER = '''\
def compile_native_helper():
    """Release build: install pre-built, pre-signed muneem-audio.app."""
    header("Step 4/10 - Installing Core Audio Tap helper (pre-built .app)")
    ok = _release_copy_app_bundle()
    if ok:
        info("Core Audio Tap configured as default system audio backend.")
    return ok
'''


def sh(cmd: list[str], *, cwd: pathlib.Path | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


# ─── Step 1: extract embedded modules ────────────────────────────────────────

def extract_modules(source: str, out_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pathlib.Path] = {}
    for name in MODULES:
        var = f"_MODULE_{name.upper()}"
        m = re.search(
            rf"{var}\s*=\s*textwrap\.dedent\((r?'''.*?''')\)",
            source,
            re.DOTALL,
        )
        if not m:
            die(f"could not find {var} in muneem-setup.py")
        code = textwrap.dedent(eval(m.group(1)))
        path = out_dir / f"{name}.py"
        path.write_text(code, encoding="utf-8")
        print(f"  extracted {var} -> {path} ({len(code)} chars)")
        out[name] = path
    return out


def extract_swift(source: str, out_path: pathlib.Path) -> None:
    m = re.search(r"SWIFT_AUDIO_HELPER\s*=\s*(r?'''.*?''')", source, re.DOTALL)
    if not m:
        die("could not find SWIFT_AUDIO_HELPER in muneem-setup.py")
    out_path.write_text(eval(m.group(1)), encoding="utf-8")
    print(f"  extracted SWIFT_AUDIO_HELPER -> {out_path}")


# ─── Step 2: compile .py -> .so via Nuitka --module ──────────────────────────

def compile_modules(srcs: dict[str, pathlib.Path], out_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    compiled: dict[str, pathlib.Path] = {}
    for name, src in srcs.items():
        cmd = [
            sys.executable, "-m", "nuitka",
            "--module",
            "--remove-output",
            f"--output-dir={out_dir}",
            "--assume-yes-for-downloads",
        ]
        for pkg in NOFOLLOW_MODULE:
            cmd.append(f"--nofollow-import-to={pkg}")
        cmd.append(str(src))
        sh(cmd)
        matches = sorted(out_dir.glob(f"{name}.*.so"))
        if not matches:
            die(f"nuitka produced no .so for {name} in {out_dir}")
        compiled[name] = matches[-1]
        print(f"  compiled {name} -> {compiled[name].name}")
    return compiled


# ─── Step 3: build the Swift .app bundle ─────────────────────────────────────

def build_swift_app(src: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "muneem-audio"
    sh([
        "swiftc", "-O", "-o", str(bin_path), str(src),
        "-framework", "Foundation",
        "-framework", "CoreAudio",
        "-framework", "AVFoundation",
    ])
    bin_path.chmod(0o755)

    app = out_dir / "muneem-audio.app"
    if app.exists():
        shutil.rmtree(app)
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Resources").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_text(INFO_PLIST)
    shutil.copy2(str(bin_path), str(app / "Contents" / "MacOS" / "muneem-audio"))
    (app / "Contents" / "MacOS" / "muneem-audio").chmod(0o755)
    sh(["codesign", "--force", "--deep", "--sign", "-", str(app)])
    print(f"  built and signed {app}")
    return app


# ─── Step 4: generate the release variant of muneem-setup.py ─────────────────

_WRITE_FN_RX = re.compile(
    r"^def write_app_modules\(\):.*?(?=^\n+def |^# )",
    re.DOTALL | re.MULTILINE,
)
_COMPILE_FN_RX = re.compile(
    r"^def compile_native_helper\(\):.*?(?=^\n+def |^# )",
    re.DOTALL | re.MULTILINE,
)


def patch_setup_for_release(source: str, out_path: pathlib.Path) -> None:
    """Generate the release variant of muneem-setup.py.

    This produces a STANDALONE copy - the committed muneem-setup.py is
    not touched. The copy:
      * Has the release helper block injected after ``__version__``.
      * Has ``write_app_modules()`` replaced with a version that copies
        pre-compiled .so files.
      * Has ``compile_native_helper()`` replaced with a version that
        copies a pre-built .app bundle.
      * Has the four ``_MODULE_*`` embedded strings blanked so the
        compiled binary never carries plaintext source.
    """
    patched = source

    # 1. Inject the release-mode runtime helpers after `__version__ = ...`.
    patched, n = re.subn(
        r'(^__version__\s*=\s*"[^"]+"\s*\n)',
        r"\1" + INJECTED_RELEASE_HEADER + "\n",
        patched,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        die("could not inject release helpers after __version__")

    # 2. Replace write_app_modules() with the release variant.
    if not _WRITE_FN_RX.search(patched):
        die("could not locate write_app_modules() to replace")
    patched = _WRITE_FN_RX.sub(RELEASE_WRITE_APP_MODULES + "\n\n", patched, count=1)

    # 3. Replace compile_native_helper() with the release variant.
    if not _COMPILE_FN_RX.search(patched):
        die("could not locate compile_native_helper() to replace")
    patched = _COMPILE_FN_RX.sub(RELEASE_COMPILE_NATIVE_HELPER + "\n\n", patched, count=1)

    # 4. Strip the four embedded _MODULE_* string literals so the compiled
    #    onefile never carries plaintext source for them. Replace with an
    #    empty string so the (now unused) references in the old
    #    write_app_modules() body couldn't leak source even if someone
    #    re-introduced the old call path.
    for name in MODULES:
        var = f"_MODULE_{name.upper()}"
        patched, m = re.subn(
            rf"{var}\s*=\s*textwrap\.dedent\(r?'''.*?'''\)",
            f"{var} = ''  # stripped by scripts/build_release.py",
            patched,
            count=1,
            flags=re.DOTALL,
        )
        if m != 1:
            die(f"could not strip {var} during release patch")

    # 5. The release build does not call swiftc; drop the Swift source and
    #    Info.plist string constants so the compiled binary doesn't carry
    #    them around unnecessarily.
    for const in ("SWIFT_AUDIO_HELPER", "_MUNEEM_AUDIO_INFO_PLIST"):
        patched = re.sub(
            rf"^{const}\s*=\s*(textwrap\.dedent\()?(r?'''.*?'''|r?\"\"\".*?\"\"\")\)?\s*\n",
            f"{const} = ''  # stripped by scripts/build_release.py\n",
            patched,
            count=1,
            flags=re.DOTALL | re.MULTILINE,
        )

    out_path.write_text(patched, encoding="utf-8")
    print(f"  wrote release-patched installer to {out_path}")


# ─── Step 5: final onefile compile ───────────────────────────────────────────

def build_onefile(
    release_src: pathlib.Path,
    compiled: dict[str, pathlib.Path],
    app_bundle: pathlib.Path,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--onefile",
        "--assume-yes-for-downloads",
        "--macos-create-app-bundle=no",
        f"--output-dir={out_dir}",
        "--output-filename=muneem-installer",
    ]
    for name, so in compiled.items():
        cmd.append(f"--include-data-files={so}=modules/{name}.so")
    cmd.append(f"--include-data-dir={app_bundle}=native/muneem-audio.app")
    cmd.append(str(release_src))
    sh(cmd)
    final = out_dir / "muneem-installer"
    if not final.exists():
        die(f"nuitka onefile output missing: {final}")
    final.chmod(0o755)
    return final


def codesign_final(binary: pathlib.Path) -> None:
    sh(["codesign", "--force", "--deep", "--sign", "-", str(binary)])
    print(f"  ad-hoc signed {binary}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Release version, must match __version__")
    parser.add_argument("--skip-swift", action="store_true",
                        help="Dev-only: skip the Swift .app build (e.g. on non-macOS)")
    parser.add_argument("--skip-nuitka", action="store_true",
                        help="Dev-only: stop after generating build/src/*.py (no compilation)")
    args = parser.parse_args()

    if platform.system() != "Darwin":
        print("WARNING: building on non-Darwin host; output will not be usable on macOS.")

    source = SETUP.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
    if not m or m.group(1) != args.version:
        die(f"__version__ ({m.group(1) if m else 'missing'}) != --version {args.version}")

    if BUILD.exists():
        shutil.rmtree(BUILD)
    if DIST.exists():
        shutil.rmtree(DIST)
    src_dir = BUILD / "src"
    out_dir = BUILD / "out"
    src_dir.mkdir(parents=True)

    print("\n── 1/6 Extract embedded modules ──────────────────────────")
    srcs = extract_modules(source, src_dir / "modules")

    print("\n── 2/6 Generate release installer copy ───────────────────")
    release_src = src_dir / "muneem-installer-release.py"
    patch_setup_for_release(source, release_src)
    # Sanity-check: the generated copy must still parse as valid Python.
    import ast
    ast.parse(release_src.read_text(encoding="utf-8"))
    print("  release copy parses as valid Python")

    if args.skip_nuitka:
        print("\nStopped after step 2 (--skip-nuitka).")
        print(f"Inspect build artefacts under {BUILD}/")
        return 0

    print("\n── 3/6 Compile modules with Nuitka ───────────────────────")
    compiled = compile_modules(srcs, out_dir)

    if args.skip_swift:
        print("\n── 4/6 Skipping Swift build (--skip-swift) ──")
        app_bundle = out_dir / "muneem-audio.app"
        (app_bundle / "Contents" / "MacOS").mkdir(parents=True, exist_ok=True)
    else:
        print("\n── 4/6 Build Swift .app bundle ───────────────────────")
        swift_src = src_dir / "muneem_audio.swift"
        extract_swift(source, swift_src)
        app_bundle = build_swift_app(swift_src, out_dir)

    print("\n── 5/6 Build onefile binary with Nuitka ──────────────────")
    final = build_onefile(release_src, compiled, app_bundle, DIST)

    print("\n── 6/6 Codesign (ad-hoc) ─────────────────────────────────")
    codesign_final(final)

    print(f"\n✓ Release binary ready: {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
