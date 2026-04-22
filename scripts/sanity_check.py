#!/usr/bin/env python3
"""
Repo sanity check.

Validates:
  1. muneem-setup.py declares __version__ == EXPECTED_VERSION
  2. muneem-setup.py parses as valid Python (syntax check)
  3. All four embedded module strings inside muneem-setup.py parse as valid Python
  4. Required top-level files exist (INSTALL.md, USAGE.md, LICENSE)
  5. PIP_PACKAGES contains the minimum expected dependency set

Exits 0 on success, non-zero on any failure.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys
import textwrap

EXPECTED_VERSION = "0.1.0"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SETUP = REPO_ROOT / "muneem-setup.py"

REQUIRED_FILES = ["INSTALL.md", "USAGE.md", "LICENSE", "muneem-setup.py"]
REQUIRED_PIP = ["whisperx", "sherpa-onnx", "torch", "torchaudio", "pyobjc-framework-Quartz"]
EMBEDDED_MODULES = [
    "_MODULE_TRANSCRIBER",
    "_MODULE_SCREEN_READER",
    "_MODULE_ENHANCER",
    "_MODULE_APP",
]

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"ok:   {msg}")


def check_required_files() -> None:
    for name in REQUIRED_FILES:
        p = REPO_ROOT / name
        if not p.exists():
            fail(f"missing required file: {name}")
        else:
            ok(f"required file present: {name}")


def check_main_syntax(source: str) -> None:
    try:
        ast.parse(source)
        ok("muneem-setup.py parses as valid Python")
    except SyntaxError as e:
        fail(f"muneem-setup.py syntax error: {e}")


def check_version(source: str) -> None:
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
    if not m:
        fail("__version__ not found in muneem-setup.py")
        return
    found = m.group(1)
    if found != EXPECTED_VERSION:
        fail(f"__version__ is {found!r}, expected {EXPECTED_VERSION!r}")
    else:
        ok(f"__version__ == {EXPECTED_VERSION}")


def check_embedded_modules(source: str) -> None:
    for name in EMBEDDED_MODULES:
        pat = rf"{name}\s*=\s*textwrap\.dedent\((r?'''.*?''')\)"
        m = re.search(pat, source, re.DOTALL)
        if not m:
            fail(f"embedded module not found: {name}")
            continue
        try:
            code = textwrap.dedent(eval(m.group(1)))
        except Exception as e:
            fail(f"could not decode embedded module {name}: {e}")
            continue
        try:
            ast.parse(code)
            ok(f"embedded module parses: {name} ({len(code)} chars)")
        except SyntaxError as e:
            fail(f"embedded module syntax error in {name}: {e}")


def check_pip_packages(source: str) -> None:
    m = re.search(r"PIP_PACKAGES\s*=\s*\[(.*?)\]", source, re.DOTALL)
    if not m:
        fail("PIP_PACKAGES list not found")
        return
    block = m.group(1)
    for pkg in REQUIRED_PIP:
        if pkg not in block:
            fail(f"PIP_PACKAGES missing required dependency: {pkg}")
        else:
            ok(f"PIP_PACKAGES contains: {pkg}")


def main() -> int:
    if not SETUP.exists():
        fail(f"muneem-setup.py not found at {SETUP}")
        return 1
    source = SETUP.read_text(encoding="utf-8")

    check_required_files()
    check_main_syntax(source)
    check_version(source)
    check_embedded_modules(source)
    check_pip_packages(source)

    print()
    if failures:
        print(f"sanity check FAILED ({len(failures)} issue(s))")
        return 1
    print("sanity check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
