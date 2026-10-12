"""Locate bundled ffmpeg/ffprobe (inside the exe) or fall back to PATH."""
import os
import shutil
import subprocess
import sys


def _candidates(name):
    exe = name + (".exe" if os.name == "nt" else "")
    base = getattr(sys, "_MEIPASS", None)
    if base:
        yield os.path.join(base, "bin", exe)
    here = os.path.dirname(os.path.abspath(sys.argv[0]))
    yield os.path.join(here, "bin", exe)
    yield os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", exe)


def find_tool(name):
    for c in _candidates(name):
        if os.path.isfile(c):
            return c
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} not found. Put it in the 'bin' folder or on PATH.")


def no_window_flags():
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def asset_path(name):
    """Path to a bundled asset (works from source and inside the PyInstaller exe)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, "assets", name)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", name)


def model_path(name="rvm_mobilenetv3_fp32.onnx"):
    """Path to a bundled AI model (works from source and inside the PyInstaller exe)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, "models", name)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", name)
