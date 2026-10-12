"""Frame and audio sources, all through ffmpeg (so camera names work exactly like in the Lecture recorder).

* camera:  dshow (Windows) -> raw RGB frames of a fixed size/fps
* file / test pattern: for trying the app without a camera
The reader keeps only the NEWEST frame, so the pipeline never builds up delay.
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from typing import List, Optional

import numpy as np

from .paths import find_tool, no_window_flags

IS_WIN = sys.platform == "win32"


# --------------------------------------------------------------------------- device discovery (Windows dshow)
def parse_dshow_list(text: str) -> dict:
    out = {"video": [], "audio": []}
    section = None
    for line in text.splitlines():
        low = line.lower()
        if "alternative name" in low:
            continue
        if "directshow video devices" in low:
            section = "video"; continue
        if "directshow audio devices" in low:
            section = "audio"; continue
        m = re.search(r'"([^"]+)"\s*\((video|audio|none)\)', line)
        if m and m.group(2) != "none":
            kind, name = m.group(2), m.group(1)
        else:
            m = re.search(r'\]\s+"([^"]+)"\s*$', line)
            if not m or not section:
                continue
            kind, name = section, m.group(1)
        if name not in out[kind]:
            out[kind].append(name)
    return out


def list_devices() -> dict:
    if not IS_WIN:
        return {"video": [], "audio": []}
    try:
        r = subprocess.run([find_tool("ffmpeg"), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                           capture_output=True, text=True, errors="replace", timeout=20, creationflags=no_window_flags())
        return parse_dshow_list(r.stderr + r.stdout)
    except Exception:  # noqa: BLE001
        return {"video": [], "audio": []}


# --------------------------------------------------------------------------- video
class FrameSource:
    """ffmpeg -> raw rgb24 frames of exactly w x h at ~fps. `latest()` returns the newest frame (or None)."""

    def __init__(self, input_args: List[str], w: int, h: int, fps: int):
        self.input_args, self.w, self.h, self.fps = input_args, w, h, fps
        self.proc: Optional[subprocess.Popen] = None
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self.error = ""
        self._err_tail: List[str] = []

    def start(self):
        w, h = self.w, self.h
        vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={self.fps}"
        cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", *self.input_args,
               "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                     creationflags=no_window_flags())
        threading.Thread(target=self._drain_err, daemon=True).start()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _drain_err(self):
        for line in iter(self.proc.stderr.readline, b""):
            self._err_tail = (self._err_tail + [line.decode("utf-8", "replace").rstrip()])[-6:]

    def _read(self):
        n = self.w * self.h * 3
        buf = bytearray(n)
        mv = memoryview(buf)
        while not self._stop:
            got = 0
            while got < n:
                chunk = self.proc.stdout.read(n - got)
                if not chunk:
                    self.error = "\n".join(self._err_tail) or "The video source stopped."
                    return
                mv[got:got + len(chunk)] = chunk
                got += len(chunk)
            frame = np.frombuffer(bytes(buf), np.uint8).reshape(self.h, self.w, 3)
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1

    def latest(self, last_seq: int = -1, timeout: float = 1.0):
        """-> (frame, seq) once a frame newer than last_seq exists (or (None, last_seq) on timeout / source ended)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._frame is not None and self._seq != last_seq:
                    return self._frame, self._seq
            if self.error:
                break
            time.sleep(0.002)
        return None, last_seq

    def stop(self):
        self._stop = True
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


def camera_args(name: str, fps: int, cam_size: str = "") -> List[str]:
    a = ["-f", "dshow", "-rtbufsize", "64M", "-fflags", "nobuffer", "-flags", "low_delay", "-framerate", str(fps)]
    if cam_size:
        a += ["-video_size", cam_size]
    return a + ["-i", f"video={name}"]


def file_args(path: str, loop: bool = True) -> List[str]:
    return ["-re", *(["-stream_loop", "-1"] if loop else []), "-i", path]


def test_pattern_args(fps: int = 30) -> List[str]:
    return ["-re", "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate={fps}"]


# --------------------------------------------------------------------------- audio
class MicSource:
    """ffmpeg -> raw 48 kHz mono s16le PCM chunks (~20 ms) delivered to `on_pcm(bytes)`."""
    RATE = 48000
    CHUNK = 960 * 2                 # 20 ms of 16-bit mono

    def __init__(self, input_args: List[str], on_pcm):
        self.input_args, self.on_pcm = input_args, on_pcm
        self.proc: Optional[subprocess.Popen] = None
        self.error = ""
        self._stop = False

    def start(self):
        cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", *self.input_args,
               "-ac", "1", "-ar", str(self.RATE), "-f", "s16le", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                     creationflags=no_window_flags())
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while not self._stop:
            data = b""
            while len(data) < self.CHUNK:
                part = self.proc.stdout.read(self.CHUNK - len(data))
                if not part:
                    if not self._stop:
                        self.error = (self.proc.stderr.read() or b"").decode("utf-8", "replace")[-300:] or "The microphone stopped."
                    return
                data += part
            self.on_pcm(data)

    def stop(self):
        self._stop = True
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


def mic_args(name: str) -> List[str]:
    return ["-f", "dshow", "-rtbufsize", "16M", "-i", f"audio={name}"]


def wav_args(path: str, realtime: bool = True) -> List[str]:
    return [*(["-re"] if realtime else []), "-i", path]
