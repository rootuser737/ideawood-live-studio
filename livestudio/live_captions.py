"""Live captions: microphone audio -> speech segments -> whisper (resident server) -> caption text on the stream.

Whisper is not a streaming model, so captions appear a moment AFTER each phrase (typically 1-3 s on a decent PC with the
tiny/base model). A sync monitor watches the delay: if the PC cannot keep up, live captions are switched off with a
notice, the recording continues, and the full captions are produced from the recording after you stop.
"""
from __future__ import annotations

import collections
import io
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.request
import wave
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .paths import no_window_flags

RATE = 16000
FRAME = 320                      # 20 ms at 16 kHz


def to_16k(pcm48: bytes) -> np.ndarray:
    """48 kHz mono s16le bytes -> 16 kHz int16 (average of 3 samples = cheap low-pass + decimation)."""
    x = np.frombuffer(pcm48[: len(pcm48) // 6 * 6], np.int16).astype(np.int32)
    return (x.reshape(-1, 3).sum(axis=1) // 3).astype(np.int16)


@dataclass
class Segment:
    pcm: np.ndarray              # int16, 16 kHz
    t_start: float               # seconds on the audio clock (since the mic started)
    t_end: float


class Segmenter:
    """Energy-based phrase detector: starts on speech, ends after a short pause or at max_len seconds."""

    def __init__(self, max_len: float = 5.0, end_silence: float = 0.45, min_speech: float = 0.30,
                 abs_threshold: float = 300.0):
        self.max_frames = int(max_len / 0.02)
        self.end_frames = int(end_silence / 0.02)
        self.min_frames = int(min_speech / 0.02)
        self.abs = abs_threshold
        self.noise = 100.0
        self.preroll: collections.deque = collections.deque(maxlen=12)
        self.cur: List[np.ndarray] = []
        self.speech_run = 0
        self.silence_run = 0
        self.speech_frames = 0
        self.t = 0.0                       # audio clock at the end of the last frame
        self._rem = np.zeros(0, np.int16)
        self.start_t = 0.0

    def push(self, pcm16: np.ndarray) -> List[Segment]:
        out: List[Segment] = []
        x = np.concatenate([self._rem, pcm16]) if len(self._rem) else pcm16
        n = len(x) // FRAME
        for i in range(n):
            fr = x[i * FRAME:(i + 1) * FRAME]
            self.t += 0.02
            e = float(np.sqrt(np.mean(fr.astype(np.float32) ** 2)))
            thr = max(self.abs, self.noise * 3.0)
            loud = e > thr
            if not self.cur:
                self.noise = min(self.noise * 0.98 + e * 0.02, 1500.0) if not loud else self.noise
                self.preroll.append(fr)
                self.speech_run = self.speech_run + 1 if loud else 0
                if self.speech_run >= 3:
                    self.cur = list(self.preroll)
                    self.start_t = self.t - 0.02 * len(self.cur)
                    self.speech_frames = self.speech_run
                    self.silence_run = 0
            else:
                self.cur.append(fr)
                if loud:
                    self.silence_run = 0
                    self.speech_frames += 1
                else:
                    self.silence_run += 1
                if self.silence_run >= self.end_frames or len(self.cur) >= self.max_frames:
                    seg = self._finish()
                    if seg:
                        out.append(seg)
        self._rem = x[n * FRAME:]
        return out

    def _finish(self) -> Optional[Segment]:
        frames = self.cur
        keep = max(0, len(frames) - max(0, self.silence_run - 5))     # trim trailing silence (keep ~100 ms)
        pcm = np.concatenate(frames[:keep]) if keep else np.zeros(0, np.int16)
        seg = Segment(pcm, self.start_t, self.start_t + len(pcm) / RATE) if self.speech_frames >= self.min_frames else None
        self.cur, self.speech_run, self.silence_run, self.speech_frames = [], 0, 0, 0
        self.preroll.clear()
        return seg

    def flush(self) -> Optional[Segment]:
        return self._finish() if self.cur else None


# --------------------------------------------------------------------------- whisper access
def wav_bytes(pcm16: np.ndarray) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm16.tobytes())
    return b.getvalue()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class WhisperServer:
    """Keeps the speech model loaded (whisper-server) so each phrase is transcribed quickly."""

    def __init__(self, server_exe: str, model: str, language: str = "en", threads: int = 2, prompt: str = ""):
        self.exe, self.model, self.language, self.threads, self.prompt = server_exe, model, language, threads, prompt
        self.port = 0
        self.proc: Optional[subprocess.Popen] = None

    def start(self, timeout: float = 40.0):
        self.port = _free_port()
        cmd = [self.exe, "-m", self.model, "--host", "127.0.0.1", "--port", str(self.port), "-t", str(self.threads),
               "-l", self.language]
        if self.prompt.strip():
            cmd += ["--prompt", self.prompt.strip()[:300]]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     cwd=os.path.dirname(os.path.abspath(self.exe)), creationflags=no_window_flags())
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("The speech server could not start (is the speech model valid?).")
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.3).close()
                return
            except OSError:
                time.sleep(0.15)
        self.stop()
        raise RuntimeError("The speech server took too long to start.")

    def transcribe(self, pcm16: np.ndarray) -> str:
        boundary = "----ideawood" + str(int(time.time() * 1000))
        fields = {"temperature": "0.0", "response_format": "json"}
        body = b""
        for k, v in fields.items():
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes(pcm16) + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/inference", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return str(json.loads(r.read().decode("utf-8", "replace")).get("text", ""))

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None


_NONSPEECH = re.compile(r"^\s*[\[\(\*♪♫].*[\]\)\*♪♫]\s*$")
_PHANTOM = {"thank you.", "thanks for watching.", "you", "bye.", "thank you for watching.", "the end."}


def clean_text(text: str, seg_seconds: float) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t or _NONSPEECH.match(t):
        return ""
    if seg_seconds < 1.4 and t.lower() in _PHANTOM:       # classic whisper hallucination on short noise
        return ""
    return t


# --------------------------------------------------------------------------- the live captioner + sync monitor
class LiveCaptioner:
    def __init__(self, transcribe: Callable[[np.ndarray], str], on_caption: Callable[[str, float], None],
                 on_state: Callable[[str, str], None] = lambda state, msg: None,
                 lag_limit: float = 8.0, backlog_limit: float = 12.0, segmenter: Optional[Segmenter] = None):
        self.transcribe, self.on_caption, self.on_state = transcribe, on_caption, on_state
        self.lag_limit, self.backlog_limit = lag_limit, backlog_limit
        self.seg = segmenter or Segmenter()
        self.q: "collections.deque[Segment]" = collections.deque()
        self.cv = threading.Condition()
        self.captions: List[tuple] = []            # (t_start, t_end, text) on the audio clock -> live SRT / fallback
        self.degraded = False
        self.lag = 0.0
        self.rtf = 0.0
        self._bad = 0
        self._stop = False
        self._wall0: Optional[float] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, pcm48: bytes):
        """Called from the microphone thread (every 20 ms)."""
        if self._wall0 is None:
            self._wall0 = time.monotonic()
        if self.degraded:
            return
        segs = self.seg.push(to_16k(pcm48))
        if segs:
            with self.cv:
                self.q.extend(segs)
                backlog = sum(s.t_end - s.t_start for s in self.q)
                if backlog > self.backlog_limit:
                    self._degrade(f"Live captions fell {backlog:.0f} s behind")
                self.cv.notify()

    def _degrade(self, why: str):
        if not self.degraded:
            self.degraded = True
            self.q.clear()
            self.on_state("behind", f"{why}. Live captions are paused so your stream stays smooth - the full "
                                    "captions will be created from the recording when you stop.")

    def _run(self):
        while not self._stop:
            with self.cv:
                while not self.q and not self._stop:
                    self.cv.wait(0.2)
                if self._stop:
                    return
                seg = self.q.popleft()
            if self.degraded:
                continue
            dur = seg.t_end - seg.t_start
            t0 = time.monotonic()
            try:
                text = clean_text(self.transcribe(seg.pcm), dur)
            except Exception as e:  # noqa: BLE001
                self._degrade(f"Live captions stopped ({e})")
                return
            took = time.monotonic() - t0
            self.rtf = took / max(dur, 0.1) if self.rtf == 0 else self.rtf * 0.7 + (took / max(dur, 0.1)) * 0.3
            self.lag = max(0.0, (time.monotonic() - (self._wall0 or t0)) - seg.t_end)
            self._bad = self._bad + 1 if self.lag > self.lag_limit else 0
            if self._bad >= 2:
                self._degrade(f"Live captions are {self.lag:.0f} s late")
                return
            if text:
                self.captions.append((seg.t_start, seg.t_end, text))
                self.on_caption(text, self.lag)

    def stop(self):
        self._stop = True
        tail = self.seg.flush()
        with self.cv:
            self.cv.notify_all()
        self._thread.join(timeout=3)
        return tail


def live_srt(captions: List[tuple]) -> str:
    from .asr import Cue, to_srt
    return to_srt([Cue(a, max(b, a + 1.0), t) for a, b, t in captions])
