"""Outputs: RTMP stream (YouTube, Facebook, Twitch, TikTok, any RTMP server) + local recording, and the virtual camera
(Zoom / Google Meet / Microsoft Teams / any app that lists webcams)."""
from __future__ import annotations

import collections
import os
import queue
import socket
import subprocess
import threading
import time
from typing import List, Optional

import numpy as np

from .paths import find_tool, no_window_flags

# Streaming destinations (server URL; the stream key is appended). Zoom/Meet/Teams use the virtual camera instead.
DESTINATIONS = {
    "YouTube Live": "rtmp://a.rtmp.youtube.com/live2",
    "Facebook Live": "rtmps://live-api-s.facebook.com:443/rtmp/",
    "Twitch": "rtmp://live.twitch.tv/app",
    "TikTok LIVE (paste the server URL TikTok gives you)": "",
    "Other RTMP server (paste URL)": "",
}


def rtmp_url(server: str, key: str) -> str:
    server = server.strip()
    key = key.strip()
    if not server:
        raise ValueError("Enter the server URL for this destination.")
    if not server.lower().startswith(("rtmp://", "rtmps://")):
        raise ValueError("The server URL must start with rtmp:// or rtmps://")
    return server.rstrip("/") + "/" + key if key else server


# --------------------------------------------------------------------------- audio bridge (python -> ffmpeg over localhost)
class AudioBridge:
    """Feeds PCM to the streamer's ffmpeg through a local TCP socket, always at real-time pace.
    Mic data comes in via push(); if the mic is silent/absent, zeros are sent so the audio track never stalls."""
    RATE, CHUNK = 48000, 1920            # 20 ms of 16-bit mono

    def __init__(self):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.q: collections.deque = collections.deque(maxlen=60)   # <= 1.2 s of buffered audio
        self.muted = False
        self._stop = False
        self._buf = b""
        threading.Thread(target=self._run, daemon=True).start()

    def push(self, pcm: bytes):
        self.q.append(pcm)

    def _run(self):
        try:
            self.srv.settimeout(15)
            conn, _ = self.srv.accept()
        except OSError:
            return
        silence = b"\0" * self.CHUNK
        nxt = time.monotonic()
        try:
            while not self._stop:
                chunk = self.q.popleft() if self.q else silence
                conn.sendall(silence if self.muted else chunk)
                nxt += 0.02 * (len(chunk) / self.CHUNK) if len(chunk) == self.CHUNK else 0.02
                delay = nxt - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.5:
                    nxt = time.monotonic()
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self._stop = True
        try:
            self.srv.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- streamer + recorder
class Streamer:
    def __init__(self, w: int, h: int, fps: int, urls: List[str], record_path: Optional[str] = None,
                 video_kbps: int = 3500, ffmpeg_extra: Optional[List[str]] = None):
        if not urls and not record_path:
            raise ValueError("Nothing to stream or record to.")
        self.w, self.h, self.fps, self.urls, self.record_path, self.kbps = w, h, fps, urls, record_path, video_kbps
        self.proc: Optional[subprocess.Popen] = None
        self.bridge: Optional[AudioBridge] = None
        self.dropped = 0
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=4)
        self._log_path = (record_path + ".log") if record_path else os.path.join(os.getcwd(), "stream.log")
        self._log = None
        self._extra = ffmpeg_extra or []
        self._stop = False

    # ---- command (public so tests can inspect it)
    def build_cmd(self, port: int) -> List[str]:
        fps, kb = self.fps, self.kbps
        cmd = [find_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "warning",
               "-use_wallclock_as_timestamps", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-video_size", f"{self.w}x{self.h}", "-framerate", str(fps), "-i", "pipe:0",
               "-use_wallclock_as_timestamps", "1", "-f", "s16le", "-ar", str(AudioBridge.RATE), "-ac", "1",
               "-i", f"tcp://127.0.0.1:{port}",
               "-filter_complex", f"[0:v]fps={fps},format=yuv420p[v];"
                                  f"[1:a]aresample=async=1:first_pts=0,aformat=sample_rates=44100:channel_layouts=stereo[a]",
               "-map", "[v]", "-map", "[a]",
               "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-b:v", f"{kb}k",
               "-maxrate", f"{kb}k", "-bufsize", f"{kb * 2}k", "-g", str(fps * 2), "-keyint_min", str(fps),
               "-sc_threshold", "0", "-c:a", "aac", "-b:a", "128k",
               "-flags", "+global_header",          # required: AAC extradata for both FLV (stream) and Matroska (recording)
               *self._extra]
        outs = [f"[f=flv:onfail=ignore]{u}" for u in self.urls]
        if self.record_path:
            outs.append(f"[f=matroska]{self.record_path}")
        if len(outs) == 1 and self.record_path and not self.urls:
            return cmd + ["-f", "matroska", self.record_path]
        return cmd + ["-f", "tee", "|".join(outs)]

    def start(self):
        if self.record_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.record_path)), exist_ok=True)
        self.bridge = AudioBridge()
        self._log = open(self._log_path, "ab")
        self.proc = subprocess.Popen(self.build_cmd(self.bridge.port), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self._log, creationflags=no_window_flags())
        threading.Thread(target=self._writer, daemon=True).start()
        time.sleep(0.5)
        err = self.error()
        if err:
            raise RuntimeError(err)

    def _writer(self):
        while True:
            data = self._q.get()
            if data is None or self.proc is None:
                return
            try:
                self.proc.stdin.write(data)
            except (BrokenPipeError, OSError, ValueError):
                return

    def write_frame(self, rgb: np.ndarray):
        """Non-blocking: if the network/encoder is behind, the frame is dropped instead of freezing the app."""
        try:
            self._q.put_nowait(rgb.tobytes())
        except queue.Full:
            self.dropped += 1

    def write_audio(self, pcm: bytes):
        if self.bridge:
            self.bridge.push(pcm)

    def set_muted(self, m: bool):
        if self.bridge:
            self.bridge.muted = m

    def error(self) -> str:
        if self.proc and self.proc.poll() is not None and not self._stop:
            return self.log_tail() or "The streaming engine stopped unexpectedly."
        return ""

    def log_tail(self, n: int = 6) -> str:
        try:
            with open(self._log_path, "rb") as f:
                return "\n".join(f.read().decode("utf-8", "replace").strip().splitlines()[-n:])
        except OSError:
            return ""

    def stop(self):
        self._stop = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            if self.bridge:
                self.bridge.close()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log:
            self._log.close()


# --------------------------------------------------------------------------- virtual camera
class VirtualCamError(Exception):
    pass


HELP_WINDOWS = ("The virtual camera is not installed. One-time setup: install OBS Studio (free, obsproject.com), open it "
                "once and click 'Start Virtual Camera', then close it. After that Ideawood Live Studio can fill the "
                "'OBS Virtual Camera' device that Zoom, Meet and Teams list as a camera.")


class VirtualCamSink:
    def __init__(self, w: int, h: int, fps: int, factory=None):
        self.w, self.h, self.fps = w, h, fps
        self._factory = factory
        self.cam = None
        self.device = ""

    def start(self):
        try:
            if self._factory:
                self.cam = self._factory(self.w, self.h, self.fps)
            else:
                import pyvirtualcam
                self.cam = pyvirtualcam.Camera(width=self.w, height=self.h, fps=self.fps,
                                               fmt=pyvirtualcam.PixelFormat.RGB)
            self.device = getattr(self.cam, "device", "")
        except Exception as e:  # noqa: BLE001
            raise VirtualCamError(f"{HELP_WINDOWS if os.name == 'nt' else 'No virtual camera driver found (Linux: v4l2loopback).'}"
                                  f"\n({e})") from e

    def send(self, rgb: np.ndarray):
        if self.cam is not None:
            self.cam.send(rgb)

    def stop(self):
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:  # noqa: BLE001
                pass
            self.cam = None
