"""The live engine: newest camera frame -> (AI background removal) -> composite -> captions -> every output."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .compose import Backdrop, composite
from .matte import LiveMatter
from .sources import FrameSource


@dataclass
class LiveConfig:
    fps: int = 30
    quality: str = "Balanced"
    edge: float = 0.2
    mirror: bool = False
    feather: bool = True


class LivePipeline:
    def __init__(self, source: FrameSource, backdrop: Backdrop, cfg: LiveConfig,
                 sinks: Optional[List[Callable[[np.ndarray], None]]] = None, overlay=None):
        self.source, self.backdrop, self.cfg, self.sinks, self.overlay = source, backdrop, cfg, sinks or [], overlay
        self.matter: Optional[LiveMatter] = None
        self.error = ""
        self.frames = 0
        self.fps = 0.0
        self.sink_errors: List[str] = []
        self._preview: Optional[np.ndarray] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ---- control
    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5)

    def preview(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._preview

    def stats(self) -> dict:
        m = self.matter
        return {"fps": round(self.fps, 1), "frames": self.frames, "matte_ms": round(m.ms, 1) if m else 0.0,
                "matte_skip": m.skip if m else 1, "error": self.error}

    # ---- loop
    def _run(self):
        try:
            seq = -1
            budget = 1000.0 / self.cfg.fps
            t_win, n_win = time.monotonic(), 0
            while not self._stop:
                frame, seq = self.source.latest(seq, timeout=0.5)
                if frame is None:
                    if self.source.error:
                        self.error = "The camera/video stopped: " + self.source.error
                        return
                    continue
                if self.cfg.mirror:
                    frame = np.ascontiguousarray(frame[:, ::-1])
                if self.backdrop.needs_matting():
                    if self.matter is None:
                        self.matter = LiveMatter(self.cfg.quality)
                    self.matter.set_quality(self.cfg.quality)
                    a = self.matter.alpha(frame, budget, self.cfg.edge)
                    out = composite(frame, a, self.backdrop.frame_bg(frame), self.cfg.feather)
                else:
                    out = frame
                if self.overlay is not None:
                    out = self.overlay.apply(out)
                for sink in self.sinks:
                    try:
                        sink(out)
                    except Exception as e:  # noqa: BLE001
                        self.sink_errors.append(str(e))
                        self.sinks.remove(sink)
                with self._lock:
                    self._preview = out
                self.frames += 1
                n_win += 1
                now = time.monotonic()
                if now - t_win >= 1.0:
                    self.fps, t_win, n_win = n_win / (now - t_win), now, 0
        except Exception as e:  # noqa: BLE001
            self.error = f"Live processing stopped: {e}"
