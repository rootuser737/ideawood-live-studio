"""Live AI background removal: Robust Video Matting (ONNX, CPU).

Speed design (measured): the model runs on a SMALL copy of the frame, its mask is scaled up and applied to the
full-quality picture. Cost is dominated by the model's internal size, and mask quality barely changes between
256 px and 512 px working width. If the PC cannot keep up, the model runs on every 2nd/3rd frame and the previous mask
is reused, so the video itself stays smooth.
"""
from __future__ import annotations

import os
import time
from typing import Tuple

import cv2
import numpy as np

from .paths import model_path

# name -> working width in pixels for the model (height follows the picture's shape)
QUALITY = {"Fast": 256, "Balanced": 320, "Best": 448}


def available() -> Tuple[bool, str]:
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"AI runtime missing ({e})"
    if not os.path.isfile(model_path()):
        return False, "AI model file missing (models/rvm_mobilenetv3_fp32.onnx)"
    return True, ""


def work_size(w: int, h: int, width: int) -> Tuple[int, int]:
    ww = max(64, min(width, w))
    hh = max(36, int(round(ww * h / w)))
    return ww - ww % 2, hh - hh % 2


class LiveMatter:
    def __init__(self, quality: str = "Balanced", threads: int | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, min(6, (os.cpu_count() or 2) - 1))
        self.sess = ort.InferenceSession(model_path(), so, providers=["CPUExecutionProvider"])
        self.ratio = np.array([1.0], np.float32)
        self.set_quality(quality)
        self.reset()
        self.ms = 0.0               # smoothed model time per run (ms)
        self.skip = 1               # run the model every N frames (adaptive)
        self._n = 0
        self._alpha: np.ndarray | None = None

    def set_quality(self, quality: str):
        self.width = QUALITY.get(quality, 320)

    def reset(self):
        """Forget previous frames (new camera / scene)."""
        self.rec = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]
        self._alpha = None
        self._n = 0

    def _run(self, small_rgb: np.ndarray) -> np.ndarray:
        src = (small_rgb.astype(np.float32) * (1.0 / 255.0)).transpose(2, 0, 1)[None]
        _fgr, pha, *self.rec = self.sess.run(None, {"src": src, "r1i": self.rec[0], "r2i": self.rec[1],
                                                    "r3i": self.rec[2], "r4i": self.rec[3],
                                                    "downsample_ratio": self.ratio})
        return pha[0, 0]

    def alpha(self, rgb: np.ndarray, budget_ms: float = 33.0, edge: float = 0.0, adaptive: bool = True) -> np.ndarray:
        """Mask (float32 0..1, same size as `rgb`). `budget_ms` = time available per frame (1000/fps)."""
        h, w = rgb.shape[:2]
        self._n += 1
        if self._alpha is None or self._n % self.skip == 0:
            ww, hh = work_size(w, h, self.width)
            small = cv2.resize(rgb, (ww, hh), interpolation=cv2.INTER_AREA)
            t = time.perf_counter()
            a = self._run(small)
            dt = (time.perf_counter() - t) * 1000.0
            self.ms = dt if self.ms == 0 else self.ms * 0.8 + dt * 0.2
            if adaptive:                       # keep the video smooth: model every Nth frame when it is too slow
                need = self.ms / max(budget_ms * 0.7, 1.0)
                self.skip = 1 if need <= 1.0 else (2 if need <= 2.0 else 3)
            if edge > 0:
                k = 1.0 + edge * 3.0
                a = np.clip((a - 0.5) * k + 0.5, 0.0, 1.0)
            self._alpha = a
        a = self._alpha
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        return a
