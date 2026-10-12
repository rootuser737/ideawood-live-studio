"""Backgrounds and compositing (numpy + OpenCV)."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

MODES = ["Original (no removal)", "Blur my room", "Solid colour", "Picture"]


def hex_to_rgb(c: str) -> Tuple[int, int, int]:
    c = c.strip().lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def load_picture(path: str, w: int, h: int) -> np.ndarray:
    """Read an image and fit it to w x h by cropping (cover)."""
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read that picture.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    img = cv2.resize(img, (int(round(iw * s)), int(round(ih * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y0, x0 = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])


def blur_background(rgb: np.ndarray, strength: int = 25) -> np.ndarray:
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(2, w // 4), max(2, h // 4)), interpolation=cv2.INTER_AREA)
    k = max(3, (strength // 2) * 2 + 1)
    small = cv2.GaussianBlur(small, (k, k), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def composite(fg_rgb: np.ndarray, alpha: np.ndarray, bg_rgb: np.ndarray, feather: bool = True) -> np.ndarray:
    """uint8 result = fg*alpha + bg*(1-alpha), in fixed point (fast)."""
    if feather:
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.9)
    a = (np.clip(alpha, 0.0, 1.0) * 256.0).astype(np.uint16)[..., None]
    return ((fg_rgb.astype(np.uint16) * a + bg_rgb.astype(np.uint16) * (256 - a)) >> 8).astype(np.uint8)


class Backdrop:
    """What replaces the background. Call `frame_bg(frame)` each frame (only 'Blur' depends on the frame)."""

    def __init__(self, mode: str = MODES[0], color: str = "#00b140", picture: Optional[str] = None, blur: int = 25):
        self.mode, self.color, self.picture, self.blur = mode, color, picture, blur
        self._cache: Optional[np.ndarray] = None
        self._key = None

    def needs_matting(self) -> bool:
        return self.mode != MODES[0]

    def frame_bg(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.mode == "Blur my room":
            return blur_background(frame, self.blur)
        key = (self.mode, self.color, self.picture, w, h)
        if key != self._key or self._cache is None:
            if self.mode == "Picture" and self.picture:
                self._cache = load_picture(self.picture, w, h)
            else:
                c = np.empty((h, w, 3), np.uint8)
                c[:] = hex_to_rgb(self.color)
                self._cache = c
            self._key = key
        return self._cache
