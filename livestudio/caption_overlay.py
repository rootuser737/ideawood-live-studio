"""Draws the live caption (white text on a soft dark panel) onto outgoing frames. Needs Qt only for text rendering."""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class CaptionOverlay:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self._rgba: Optional[np.ndarray] = None
        self._y0 = 0
        self._until = 0.0
        self._lock = threading.Lock()

    def set_text(self, text: str, hold: Optional[float] = None):
        text = (text or "").strip()
        if not text:
            self.clear()
            return
        arr, y0 = self._render(text)
        hold = hold if hold is not None else max(2.5, min(8.0, 0.07 * len(text)))
        with self._lock:
            self._rgba, self._y0, self._until = arr, y0, time.monotonic() + hold

    def clear(self):
        with self._lock:
            self._rgba, self._until = None, 0.0

    def _render(self, text: str):
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter
        w, h = self.w, self.h
        size = max(16, int(h * 0.052))
        font = QFont("Segoe UI")
        font.setPixelSize(size)
        font.setWeight(QFont.DemiBold)
        fm = QFontMetrics(font)
        max_w = int(w * 0.84)
        flags = int(Qt.AlignCenter | Qt.TextWordWrap)
        box = fm.boundingRect(0, 0, max_w, 10_000, flags, text)
        lines = max(1, round(box.height() / fm.lineSpacing()))
        if lines > 2:                                   # keep the last two lines only (newest words)
            words = text.split()
            while len(words) > 1 and round(fm.boundingRect(0, 0, max_w, 10_000, flags, " ".join(words)).height() / fm.lineSpacing()) > 2:
                words.pop(0)
            text = " ".join(words)
            box = fm.boundingRect(0, 0, max_w, 10_000, flags, text)
        pad = int(size * 0.55)
        bh = box.height() + 2 * pad
        img = QImage(w, bh, QImage.Format_RGBA8888)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        pw = min(w - 8, box.width() + 2 * pad)
        px = (w - pw) / 2
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 165))
        p.drawRoundedRect(QRectF(px, 0, pw, bh), pad * 0.6, pad * 0.6)
        p.setFont(font)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF((w - max_w) / 2, pad, max_w, box.height()), flags, text)
        p.end()
        ptr = img.constBits()
        arr = np.frombuffer(ptr, np.uint8).reshape(bh, img.bytesPerLine() // 4, 4)[:, :w].copy()
        return arr, int(h * 0.93) - bh

    def apply(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            rgba, y0, until = self._rgba, self._y0, self._until
        if rgba is None or time.monotonic() > until:
            return frame
        out = frame.copy()
        y0 = max(0, min(y0, self.h - rgba.shape[0]))
        reg = out[y0:y0 + rgba.shape[0]]
        a = rgba[..., 3:4].astype(np.uint16)
        reg[:] = ((rgba[..., :3].astype(np.uint16) * a + reg.astype(np.uint16) * (255 - a)) // 255).astype(np.uint8)
        return out
