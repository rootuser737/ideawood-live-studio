"""After the stream: remux the recording to MP4 and create full-quality captions from it (the fallback when live ASR
could not keep in sync - and a quality upgrade even when it could)."""
from __future__ import annotations

import os
import subprocess
from typing import Callable, Optional

from . import asr
from .paths import find_tool, no_window_flags


def remux_to_mp4(mkv: str) -> str:
    """Crash-safe MKV -> MP4 without re-encoding. Returns the MP4 path (keeps the MKV if remux fails)."""
    mp4 = os.path.splitext(mkv)[0] + ".mp4"
    r = subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-i", mkv, "-c", "copy", "-movflags", "+faststart", mp4],
                       capture_output=True, text=True, creationflags=no_window_flags())
    if r.returncode == 0 and os.path.exists(mp4) and os.path.getsize(mp4) > 1000:
        os.remove(mkv)
        return mp4
    return mkv


def make_final_captions(video: str, model_path: str, language: str = "en", prompt: str = "",
                        on_status: Callable[[str], None] = lambda s: None,
                        on_progress: Callable[[float], None] = lambda p: None,
                        should_cancel: Callable[[], bool] = lambda: False, engine: Optional[asr.WhisperEngine] = None,
                        max_chars: int = 42) -> dict:
    """-> {'srt': path, 'cues': [...], 'transcript': Transcript}"""
    eng = engine or asr.WhisperEngine(model_path)
    tr = asr.transcribe_file(video, eng, language, prompt, on_status, on_progress, should_cancel)
    cues = asr.make_cues(tr, max_chars=max_chars)
    srt = os.path.splitext(video)[0] + ".srt"
    with open(srt, "w", encoding="utf-8-sig") as f:
        f.write(asr.to_srt(cues))
    return {"srt": srt, "cues": cues, "transcript": tr}
