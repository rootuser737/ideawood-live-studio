"""Small ffmpeg/ffprobe helpers (probe + a progress-reporting runner)."""
from __future__ import annotations

import json
import re
import subprocess
from typing import Callable, List, Optional

from .paths import find_tool, no_window_flags


def media_info(path: str) -> dict:
    try:
        r = subprocess.run([find_tool("ffprobe"), "-v", "error", "-show_entries",
                            "stream=codec_type,width,height:stream_disposition=attached_pic:format=duration",
                            "-of", "json", path], capture_output=True, text=True, errors="replace",
                           creationflags=no_window_flags())
        d = json.loads(r.stdout or "{}")
    except Exception:  # noqa: BLE001
        d = {}
    vids = [s for s in d.get("streams", []) if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")]
    return {"has_video": bool(vids), "has_audio": any(s.get("codec_type") == "audio" for s in d.get("streams", [])),
            "w": int(vids[0].get("width", 0)) if vids else 0, "h": int(vids[0].get("height", 0)) if vids else 0,
            "duration": float((d.get("format") or {}).get("duration") or 0)}


def run_ffmpeg(cmd: List[str], duration: float, on_progress: Callable[[float], None] = lambda p: None,
               cwd: Optional[str] = None, should_cancel: Callable[[], bool] = lambda: False) -> None:
    """Run an ffmpeg command that was built with `-progress pipe:1`; raises RuntimeError with the tail of stderr."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                         creationflags=no_window_flags(), cwd=cwd)
    err: List[str] = []
    import threading

    def drain():
        for line in p.stderr:
            err.append(line)
    t = threading.Thread(target=drain, daemon=True)
    t.start()
    for line in p.stdout:
        m = re.match(r"out_time_(?:us|ms)=(\d+)", line)
        if m:
            on_progress(min(99.0, int(m.group(1)) / 1_000_000.0 / max(duration, 0.1) * 100.0))
        if should_cancel():
            p.kill()
            break
    p.wait()
    t.join(timeout=2)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + "".join(err)[-400:])
    on_progress(100.0)
