"""Offline speech recognition (ASR) and subtitles, powered by whisper.cpp.

* whisper-cli (bundled in the EXE under bin/whisper/) does the recognition on the CPU - nothing is uploaded.
* The speech *model* (75-466 MB) is downloaded once from Hugging Face into %LOCALAPPDATA%/IdeawoodStudio/models
  (or choose a model file you already have), so the EXE stays small and the app works offline afterwards.
* Output is parsed from ``whisper-cli -ojf`` JSON (segments + per-word timestamps) into captions ("cues") that are
  split for readability, then written as SRT / VTT / TXT or a styled ASS file for burning into video.

No Qt imports: everything here is unit-testable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .paths import find_tool, no_window_flags

# key -> (label, file name, approx MB, multilingual)
MODELS: Dict[str, Tuple[str, str, int, bool]] = {
    "tiny.en": ("Tiny, English only (75 MB) - fastest", "ggml-tiny.en.bin", 75, False),
    "base.en": ("Base, English only (142 MB) - good balance", "ggml-base.en.bin", 142, False),
    "small.en": ("Small, English only (466 MB) - most accurate", "ggml-small.en.bin", 466, False),
    "base": ("Base, 99 languages (142 MB)", "ggml-base.bin", 142, True),
    "small": ("Small, 99 languages (466 MB) - better for other languages", "ggml-small.bin", 466, True),
}
MODEL_URL = os.environ.get("IDEAWOOD_MODEL_URL", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file}")

# Whisper language codes (no Luganda: Whisper does not support it). Multilingual models only.
LANGUAGES = {
    "Auto-detect": "auto", "English": "en", "Swahili": "sw", "French": "fr", "Spanish": "es", "Portuguese": "pt",
    "German": "de", "Italian": "it", "Arabic": "ar", "Hindi": "hi", "Chinese": "zh", "Japanese": "ja", "Korean": "ko",
    "Russian": "ru", "Turkish": "tr", "Dutch": "nl", "Polish": "pl", "Ukrainian": "uk", "Indonesian": "id",
    "Afrikaans": "af", "Amharic": "am", "Yoruba": "yo", "Shona": "sn", "Somali": "so", "Lingala": "ln",
    "Hausa": "ha", "Bengali": "bn", "Urdu": "ur", "Vietnamese": "vi", "Thai": "th", "Greek": "el",
}


class AsrError(Exception):
    pass


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- locating tools and models
def models_dir() -> str:
    d = os.environ.get("IDEAWOOD_MODELS")
    if not d:
        if os.name == "nt":
            d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "IdeawoodStudio", "models")
        else:
            d = os.path.join(os.path.expanduser("~"), ".local", "share", "IdeawoodStudio", "models")
    os.makedirs(d, exist_ok=True)
    return d


def model_file(key: str) -> str:
    return os.path.join(models_dir(), MODELS[key][1])


def bundled_models_dirs() -> List[str]:
    """Folders holding models shipped *inside* the app (EXE) - no download needed for these."""
    dirs = []
    if os.environ.get("IDEAWOOD_BUNDLED_MODELS"):
        dirs.append(os.environ["IDEAWOOD_BUNDLED_MODELS"])
    base = getattr(sys, "_MEIPASS", None)
    if base:
        dirs.append(os.path.join(base, "models", "whisper"))
    dirs.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "whisper"))
    return dirs


def _ok(p: str) -> bool:
    return os.path.isfile(p) and os.path.getsize(p) > 10_000_000


def find_model(key: str) -> Optional[str]:
    """Path of an installed model: the user's downloaded copy first, otherwise one built into the app."""
    fname = MODELS[key][1]
    for d in [models_dir()] + bundled_models_dirs():
        p = os.path.join(d, fname)
        if _ok(p):
            return p
    return None


def is_bundled(key: str) -> bool:
    p = find_model(key)
    return bool(p) and os.path.dirname(p) in bundled_models_dirs()


def installed_models() -> List[str]:
    return [k for k in MODELS if find_model(k)]


def find_whisper_cli() -> str:
    exe = "whisper-cli" + (".exe" if os.name == "nt" else "")
    cands = []
    base = getattr(sys, "_MEIPASS", None)
    if base:
        cands.append(os.path.join(base, "bin", "whisper", exe))
    cands.append(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "bin", "whisper", exe))
    cands.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "whisper", exe))
    if os.environ.get("WHISPER_CLI"):
        cands.insert(0, os.environ["WHISPER_CLI"])
    for c in cands:
        if os.path.isfile(c):
            return c
    found = shutil.which("whisper-cli")
    if found:
        return found
    raise AsrError("The speech engine (whisper-cli) was not found in this build.")


def download_model(key: str, on_progress: Callable[[float], None] = lambda p: None,
                   should_cancel: Callable[[], bool] = lambda: False) -> str:
    """Download a model (resumable). Returns the final path."""
    label, fname, approx_mb, _ = MODELS[key]
    dest = model_file(key)
    part = dest + ".part"
    url = MODEL_URL.format(file=fname)
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "IdeawoodStudio", **({"Range": f"bytes={have}-"} if have else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            code = getattr(r, "status", 200)
            if code == 200:
                have = 0                       # server ignored Range: start over
            total = have + int(r.headers.get("Content-Length") or 0)
            mode = "ab" if have else "wb"
            with open(part, mode) as f:
                while True:
                    if should_cancel():
                        raise Cancelled()
                    chunk = r.read(1 << 18)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if total:
                        on_progress(min(100.0, have * 100.0 / total))
    except Cancelled:
        raise
    except Exception as e:  # noqa: BLE001
        raise AsrError(f"Could not download the speech model: {e}. Check your internet connection and try again "
                       f"(the download resumes where it stopped).") from e
    size = os.path.getsize(part)
    if size < approx_mb * 1_000_000 * 0.6:
        raise AsrError("The downloaded model file looks incomplete. Try again.")
    os.replace(part, dest)
    on_progress(100.0)
    return dest


# --------------------------------------------------------------------------- transcript model
@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str = ""
    segments: List[Segment] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())


_SPECIAL = re.compile(r"^(\[_[^\]]*\]|<\|.*\|>)$")      # [_BEG_], [_TT_400], <|endoftext|> ...
_NONSPEECH = re.compile(r"^\s*[\[\(\*♪♫].*[\]\)\*♪♫]\s*$")


def parse_whisper_json(data: dict, hide_nonspeech: bool = True) -> Transcript:
    """Parse the JSON written by ``whisper-cli -ojf`` (layout verified against whisper.cpp's cli.cpp)."""
    tr = Transcript(language=(data.get("result") or {}).get("language", ""))
    for seg in data.get("transcription") or []:
        off = seg.get("offsets") or {}
        s0, s1 = off.get("from", 0) / 1000.0, off.get("to", 0) / 1000.0
        text = (seg.get("text") or "").strip()
        if not text or (hide_nonspeech and _NONSPEECH.match(text)):
            continue
        words: List[Word] = []
        for tok in seg.get("tokens") or []:
            t = tok.get("text", "")
            if not t or _SPECIAL.match(t.strip()):
                continue
            o = tok.get("offsets") or {}
            a, b = (o["from"] / 1000.0, o["to"] / 1000.0) if "from" in o and "to" in o else (None, None)
            if t.startswith(" ") or not words:
                words.append(Word(a if a is not None else -1, b if b is not None else -1, t.strip()))
            else:                                      # sub-word piece or punctuation: glue to the previous word
                w = words[-1]
                w.text += t
                if b is not None:
                    w.end = b
                if w.start < 0 and a is not None:
                    w.start = a
        words = [w for w in words if w.text.strip()]
        _repair_word_times(words, s0, s1)
        tr.segments.append(Segment(s0, max(s1, s0 + 0.2), text, words))
    return tr


def _repair_word_times(words: List[Word], s0: float, s1: float):
    """Fill missing/invalid word times and keep them inside the segment and in order."""
    n = len(words)
    if not n:
        return
    if all(w.start < 0 or w.end < 0 for w in words):
        total = sum(max(len(w.text), 1) for w in words)
        t = s0
        for w in words:
            d = (s1 - s0) * max(len(w.text), 1) / total
            w.start, w.end = t, t + d
            t += d
        return
    prev_end = s0
    for i, w in enumerate(words):
        if w.start < 0:
            w.start = prev_end
        if w.end < 0 or w.end < w.start:
            nxt = next((x.start for x in words[i + 1:] if x.start >= 0), s1)
            w.end = max(w.start + 0.05, min(nxt, s1))
        w.start = max(w.start, prev_end - 0.0001) if w.start < prev_end else w.start
        prev_end = max(prev_end, w.end)


# --------------------------------------------------------------------------- captions (cues)
@dataclass
class Cue:
    start: float
    end: float
    text: str                                  # may contain "\n" for the 2nd line
    words: List[Word] = field(default_factory=list)


_SENT_END = re.compile(r"[.!?…]['\")\]]*$")
_CLAUSE_END = re.compile(r"[,;:–—]['\")\]]*$")


def _balance_lines(text: str, max_chars: int, max_lines: int) -> str:
    text = text.strip()
    if len(text) <= max_chars or max_lines < 2:
        return text
    words = text.split()
    best, best_score = None, 1e9
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) > max_chars or len(b) > max_chars:
            continue
        score = abs(len(a) - len(b)) - (6 if _CLAUSE_END.search(a) else 0)
        if score < best_score:
            best, best_score = f"{a}\n{b}", score
    if best:
        return best
    mid = len(text) // 2                       # no clean fit: break at the space nearest the middle
    cut = text.rfind(" ", 0, mid + 1)
    if cut <= 0:
        cut = text.find(" ", mid)
    return text if cut <= 0 else text[:cut] + "\n" + text[cut + 1:]


def make_cues(tr: Transcript, max_chars: int = 42, max_lines: int = 2, max_dur: float = 6.5,
              min_dur: float = 1.0, gap_split: float = 0.9) -> List[Cue]:
    """Split the transcript into readable subtitle cues (<= max_lines lines of <= max_chars characters)."""
    cap = max_chars * max_lines
    cues: List[Cue] = []
    cur: List[Word] = []

    def flush():
        nonlocal cur
        if cur:
            txt = " ".join(w.text for w in cur)
            cues.append(Cue(cur[0].start, cur[-1].end, _balance_lines(txt, max_chars, max_lines), list(cur)))
        cur = []

    for seg in tr.segments:
        words = seg.words or _fake_words(seg)
        for w in words:
            if cur:
                length = len(" ".join(x.text for x in cur)) + 1 + len(w.text)
                prev = cur[-1]
                too_long = length > cap
                too_slow = (w.end - cur[0].start) > max_dur
                gap = (w.start - prev.end) > gap_split
                sentence = bool(_SENT_END.search(prev.text)) and length > cap * 0.45
                clause = bool(_CLAUSE_END.search(prev.text)) and length > cap * 0.8
                if too_long or too_slow or gap or sentence or clause:
                    flush()
            cur.append(w)
        flush()                                 # never carry a cue across a recognition segment boundary
    # timing polish: minimum duration, no overlaps
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else c.end + 5
        if c.end - c.start < min_dur:
            c.end = min(c.start + min_dur, max(nxt - 0.04, c.end))
        if c.end > nxt - 0.02:
            c.end = max(c.start + 0.2, nxt - 0.02)
    return cues


def _fake_words(seg: Segment) -> List[Word]:
    toks = seg.text.split()
    if not toks:
        return []
    total = sum(len(t) for t in toks)
    t, out = seg.start, []
    for tok in toks:
        d = (seg.end - seg.start) * len(tok) / total
        out.append(Word(t, t + d, tok))
        t += d
    return out


# --------------------------------------------------------------------------- writers
def _ts(sec: float, comma: bool) -> str:
    ms = int(round(max(sec, 0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{ms:03d}"


def to_srt(cues: List[Cue]) -> str:
    return "".join(f"{i}\n{_ts(c.start, True)} --> {_ts(c.end, True)}\n{c.text}\n\n" for i, c in enumerate(cues, 1))


def to_vtt(cues: List[Cue]) -> str:
    return "WEBVTT\n\n" + "".join(f"{_ts(c.start, False)} --> {_ts(c.end, False)}\n{c.text}\n\n" for c in cues)


def to_txt(tr_or_cues) -> str:
    if isinstance(tr_or_cues, Transcript):
        return tr_or_cues.text() + "\n"
    return "\n".join(c.text.replace("\n", " ") for c in tr_or_cues) + "\n"


def parse_srt(text: str) -> List[Cue]:
    cues = []
    for block in re.split(r"\n\s*\n", text.strip().replace("\r\n", "\n")):
        lines = block.split("\n")
        m = None
        for i, ln in enumerate(lines[:2]):
            m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", ln.strip())
            if m:
                txt = "\n".join(lines[i + 1:]).strip()
                a = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4].ljust(3, "0")[:3]) / 1000
                b = int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8].ljust(3, "0")[:3]) / 1000
                if txt:
                    cues.append(Cue(a, b, txt))
                break
    return cues


# --------------------------------------------------------------------------- styled burn-in (ASS)
POSITIONS = {"Bottom": 2, "Middle": 5, "Top": 8}
SIZES = {"Small": 0.045, "Medium": 0.06, "Large": 0.08}
COLORS = {"White": "FFFFFF", "Yellow": "00E5FF", "Light green": "7CFC9A", "Sky blue": "FFE066"}   # BGR hex for ASS


def font_px(width: int, height: int, size: str = "Medium") -> int:
    """Subtitle font size from the *narrower* side, so portrait and landscape both look right."""
    return max(14, int(min(height, width * 0.9) * SIZES.get(size, 0.06)))


def auto_max_chars(width: int, height: int, size: str = "Medium") -> int:
    """How many characters fit on one subtitle line for this video shape and text size."""
    fs = font_px(width, height, size)
    return int(max(18, min(48, (width * 0.88) / (0.58 * fs))))


def _ass_time(sec: float) -> str:
    cs = int(round(max(sec, 0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def to_ass(cues: List[Cue], width: int, height: int, position: str = "Bottom", size: str = "Medium",
           color: str = "White", box: bool = True, highlight: bool = False, font: str = "Arial",
           bold: bool = True) -> str:
    fs = font_px(width, height, size)
    align = POSITIONS.get(position, 2)
    margin_v = int(height * 0.06)
    margin_h = int(width * 0.06)
    base = COLORS.get(color, "FFFFFF")
    primary, secondary = (base, "FFFFFF") if highlight else (base, base)
    if highlight:
        primary, secondary = "00E5FF", "FFFFFF"            # spoken word turns yellow, others stay white
    back = "&H99000000" if box else "&H00000000"
    border_style = 3 if box else 1
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\n"
            f"ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{font},{fs},&H00{primary},&H00{secondary},&H00000000,{back},{-1 if bold else 0},0,0,0,"
            f"100,100,0,0,{border_style},{max(2, fs // 14)},0,{align},{margin_h},{margin_h},{margin_v},1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    lines = []
    for c in cues:
        if highlight and c.words and " ".join(w.text for w in c.words).split() == c.text.replace("\n", " ").split():
            parts = []
            nl_after = _line_break_after(c)
            for i, w in enumerate(c.words):
                nxt = c.words[i + 1].start if i + 1 < len(c.words) else w.end
                d = max(0.01, nxt - w.start)             # word i lights up at ITS start (ASS \k = slot length)
                parts.append(f"{{\\k{int(round(d * 100))}}}{_ass_escape(w.text)}")
                parts.append("\\N" if i == nl_after else " ")
            txt = "".join(parts).rstrip()
        else:
            txt = _ass_escape(c.text).replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Default,,0,0,0,,{txt}")
    return head + "\n".join(lines) + "\n"


def _line_break_after(c: Cue) -> int:
    """Index of the word after which the cue text has its line break (-1 if single line)."""
    if "\n" not in c.text:
        return -1
    first = c.text.split("\n")[0].split()
    return len(first) - 1


def has_real_video(path: str) -> bool:
    """True only for a real moving-picture stream. An MP3/M4A with cover art counts as audio-only."""
    try:
        r = subprocess.run([find_tool("ffprobe"), "-v", "error", "-select_streams", "v", "-show_entries",
                            "stream=codec_name:stream_disposition=attached_pic", "-of", "json", path],
                           capture_output=True, text=True, errors="replace", creationflags=no_window_flags())
        return any(not (st.get("disposition") or {}).get("attached_pic") for st in json.loads(r.stdout).get("streams", []))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- audio extraction + recognition
def extract_audio(src: str, dst_wav: str, should_cancel=lambda: False) -> float:
    """Any audio/video -> 16 kHz mono WAV (what whisper wants). Returns the duration in seconds."""
    ff = find_tool("ffmpeg")
    cmd = [ff, "-y", "-nostdin", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst_wav]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", creationflags=no_window_flags())
    if r.returncode != 0 or not os.path.exists(dst_wav):
        raise AsrError("Could not read audio from this file. " + (r.stderr or "").strip()[-200:])
    return os.path.getsize(dst_wav) / 32000.0       # 16 kHz * 2 bytes


class WhisperEngine:
    def __init__(self, model_path: str, cli: Optional[str] = None, threads: Optional[int] = None,
                 cmd_prefix: Optional[List[str]] = None):
        self.model_path = model_path
        self.cli = cli
        self.threads = threads or max(1, min(8, (os.cpu_count() or 4) - 1))
        self.cmd_prefix = cmd_prefix            # tests: run a fake CLI through [sys.executable, fake.py]

    def transcribe(self, wav: str, language: str = "auto", prompt: str = "",
                   on_progress: Callable[[float], None] = lambda p: None,
                   should_cancel: Callable[[], bool] = lambda: False, hide_nonspeech: bool = True) -> Transcript:
        if not os.path.isfile(self.model_path):
            raise AsrError("Speech model not found. Download or choose a model first.")
        work = tempfile.mkdtemp(prefix="ideawood_asr_")
        try:
            base = os.path.join(work, "out")
            cli = self.cmd_prefix or [self.cli or find_whisper_cli()]
            cmd = [*cli, "-m", self.model_path, "-f", wav, "-l", language or "auto", "-ojf", "-of", base,
                   "-t", str(self.threads), "-pp"]
            if prompt.strip():
                cmd += ["--prompt", prompt.strip()[:400]]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    errors="replace", creationflags=no_window_flags(),
                                    cwd=os.path.dirname(cli[-1]) if os.path.isabs(cli[-1]) else None)
            tail: List[str] = []
            for line in proc.stdout:
                m = re.search(r"progress\s*=\s*(\d+)\s*%", line)
                if m:
                    on_progress(float(m.group(1)))
                else:
                    tail.append(line.rstrip())
                    tail[:] = tail[-12:]
                if should_cancel():
                    proc.kill()
                    proc.wait()
                    raise Cancelled()
            rc = proc.wait()
            out = base + ".json"
            if rc != 0 or not os.path.exists(out):
                raise AsrError("Speech recognition failed.\n" + "\n".join(tail[-5:]))
            with open(out, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            on_progress(100.0)
            return parse_whisper_json(data, hide_nonspeech)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def transcribe_file(src: str, engine: WhisperEngine, language: str = "auto", prompt: str = "",
                    on_status: Callable[[str], None] = lambda s: None,
                    on_progress: Callable[[float], None] = lambda p: None,
                    should_cancel: Callable[[], bool] = lambda: False) -> Transcript:
    work = tempfile.mkdtemp(prefix="ideawood_wav_")
    try:
        on_status("Preparing the audio...")
        wav = os.path.join(work, "audio.wav")
        extract_audio(src, wav)
        if should_cancel():
            raise Cancelled()
        on_status("Recognising speech (this runs on your PC)...")
        return engine.transcribe(wav, language, prompt, on_progress, should_cancel)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- burning / embedding with ffmpeg
def escape_filter_value(p: str) -> str:
    """Escape a path for use as an option value inside an ffmpeg -vf string (verified: ':' needs TWO backslashes)."""
    return p.replace("\\", "/").replace(":", "\\\\:").replace("'", "\\\\'").replace(",", "\\\\,")


def _fontsdir() -> Optional[str]:
    if os.name == "nt":
        d = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        return escape_filter_value(d) if os.path.isdir(d) else None
    for d in ("/usr/share/fonts", "/usr/local/share/fonts"):
        if os.path.isdir(d):
            return escape_filter_value(d)
    return None


def burn_cmd(src: str, ass_name: str, dst: str, ff: Optional[str] = None, fontsdir: Optional[str] = None) -> List[str]:
    """Hard-burn an ASS file (given by *relative* name; run with cwd=its folder to avoid Windows path escaping)."""
    ff = ff or find_tool("ffmpeg")
    fd = escape_filter_value(fontsdir) if fontsdir else _fontsdir()
    vf = f"ass={ass_name}" + (f":fontsdir={fd}" if fd else "")
    return [ff, "-y", "-nostdin", "-progress", "pipe:1", "-nostats", "-i", src, "-vf", vf, "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", dst]


def embed_cmd(src: str, srt: str, dst: str, lang: str = "eng", ff: Optional[str] = None) -> List[str]:
    """Add the subtitles as a selectable track (viewers can switch them off); no re-encoding."""
    ff = ff or find_tool("ffmpeg")
    return [ff, "-y", "-nostdin", "-i", src, "-i", srt, "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text",
            f"-metadata:s:s:0", f"language={lang}", "-movflags", "+faststart", dst]
