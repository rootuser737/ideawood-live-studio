"""Live captions: phrase detection, delivery in real time, the sync monitor / fallback, and the real whisper-server."""
import os, sys, time, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from livestudio.live_captions import (LiveCaptioner, Segmenter, WhisperServer, clean_text, live_srt, to_16k, wav_bytes)
from livestudio.sources import MicSource, wav_args

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)


def tone(sec, amp=4000):
    t = np.arange(int(sec * 16000)) / 16000
    return (amp * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.int16)


def quiet(sec):
    return (rng.normal(0, 40, int(sec * 16000))).astype(np.int16)


# 1) 48k -> 16k conversion keeps duration and level
pcm48 = (np.sin(np.arange(4800) / 20) * 5000).astype(np.int16).tobytes()
x = to_16k(pcm48); assert len(x) == 1600 and 2500 < np.abs(x).max() < 5100

# 2) segmenter: silence -> nothing; two phrases with a pause -> two segments with sane times; long speech is cut
sg = Segmenter()
assert sg.push(quiet(3.0)) == []
sg = Segmenter(); out = []
for chunk in (quiet(1.0), tone(1.6), quiet(1.0), tone(2.0), quiet(1.0)):
    out += sg.push(chunk)
assert len(out) == 2, [(round(s.t_start, 2), round(s.t_end, 2)) for s in out]
assert 0.7 < out[0].t_start < 1.05 and 2.5 < out[0].t_end < 2.9 and 3.2 < out[1].t_start < 3.7, [(s.t_start, s.t_end) for s in out]
sg = Segmenter(max_len=3.0); out = sg.push(tone(7.0)) + [s for s in [sg.flush()] if s]
assert len(out) >= 2 and all(s.t_end - s.t_start <= 3.2 for s in out)
sg = Segmenter(); assert sg.push(tone(0.2)) == [] and sg.flush() is None          # a click is not a phrase

# 3) text clean-up (hallucinations on noise are dropped, real short answers kept)
assert clean_text(" [BLANK_AUDIO] ", 2) == "" and clean_text("(music)", 2) == "" and clean_text("Thank you.", 0.8) == ""
assert clean_text("Thank you.", 3.0) == "Thank you." and clean_text("  hello   world ", 2) == "hello world"

# 4) real-time delivery with the real JFK audio (fake recogniser): captions arrive shortly after each phrase
got, states = [], []
cap = LiveCaptioner(lambda pcm: f"phrase of {len(pcm) / 16000:.1f}s", lambda t, lag: got.append((time.monotonic(), t, lag)),
                    lambda s, m: states.append((s, m)))
mic = MicSource(wav_args(os.path.join(HERE, "data", "jfk.wav")), cap.feed); mic.start()
time.sleep(13.5); mic.stop(); cap.stop()
assert len(got) >= 2 and not cap.degraded and not states, (got, states)
assert max(g[2] for g in got) < 2.0, [g[2] for g in got]                         # lag stays small
assert len(cap.captions) == len(got) and cap.captions[0][0] < cap.captions[-1][0]
srt = live_srt(cap.captions); assert srt.startswith("1\n00:00:") and "phrase of" in srt
print(f"JFK audio -> {len(got)} live captions, max lag {max(g[2] for g in got):.2f}s")

# 5) sync monitor: a recogniser that is far too slow -> live captions switch off with a notice (recording would continue)
states2 = []
cap2 = LiveCaptioner(lambda pcm: (time.sleep(4.0), "late")[1], lambda t, lag: None, lambda s, m: states2.append((s, m)),
                     lag_limit=5.0)
t_end = time.time() + 40
while not states2 and time.time() < t_end:                      # real-time audio: 1.2 s of speech, 0.7 s pause, repeat
    for ch in (tone(1.2), quiet(0.7)):
        for i in range(0, len(ch), 320):
            cap2.feed(np.repeat(ch[i:i + 320], 3).tobytes()); time.sleep(0.02)
        if states2: break
cap2.stop()
assert cap2.degraded and states2 and states2[0][0] == "behind" and "created from the recording" in states2[0][1], states2
print("sync monitor OK after", round(40 - (t_end - time.time())), "s:", states2[0][1][:70])

# 6) the real whisper-server: starts, accepts audio, answers (real words are checked in CI with a real model)
exe, model = os.environ.get("WHISPER_SERVER"), os.environ.get("IDEAWOOD_TEST_MODEL")
if exe and model and os.path.isfile(exe) and os.path.isfile(model):
    srv = WhisperServer(exe, model); t0 = time.time(); srv.start()
    w = wave.open(os.path.join(HERE, "data", "jfk.wav")); pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    t1 = time.time(); text = srv.transcribe(pcm); took = time.time() - t1
    print(f"whisper-server: up in {t1 - t0:.1f}s, {len(pcm) / 16000:.1f}s audio in {took:.1f}s -> {text!r}")
    if os.environ.get("IDEAWOOD_REAL_ASR") == "1":
        assert "ask not what your country can do for you" in text.lower(), text
        got2 = []
        cap3 = LiveCaptioner(srv.transcribe, lambda t, lag: got2.append((t, lag)))
        mic = MicSource(wav_args(os.path.join(HERE, "data", "jfk.wav")), cap3.feed); mic.start()
        time.sleep(16); mic.stop(); cap3.stop()
        joined = " ".join(t for t, _ in got2).lower()
        print("REAL LIVE CAPTIONS:", got2, "| degraded:", cap3.degraded)
        assert "country" in joined and not cap3.degraded, (joined, cap3.degraded)
    srv.stop()
print("live captions tests OK")
