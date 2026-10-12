"""The whole window, end to end: stand-in camera + microphone, background removal, virtual camera, RTMP stream,
recording, live captions, full captions afterwards, captioned copy. Only the hardware is replaced."""
import os, subprocess, sys, tempfile, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox
from livestudio import asr, ui
from livestudio.mediautil import media_info
from livestudio.paths import find_tool

HERE = os.path.dirname(os.path.abspath(__file__)); T = tempfile.mkdtemp(); ff = find_tool("ffmpeg")
app = QApplication.instance() or QApplication([])
QSettings("Ideawood", "LiveStudio").clear()
msgs = []
for n in ("information", "warning"):
    setattr(QMessageBox, n, staticmethod(lambda *a, **k: msgs.append(a[2])))


def wait(cond, s=60):
    t0 = time.time()
    while not cond() and time.time() - t0 < s:
        app.processEvents(); time.sleep(0.03)
    assert cond(), "timeout"


class FakeCam:
    device = "Fake Virtual Camera"
    def __init__(self, w, h, fps): self.frames = []; self.closed = False
    def send(self, f): self.frames.append(f.shape)
    def close(self): self.closed = True


cams = []
class FakeEngine:
    def transcribe(self, wav, language="auto", prompt="", on_progress=lambda p: None, should_cancel=lambda: False, hide_nonspeech=True):
        on_progress(100.0)
        return asr.parse_whisper_json({"result": {"language": "en"}, "transcription": [
            {"offsets": {"from": 0, "to": 3000}, "text": " And so my fellow Americans, ask not what your country can do.", "tokens": []}]})

asr.find_model = lambda k: "/fake/model.bin"; asr.installed_models = lambda: ["base.en"]; asr.is_bundled = lambda k: False
ui.MainWindow.hooks = {
    "source_args": ["-re", "-loop", "1", "-framerate", "15", "-i", os.path.join(HERE, "data", "person.png")],
    "mic_args": ["-re", "-i", os.path.join(HERE, "data", "jfk.wav")],
    "transcribe": lambda pcm: "Hello live caption",
    "vcam_factory": lambda w, h, fps: cams.append(FakeCam(w, h, fps)) or cams[-1],
    "final_engine": FakeEngine(),
}
w = ui.MainWindow(); w.show(); app.processEvents()
w.folder.setText(os.path.join(T, "rec")); w.res.setCurrentText("640 x 360 (lightest)"); w.fps.setCurrentText("15")
w.quality.setCurrentText("Fast"); w.mode.setCurrentText("Solid colour"); w.color = "#00b140"

# 1) stream with an empty key -> friendly message, nothing starts
w.stream_cb.setChecked(True); w.dest.setCurrentText("Other RTMP server (paste URL)"); w.server.setText("rtmp://127.0.0.1:19352/live")
w.go_live(); assert w.live is None and any("stream key" in m for m in msgs), msgs

# 2) go live: stream to a local RTMP server + record + virtual camera + live captions
recv = os.path.join(T, "received.flv")
srv = subprocess.Popen([ff, "-y", "-v", "error", "-listen", "1", "-f", "flv", "-i", "rtmp://127.0.0.1:19352/live/k", "-c", "copy", recv],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.0)
w.key.setText("k"); w.vcam_cb.setChecked(True)
w.go_live(); assert w.live is not None and w.stop_btn.isEnabled() and not w.go.isEnabled(), msgs
wait(lambda: w.live["pipe"].frames > 25 and w.live.get("last_caption"), 40)
assert w.live["last_caption"] == "Hello live caption"
wait(lambda: not w.preview.pixmap().isNull(), 10)
w.mute.setChecked(True); app.processEvents(); assert w.live["muted"]; w.mute.setChecked(False)
t0 = time.time()
while time.time() - t0 < 4: app.processEvents(); time.sleep(0.03)
assert "LIVE" in w.status.text() and "fps" in w.status.text(), w.status.text()
frames_seen = len(cams[0].frames)
w.stop_live()
wait(lambda: w.worker is None and w.last_output.endswith(".mp4"), 90)
srv.wait(timeout=20)

# 3) results: recording, received stream, virtual camera frames, full captions
v = media_info(w.last_output)
assert v["has_video"] and v["has_audio"] and (v["w"], v["h"]) == (640, 360) and v["duration"] > 5, v
r = media_info(recv); assert r["has_video"] and r["has_audio"] and r["duration"] > 5, r
assert frames_seen > 20 and set(cams[0].frames) == {(360, 640, 3)} and cams[0].closed
assert w.last_srt.endswith(".srt") and "ask not what your country" in open(w.last_srt, encoding="utf-8-sig").read()
assert not os.path.exists(os.path.splitext(w.last_output)[0] + ".mkv")                     # MKV remuxed to MP4
assert w.play_btn.isVisible() and w.burn_btn.isVisible()
print("session OK:", {k: v[k] for k in ("w", "h", "duration")}, "| virtual cam frames:", frames_seen, "| srt:", os.path.basename(w.last_srt))

# 4) captioned copy (burned in) really shows the captions
w.burn_copy(); wait(lambda: w.worker is None and w.last_output.endswith("(captioned).mp4"), 90)
def frame(p, t):
    raw = subprocess.run([ff, "-v", "error", "-ss", str(t), "-i", p, "-frames:v", "1", "-vf", "crop=640:100:0:260,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).astype(int)
orig = [p for p in os.listdir(os.path.join(T, "rec")) if p.endswith(".mp4") and "captioned" not in p][0]
a, b = frame(os.path.join(T, "rec", orig), 1.5), frame(w.last_output, 1.5)
assert (np.abs(a - b) > 40).sum() > 300, "captions not burned in"

# 5) live-caption fallback: if the recogniser fails, the session continues and a notice is shown
ui.MainWindow.hooks = dict(ui.MainWindow.hooks, transcribe=lambda pcm: (_ for _ in ()).throw(RuntimeError("model crashed")))
w.stream_cb.setChecked(False); w.vcam_cb.setChecked(False)
w.go_live(); wait(lambda: w.live and w.live["pipe"].frames > 10, 30)
wait(lambda: w.live["captioner"].degraded, 40)
wait(lambda: "recording when you stop" in w.cap_state.text(), 5)
assert w.live is not None                                              # still live
w.stop_live(); wait(lambda: w.worker is None, 90); w.close()
print("ui tests OK")
