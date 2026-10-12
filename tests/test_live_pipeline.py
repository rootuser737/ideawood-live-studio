"""End to end: picture-as-camera -> AI background removal -> caption overlay -> RTMP stream + local recording.
Then the RECORDED video is inspected: corners must be the new background, the person must still be there."""
import os, subprocess, sys, tempfile, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PySide6.QtWidgets import QApplication
from livestudio.caption_overlay import CaptionOverlay
from livestudio.compose import Backdrop
from livestudio.finalize import remux_to_mp4
from livestudio.mediautil import media_info
from livestudio.paths import find_tool
from livestudio.pipeline import LiveConfig, LivePipeline
from livestudio.sinks import Streamer, rtmp_url
from livestudio.sources import FrameSource, MicSource, wav_args

app = QApplication.instance() or QApplication([])
HERE = os.path.dirname(os.path.abspath(__file__)); T = tempfile.mkdtemp(); ff = find_tool("ffmpeg")
W, H, FPS = 640, 360, 15
recv = os.path.join(T, "received.flv")
srv = subprocess.Popen([ff, "-y", "-v", "error", "-listen", "1", "-f", "flv", "-i", "rtmp://127.0.0.1:19351/live/key", "-c", "copy", recv],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.0)
rec = os.path.join(T, "live.mkv")
streamer = Streamer(W, H, FPS, [rtmp_url("rtmp://127.0.0.1:19351/live", "key")], rec, 1200); streamer.start()
mic = MicSource(wav_args(os.path.join(HERE, "data", "jfk.wav")), streamer.write_audio); mic.start()
src = FrameSource(["-re", "-loop", "1", "-framerate", str(FPS), "-i", os.path.join(HERE, "data", "person.png")], W, H, FPS); src.start()
overlay = CaptionOverlay(W, H)
pipe = LivePipeline(src, Backdrop("Solid colour", color="#00b140"), LiveConfig(fps=FPS, quality="Fast"), [streamer.write_frame], overlay)
pipe.start()
time.sleep(2.0); overlay.set_text("Hello from Ideawood Live Studio", hold=6)
t0 = time.time()
while time.time() - t0 < 5.0:
    app.processEvents(); time.sleep(0.05)
st = pipe.stats(); pipe.stop(); src.stop(); mic.stop(); streamer.stop(); srv.wait(timeout=20)
print("pipeline stats:", st, "| streamer dropped:", streamer.dropped, "| error:", repr(pipe.error), repr(streamer.error()))
assert not pipe.error and st["frames"] > 20 and st["matte_ms"] > 0

mp4 = remux_to_mp4(rec)
for label, p in (("recording", mp4), ("received over RTMP", recv)):
    i = media_info(p)
    assert i["has_video"] and i["has_audio"] and (i["w"], i["h"]) == (W, H) and i["duration"] > 4.0, (label, i)
    print(label, i)

def frame(p, t):
    raw = subprocess.run([ff, "-v", "error", "-ss", str(t), "-i", p, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(int)
f = frame(recv, 3.5)                                    # the stream a viewer would see (after captions appeared)
green = np.array([0, 177, 64])
corners = [f[5:40, 5:40], f[5:40, W - 40:W - 5]]
for c in corners:
    d = np.abs(c.reshape(-1, 3).mean(axis=0) - green).max()
    assert d < 40, f"corner is not the new background: {c.reshape(-1,3).mean(axis=0)}"
mid = f[120:260, 220:420].reshape(-1, 3)
frac_green = float((np.abs(mid - green).max(axis=1) < 45).mean())
assert frac_green < 0.6, f"the person vanished (green fraction in the centre = {frac_green:.2f})"
cap = f[int(H * 0.8):int(H * 0.95)].reshape(-1, 3)
white = float((cap.min(axis=1) > 215).mean())
assert white > 0.01, f"caption text not visible on the stream (white fraction {white:.3f})"
print(f"background replaced: corners ~green, centre green fraction {frac_green:.2f}, caption white pixels {white:.3f}")
print("live pipeline tests OK")
