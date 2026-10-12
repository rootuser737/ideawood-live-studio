"""Windows-only checks (run in CI on a real Windows runner with the bundled FFmpeg): device discovery, the camera and
microphone command lines are accepted by FFmpeg, and the virtual camera fails gracefully when OBS is not installed."""
import os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform != "win32":
    print("windows tests skipped (not Windows)"); sys.exit(0)
from livestudio import sources
from livestudio.paths import find_tool
from livestudio.sinks import VirtualCamError, VirtualCamSink

PARSE = re.compile(r"cannot be applied|Unrecognized option|Error parsing (global )?options|Option .* not found", re.I)
ff = find_tool("ffmpeg")
print(subprocess.run([ff, "-version"], capture_output=True, text=True).stdout.splitlines()[0])
dev = sources.list_devices(); assert set(dev) == {"video", "audio"}; print("devices:", dev)

def run(args):
    return subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", *args, "-t", "1", "-f", "null", "-"],
                          capture_output=True, text=True, errors="replace", timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
for label, a in (("camera", sources.camera_args("__no_such_camera__", 30)), ("camera sized", sources.camera_args("__no_such_camera__", 30, "1280x720")),
                 ("microphone", sources.mic_args("__no_such_mic__"))):
    r = run(a)
    assert not PARSE.search(r.stderr), f"{label}: FFmpeg rejected the command line:\n{r.stderr[-500:]}"
    assert r.returncode != 0                                             # no such device: fails on the DEVICE, not on syntax
    print(label, "command line accepted (device missing, as expected)")

try:
    v = VirtualCamSink(640, 360, 30); v.start(); print("virtual camera AVAILABLE:", v.device); v.stop()
except VirtualCamError as e:
    assert "OBS" in str(e); print("virtual camera not installed on this machine: friendly message OK")
print("windows tests OK")
