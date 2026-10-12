# Ideawood Live Studio

Go live with a **clean background and auto-captions** - for Zoom, Google Meet, Microsoft Teams, YouTube, Facebook, Twitch,
TikTok LIVE and any RTMP server. Windows desktop app; the AI runs **on your PC**.

* **Live background removal** (Robust Video Matting, CPU): blur your room, solid colour, or any picture. Adaptive: if the PC is busy
  the AI runs on every 2nd/3rd frame so the video stays smooth.
* **Virtual camera** - Zoom, Meet, Teams, Webex, WhatsApp Web... pick "OBS Virtual Camera" as the camera (one-time OBS install).
* **Live streaming** to YouTube / Facebook / Twitch / TikTok / custom RTMP, plus a **local recording** at the same time.
* **Live captions** shown on the video a moment after you speak (offline speech recognition, whisper.cpp). A **sync monitor** switches
  them off if the PC cannot keep up, and **full captions (SRT) are created from the recording** after you stop - you can also burn them
  into a captioned copy.
* Everything needed is bundled: FFmpeg, the AI model, the speech engine and an English speech model.

See [docs/SETUP.md](docs/SETUP.md) (Zoom/Meet/Teams, YouTube, TikTok...), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [CHANGELOG.md](CHANGELOG.md).

## Run from source
```
pip install -r requirements.txt
# put ffmpeg.exe / ffprobe.exe in bin/, whisper-cli.exe + whisper-server.exe (+DLLs) in bin/whisper/, a ggml model in models/whisper/
python main.py
```
Tests: `python tests/test_live_pipeline.py`, `test_live_captions.py`, `test_ui.py`, `test_windows.py` (Windows only).
CI (Windows) builds `IdeawoodLiveStudio.exe` and publishes releases on `v*` tags.

## Honest status (v0.1.0)
Verified by automated tests: the AI background replacement on the *recorded and streamed* video, RTMP push (video + audio) and local
recording, live-caption phrase detection / delay monitor / fallback, the real whisper-server protocol, the whole window end to end with
stand-in camera/mic. **Not yet verified on real hardware:** a real webcam and microphone, the OBS virtual camera inside Zoom/Meet/Teams,
streaming to the real platforms, and performance on your PC (measured on a 1-core test machine: the AI takes ~50-80 ms per run, so
expect smooth video via frame-skipping there and much faster on a normal multi-core PC).
