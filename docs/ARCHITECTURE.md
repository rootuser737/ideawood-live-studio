# Architecture

```
camera (ffmpeg dshow) --> FrameSource (newest frame only) --> LiveMatter (RVM ONNX on a SMALL copy) --> composite --> CaptionOverlay
                                                                                                             |
mic (ffmpeg dshow, 48k mono) --> MicSource --+--> AudioBridge --(tcp localhost)--> Streamer(ffmpeg) <----------+  frames (stdin)
                                             |                                      +--> RTMP (flv) + local MKV (tee)
                                             +--> LiveCaptioner: Segmenter -> WhisperServer (whisper-server) -> caption text
                                                     sync monitor -> "degraded" -> full captions after the show (finalize.py)
                                    VirtualCamSink (pyvirtualcam / OBS Virtual Camera) <-- frames
```
* `matte.py` - model runs at 256-448 px working width; mask scaled up over the full-quality frame; adaptive skip (every 1/2/3 frames).
* `compose.py` - blur / colour / picture backdrops, fixed-point compositing. `caption_overlay.py` - Qt-rendered caption panel.
* `sources.py` - everything through ffmpeg (names match the Lecture recorder). `sinks.py` - tee needs `-flags +global_header`
  (AAC in both FLV and Matroska); newer ffmpeg rejects `-thread_queue_size` on inputs - do not use it.
* `live_captions.py` - energy VAD segments (<=5 s), resident whisper-server (multipart POST /inference), hallucination filter,
  lag/backlog monitor. `finalize.py` - MKV->MP4 remux, full-file ASR via `asr.py` (shared with Ideawood Studio), SRT + burn-in.
* `ui.py` - window + live controller; tests inject hardware stand-ins through `MainWindow.hooks`.

## Not done yet / ideas
Virtual microphone (so Zoom can hear "denoised" audio), multi-destination streaming in the UI (the Streamer already accepts several URLs),
scene/overlay graphics (lower thirds, logos), hotkeys, GPU acceleration (DirectML/CUDA ONNX providers), auto-framing, a proper installer.
