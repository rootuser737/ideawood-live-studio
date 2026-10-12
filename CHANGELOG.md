# Changelog

## 0.1.0 - first release
* Live AI background removal (blur / colour / picture) with adaptive speed.
* Virtual camera output (Zoom, Meet, Teams) via the OBS Virtual Camera driver.
* RTMP streaming (YouTube, Facebook, Twitch, TikTok, custom) with simultaneous local recording; MKV remuxed to MP4.
* Live captions on the video with a sync monitor; automatic fallback to full captions from the recording; optional burned-in copy.
* Bundled: FFmpeg, RVM AI model, whisper.cpp (cli + server) and the English Base speech model. Windows CI with real-ASR checks.
* Known gaps: not tested with a real webcam/mic or inside the meeting apps; no virtual microphone; single stream destination in the UI.
