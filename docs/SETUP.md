# Setup guide

## Zoom / Google Meet / Microsoft Teams / Webex (virtual camera)
1. **One time:** install OBS Studio (free, obsproject.com), open it, click **Start Virtual Camera**, close OBS.
2. Open Ideawood Live Studio, choose your camera and a background, tick **Virtual camera**, press **Go live**.
3. In Zoom/Meet/Teams > video settings choose **OBS Virtual Camera**. Keep Ideawood Live Studio running (live).
Your microphone is used directly by the meeting app; live captions here are for your own stream/recording.
If Zoom/Teams show a black picture: close and reopen the meeting app after pressing Go live.

## YouTube Live
YouTube Studio > Go live > Stream: copy the **Stream key**. In the app: tick *Live stream*, platform **YouTube Live**, paste the key, Go live.
(Stream latency, "normal/low/ultra-low" and DVR are set in YouTube Studio.)

## Facebook Live / Twitch
Facebook: Live Producer > Streaming software > copy the key (Server URL is preset). Twitch: dashboard > Settings > Stream > Primary stream key.

## TikTok LIVE
TikTok only gives an RTMP **server URL + stream key** to accounts that are eligible (via TikTok LIVE Studio / the web live page; requirements
such as follower count apply and change). Choose *TikTok LIVE (paste the server URL)*, paste the server URL and the key. Keys are single-use.

## Any other platform
Choose *Other RTMP server* and paste its `rtmp://` or `rtmps://` URL and key. Bitrate: 3500 kbps suits 720p30.

## Captions
Live captions need a speech model (built in: English Base). Tiny is faster on slow PCs (download it in the app). If the PC cannot keep up,
live captions switch off automatically (notice shown) and the **full captions are created from the recording after you stop**
(`<recording>.srt`), with an optional burned-in copy. Whisper does not support Luganda.

## Tips
Light your face from the front; avoid a background that matches your clothes; use a plain wall for the best cut-out. Start with *Fast* or
*Balanced*; the status line shows fps and the AI time per run. 720p/30 on a modern 4-core laptop is the target (unverified on real hardware).
