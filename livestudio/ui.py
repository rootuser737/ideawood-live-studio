"""Ideawood Live Studio - main window."""
from __future__ import annotations

import datetime
import os
import sys
import tempfile
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
                               QVBoxLayout, QWidget)

from . import __version__, asr, matte
from .caption_overlay import CaptionOverlay
from .compose import MODES, Backdrop
from .finalize import make_final_captions, remux_to_mp4
from .live_captions import LiveCaptioner, WhisperServer, live_srt
from .mediautil import media_info, run_ffmpeg
from .paths import asset_path, find_tool
from .pipeline import LiveConfig, LivePipeline
from .sinks import DESTINATIONS, HELP_WINDOWS, Streamer, VirtualCamError, VirtualCamSink, rtmp_url
from .sources import FrameSource, IS_WIN, MicSource, camera_args, file_args, list_devices, mic_args

STYLE = """
QWidget { background:#14162b; color:#e8eaff; font-size:13px; }
QFrame#card { background:#1c1f3a; border:1px solid #2e3360; border-radius:10px; }
QLabel#h { font-weight:700; font-size:14px; color:#ffffff; } QLabel#muted { color:#9aa0d0; }
QPushButton { background:#2a2e52; border:1px solid #3a3f66; border-radius:7px; padding:6px 12px; }
QPushButton:hover { background:#3a3f70; } QPushButton:disabled { color:#6b7099; }
QPushButton#live { background:#d32f2f; border-color:#ef5350; font-weight:700; font-size:15px; padding:10px; }
QPushButton#live:disabled { background:#4a2a2f; }
QPushButton#stop { background:#2a2e52; font-weight:700; font-size:15px; padding:10px; }
QLineEdit, QComboBox { background:#10122a; border:1px solid #3a3f66; border-radius:6px; padding:5px; }
QProgressBar { background:#10122a; border:1px solid #3a3f66; border-radius:5px; height:10px; text-align:center; }
QProgressBar::chunk { background:#5b6cff; border-radius:4px; }
"""
RESOLUTIONS = {"1280 x 720 (HD)": (1280, 720), "854 x 480": (854, 480), "640 x 360 (lightest)": (640, 360)}
SOURCES = ["Camera", "Demo picture (try it without a camera)", "Video file (test)"]
STREAM_NOTE = ("Stream key: copy it from your platform's live page (YouTube Studio > Go live, Facebook Live Producer, "
               "Twitch dashboard, TikTok LIVE access). Treat it like a password.")


def card(title: str, sub: str = ""):
    f = QFrame(objectName="card")
    v = QVBoxLayout(f)
    v.setContentsMargins(14, 12, 14, 12)
    t = QLabel(title, objectName="h")
    v.addWidget(t)
    if sub:
        s = QLabel(sub, objectName="muted")
        s.setWordWrap(True)
        v.addWidget(s)
    return f, v


def row(label: str, w: QWidget) -> QWidget:
    r = QWidget()
    h = QHBoxLayout(r)
    h.setContentsMargins(0, 0, 0, 0)
    l = QLabel(label)
    l.setMinimumWidth(110)
    h.addWidget(l)
    h.addWidget(w, 1)
    return r


class _Finish(QThread):
    status = Signal(str)
    progress = Signal(float)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    # tests replace these to run without a camera/microphone/driver
    hooks: dict = {}

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Ideawood Live Studio {__version__}")
        ico = asset_path("icon.ico")
        if os.path.isfile(ico):
            self.setWindowIcon(QIcon(ico))
        self.resize(1240, 780)
        self.qs = QSettings("Ideawood", "LiveStudio")
        self.devices = {"video": [], "audio": []}
        self.live: Optional[dict] = None
        self.video_path = ""
        self.picture_path = ""
        self.color = self.qs.value("color", "#00b140")
        self.level = 0
        self.worker: Optional[_Finish] = None
        self.last_output = ""
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        QTimer.singleShot(300, self.scan)

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        left = QVBoxLayout()
        h.addLayout(left, 3)
        self.preview = QLabel("Your live picture appears here")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("background:#0b0c1c;border:1px solid #2e3360;border-radius:10px;")
        left.addWidget(self.preview, 1)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        self.status.setObjectName("muted")
        left.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.hide()
        left.addWidget(self.bar)
        b = QHBoxLayout()
        self.go = QPushButton("Go live", objectName="live")
        self.go.clicked.connect(self.go_live)
        self.stop_btn = QPushButton("Stop", objectName="stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_live)
        self.mute = QPushButton("Mute mic")
        self.mute.setCheckable(True)
        self.mute.toggled.connect(self._mute)
        b.addWidget(self.go, 2)
        b.addWidget(self.stop_btn, 1)
        b.addWidget(self.mute)
        left.addLayout(b)
        res = QHBoxLayout()
        self.play_btn = QPushButton("Play recording")
        self.folder_btn = QPushButton("Open folder")
        self.burn_btn = QPushButton("Make a captioned copy (burned in)")
        for x in (self.play_btn, self.folder_btn, self.burn_btn):
            x.hide()
            res.addWidget(x)
        self.play_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output)))
        self.folder_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(self.last_output))))
        self.burn_btn.clicked.connect(self.burn_copy)
        left.addLayout(res)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        panel = QWidget()
        pv = QVBoxLayout(panel)
        scroll.setWidget(panel)
        h.addWidget(scroll, 2)

        # --- camera
        f, v = card("1. Camera")
        self.src = QComboBox(); self.src.addItems(SOURCES)
        self.cam = QComboBox()
        self.res = QComboBox(); self.res.addItems(list(RESOLUTIONS))
        self.fps = QComboBox(); self.fps.addItems(["30", "24", "15"])
        self.mirror = QCheckBox("Mirror the picture")
        rb = QPushButton("Refresh devices"); rb.clicked.connect(self.scan)
        self.src.currentIndexChanged.connect(self._src_changed)
        for lab, w in (("Source", self.src), ("Camera", self.cam), ("Size", self.res), ("Smoothness", self.fps)):
            v.addWidget(row(lab, w))
        v.addWidget(self.mirror); v.addWidget(rb)
        pv.addWidget(f)

        # --- background
        f, v = card("2. Background", "The AI cuts you out of the picture live.")
        self.mode = QComboBox(); self.mode.addItems(MODES)
        self.mode.setCurrentText(self.qs.value("mode", MODES[1]))
        self.color_btn = QPushButton("Choose colour..."); self.color_btn.clicked.connect(self.pick_color)
        self.pic_btn = QPushButton("Choose picture..."); self.pic_btn.clicked.connect(self.pick_picture)
        self.blur = QSlider(Qt.Horizontal); self.blur.setRange(5, 61); self.blur.setValue(25)
        self.quality = QComboBox(); self.quality.addItems(["Fast", "Balanced", "Best"]); self.quality.setCurrentText("Balanced")
        self.edge = QSlider(Qt.Horizontal); self.edge.setRange(0, 100); self.edge.setValue(25)
        self.mode.currentIndexChanged.connect(self._apply_backdrop)
        self.blur.valueChanged.connect(self._apply_backdrop)
        self.quality.currentIndexChanged.connect(self._apply_cfg)
        self.edge.valueChanged.connect(self._apply_cfg)
        self.mirror.toggled.connect(self._apply_cfg)
        for lab, w in (("Replace with", self.mode), ("Blur strength", self.blur), ("Speed / quality", self.quality),
                       ("Edge sharpness", self.edge)):
            v.addWidget(row(lab, w))
        v.addWidget(self.color_btn); v.addWidget(self.pic_btn)
        self.perf = QLabel("", objectName="muted"); self.perf.setWordWrap(True)
        v.addWidget(self.perf)
        pv.addWidget(f)

        # --- microphone
        f, v = card("3. Microphone")
        self.mic = QComboBox()
        self.meter = QProgressBar(); self.meter.setRange(0, 100); self.meter.setTextVisible(False)
        v.addWidget(row("Microphone", self.mic)); v.addWidget(self.meter)
        pv.addWidget(f)

        # --- outputs
        f, v = card("4. Where should it go?")
        self.vcam_cb = QCheckBox("Virtual camera - for Zoom, Google Meet, Microsoft Teams, Webex, WhatsApp Web...")
        self.vcam_cb.setChecked(False)
        vh = QPushButton("How to set up the virtual camera"); vh.clicked.connect(self.vcam_help)
        self.stream_cb = QCheckBox("Live stream - YouTube, Facebook, Twitch, TikTok or any RTMP server")
        self.dest = QComboBox(); self.dest.addItems(list(DESTINATIONS)); self.dest.currentIndexChanged.connect(self._dest_changed)
        self.server = QLineEdit(); self.server.setPlaceholderText("rtmp://...")
        self.key = QLineEdit(); self.key.setEchoMode(QLineEdit.Password); self.key.setPlaceholderText("stream key")
        self.kbps = QComboBox(); self.kbps.addItems(["2500", "3500", "4500", "6000"]); self.kbps.setCurrentText("3500")
        note = QLabel(STREAM_NOTE, objectName="muted"); note.setWordWrap(True)
        self.rec_cb = QCheckBox("Record to a file on this PC (needed for full captions afterwards)"); self.rec_cb.setChecked(True)
        vid = os.path.join(os.path.expanduser("~"), "Videos")
        self.folder = QLineEdit(self.qs.value("folder", vid if os.path.isdir(vid) else os.path.expanduser("~")))
        fb = QPushButton("Browse..."); fb.clicked.connect(self.browse)
        v.addWidget(self.vcam_cb); v.addWidget(vh); v.addWidget(self.stream_cb)
        for lab, w in (("Platform", self.dest), ("Server URL", self.server), ("Stream key", self.key), ("Video bitrate (kbps)", self.kbps)):
            v.addWidget(row(lab, w))
        v.addWidget(note); v.addWidget(self.rec_cb); v.addWidget(row("Folder", self.folder)); v.addWidget(fb)
        pv.addWidget(f)
        self._dest_changed()

        # --- captions
        f, v = card("5. Captions", "Live captions appear a moment after you speak. If the PC cannot keep up they switch "
                    "off by themselves and the full captions are made from the recording after you stop.")
        self.live_cap = QCheckBox("Show live captions on the video"); self.live_cap.setChecked(True)
        self.final_cap = QCheckBox("Create full captions (SRT) from the recording after I stop"); self.final_cap.setChecked(True)
        self.lang = QComboBox(); self.lang.addItems(list(asr.LANGUAGES)); self.lang.setCurrentText("English")
        self.model = QComboBox()
        self.model_note = QLabel("", objectName="muted"); self.model_note.setWordWrap(True)
        dl = QPushButton("Download a speech model"); dl.clicked.connect(self.download_model)
        self.cap_state = QLabel("", objectName="muted"); self.cap_state.setWordWrap(True)
        for lab, w in (("Language", self.lang), ("Speech model", self.model)):
            v.addWidget(row(lab, w))
        v.addWidget(self.model_note); v.addWidget(dl); v.addWidget(self.live_cap); v.addWidget(self.final_cap); v.addWidget(self.cap_state)
        pv.addWidget(f)
        pv.addStretch(1)
        self._fill_models()
        self._src_changed()

    # ------------------------------------------------------------------ small handlers
    def scan(self):
        self.devices = list_devices()
        for cb, kind in ((self.cam, "video"), (self.mic, "audio")):
            cur = cb.currentText()
            cb.clear()
            cb.addItems(self.devices[kind] or (["(no device found)"] if IS_WIN else ["(Windows only)"]))
            if cur:
                cb.setCurrentText(cur)

    def _src_changed(self, *_):
        self.cam.setEnabled(self.src.currentIndex() == 0)

    def _dest_changed(self, *_):
        d = self.dest.currentText()
        preset = DESTINATIONS.get(d, "")
        self.server.setText(preset)
        self.server.setEnabled(not preset)

    def pick_color(self):
        c = QColorDialog.getColor(QColor(self.color), self, "Background colour")
        if c.isValid():
            self.color = c.name()
            self.qs.setValue("color", self.color)
            self.mode.setCurrentText("Solid colour")
            self._apply_backdrop()

    def pick_picture(self):
        p, _ = QFileDialog.getOpenFileName(self, "Background picture", "", "Pictures (*.png *.jpg *.jpeg *.bmp *.webp)")
        if p:
            self.picture_path = p
            self.mode.setCurrentText("Picture")
            self._apply_backdrop()

    def browse(self):
        d = QFileDialog.getExistingDirectory(self, "Save recordings in", self.folder.text())
        if d:
            self.folder.setText(d)

    def vcam_help(self):
        QMessageBox.information(self, "Virtual camera", HELP_WINDOWS + "\n\nThen in Zoom / Meet / Teams open the video "
                                "settings and choose 'OBS Virtual Camera' as your camera. Ideawood Live Studio must be "
                                "live (Go live) for it to show your picture.")

    def _mute(self, on):
        self.mute.setText("Unmute mic" if on else "Mute mic")
        if self.live:
            self.live["muted"] = on
            if self.live.get("streamer"):
                self.live["streamer"].set_muted(on)

    def _apply_backdrop(self, *_):
        self.qs.setValue("mode", self.mode.currentText())
        if self.live:
            bd = self.live["backdrop"]
            bd.mode, bd.color, bd.picture, bd.blur = self.mode.currentText(), self.color, self.picture_path or None, self.blur.value()
            bd._key = None

    def _apply_cfg(self, *_):
        if self.live:
            cfg = self.live["pipe"].cfg
            cfg.quality, cfg.edge, cfg.mirror = self.quality.currentText(), self.edge.value() / 100.0, self.mirror.isChecked()

    # ------------------------------------------------------------------ speech models
    def _fill_models(self):
        cur = self.model.currentData()
        self.model.clear()
        inst = asr.installed_models()
        for key, (label, _f, _mb, multi) in asr.MODELS.items():
            tag = ("  [built in]" if asr.is_bundled(key) else "  [ready]") if key in inst else "  [download]"
            self.model.addItem(label + tag, key)
        pick = cur or ("base.en" if "base.en" in inst else (inst[0] if inst else "base.en"))
        self.model.setCurrentIndex(max(0, self.model.findData(pick)))
        self.model_note.setText("Tip: Tiny is fastest for live captions on slower PCs; Base/Small are more accurate for the full captions."
                                if inst else "No speech model yet - download one to use captions (works offline afterwards).")

    def download_model(self):
        key = self.model.currentData()
        if self.worker:
            return
        self._busy(f"Downloading the speech model ({asr.MODELS[key][2]} MB)...")
        self.worker = _Finish(lambda w: asr.download_model(key, w.progress.emit))
        self.worker.progress.connect(lambda p: self.bar.setValue(int(p)))
        self.worker.done.connect(lambda _p: (self._idle("Speech model downloaded."), self._fill_models()))
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _busy(self, msg):
        self.status.setText(msg)
        self.bar.setValue(0)
        self.bar.show()

    def _idle(self, msg):
        self.worker = None
        self.status.setText(msg)
        self.bar.hide()

    def _failed(self, msg):
        self._idle("Stopped: " + msg)
        QMessageBox.warning(self, "Ideawood Live Studio", msg)

    # ------------------------------------------------------------------ going live
    def _source(self, w, h, fps):
        hk = self.hooks.get("source_args")
        if hk:
            return FrameSource(hk, w, h, fps)
        i = self.src.currentIndex()
        if i == 0:
            if not IS_WIN:
                raise RuntimeError("Camera capture works on Windows. Use 'Demo picture' to try the app here.")
            name = self.cam.currentText()
            if name.startswith("("):
                raise RuntimeError("No camera found. Plug one in and click 'Refresh devices'.")
            return FrameSource(camera_args(name, fps), w, h, fps)
        if i == 1:
            p = asset_path("demo_person.png")
            return FrameSource(["-re", "-loop", "1", "-framerate", str(fps), "-i", p], w, h, fps)
        if not self.video_path:
            p, _ = QFileDialog.getOpenFileName(self, "Choose a video to use as the camera", "", "Video (*.mp4 *.mov *.mkv *.avi *.webm)")
            if not p:
                raise RuntimeError("No video chosen.")
            self.video_path = p
        return FrameSource(file_args(self.video_path), w, h, fps)

    def go_live(self):
        if self.live or self.worker:
            return
        w, h = RESOLUTIONS[self.res.currentText()]
        fps = int(self.fps.currentText())
        mode = self.mode.currentText()
        if mode != MODES[0]:
            ok, why = matte.available()
            if not ok:
                QMessageBox.warning(self, "Ideawood Live Studio", why)
                return
        if self.stream_cb.isChecked():
            try:
                url = rtmp_url(self.server.text(), self.key.text())
            except ValueError as e:
                QMessageBox.information(self, "Ideawood Live Studio", str(e))
                return
            if not self.key.text().strip():
                QMessageBox.information(self, "Ideawood Live Studio", "Paste the stream key from your platform first.")
                return
        else:
            url = ""
        live: dict = {"muted": self.mute.isChecked(), "w": w, "h": h, "fps": fps, "t0": time.time()}
        sinks = []
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            live["source"] = self._source(w, h, fps)
            # outputs
            rec_path = None
            if self.rec_cb.isChecked():
                os.makedirs(self.folder.text(), exist_ok=True)
                rec_path = os.path.join(self.folder.text(), "Live " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M") + ".mkv")
            if url or rec_path:
                st = Streamer(w, h, fps, [url] if url else [], rec_path, int(self.kbps.currentText()))
                st.start()
                live["streamer"], live["rec_path"] = st, rec_path
                sinks.append(st.write_frame)
            if self.vcam_cb.isChecked():
                try:
                    vc = VirtualCamSink(w, h, fps, factory=self.hooks.get("vcam_factory"))
                    vc.start()
                    live["vcam"] = vc
                    sinks.append(vc.send)
                except VirtualCamError as e:
                    self.vcam_cb.setChecked(False)
                    QMessageBox.warning(self, "Virtual camera", str(e) + "\n\nGoing live without the virtual camera.")
            # captions
            live["overlay"] = CaptionOverlay(w, h) if self.live_cap.isChecked() else None
            self._start_captions(live)
            # microphone
            self._start_mic(live)
            live["backdrop"] = Backdrop(mode, self.color, self.picture_path or None, self.blur.value())
            cfg = LiveConfig(fps=fps, quality=self.quality.currentText(), edge=self.edge.value() / 100.0, mirror=self.mirror.isChecked())
            live["pipe"] = LivePipeline(live["source"], live["backdrop"], cfg, sinks, live["overlay"])
            live["source"].start()
            live["pipe"].start()
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            self.live = live
            self._teardown(live)
            self.live = None
            QMessageBox.warning(self, "Ideawood Live Studio", f"Could not go live:\n{e}")
            return
        QApplication.restoreOverrideCursor()
        self.qs.setValue("folder", self.folder.text())
        self.live = live
        self.go.setEnabled(False)
        self.stop_btn.setEnabled(True)
        for b in (self.play_btn, self.folder_btn, self.burn_btn):
            b.hide()
        outs = [n for n, k in (("virtual camera", "vcam"), ("stream", "streamer")) if live.get(k)]
        self.status.setText("LIVE" + (" - " + ", ".join(outs) if outs else " (preview only)"))

    def _start_captions(self, live):
        live["captioner"] = live["server"] = None
        if not self.live_cap.isChecked():
            return
        key = self.model.currentData()
        mp = asr.find_model(key)
        if not mp:
            self.cap_state.setText("Live captions off: no speech model yet (use 'Download a speech model').")
            return
        lang = asr.LANGUAGES[self.lang.currentText()]
        if not asr.MODELS[key][3] and lang not in ("en", "auto"):
            lang = "en"
        tr = self.hooks.get("transcribe")
        if tr is None:
            try:
                exe = os.path.join(os.path.dirname(asr.find_whisper_cli()), "whisper-server" + (".exe" if os.name == "nt" else ""))
                if os.path.isfile(exe):
                    srv = WhisperServer(exe, mp, lang if asr.MODELS[key][3] else "en", threads=2)
                    srv.start()
                    live["server"] = srv
                    tr = srv.transcribe
                else:
                    eng = asr.WhisperEngine(mp, threads=2)
                    tr = lambda pcm: _cli_transcribe(eng, pcm, lang)  # noqa: E731
            except Exception as e:  # noqa: BLE001
                self.cap_state.setText(f"Live captions off ({e}). Full captions will be made after you stop.")
                return
        ov = live["overlay"]

        def on_caption(text, lag):
            if ov is not None:
                ov.set_text(text)
            live["last_caption"] = text
        live["captioner"] = LiveCaptioner(tr, on_caption, lambda s, m: live.__setitem__("cap_msg", m))
        self.cap_state.setText("Live captions on.")

    def _start_mic(self, live):
        args = self.hooks.get("mic_args")
        if not args:
            name = self.mic.currentText()
            if not IS_WIN or name.startswith("("):
                live["mic"] = None
                return
            args = mic_args(name)
        cap, st = live.get("captioner"), live.get("streamer")

        def on_pcm(pcm: bytes):
            self.level = int(min(100, (np.abs(np.frombuffer(pcm, np.int16)).max() / 32768.0) * 160))
            if live["muted"]:
                return
            if st:
                st.write_audio(pcm)
            if cap:
                cap.feed(pcm)
        live["mic"] = MicSource(args, on_pcm)
        live["mic"].start()

    # ------------------------------------------------------------------ live status
    def _tick(self):
        lv = self.live
        if not lv:
            self.meter.setValue(0)
            return
        frame = lv["pipe"].preview()
        if frame is not None:
            h, w = frame.shape[:2]
            img = QImage(frame.data, w, h, w * 3, QImage.Format_RGB888)
            self.preview.setPixmap(QPixmap.fromImage(img).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.meter.setValue(0 if lv["muted"] else self.level)
        s = lv["pipe"].stats()
        secs = int(time.time() - lv["t0"])
        extra = f"  |  {s['fps']} fps"
        if s["matte_ms"]:
            extra += f", AI {s['matte_ms']:.0f} ms" + (f" (every {s['matte_skip']} frames)" if s["matte_skip"] > 1 else "")
        st = lv.get("streamer")
        if st and st.dropped:
            extra += f", {st.dropped} frames dropped (slow network/PC)"
        self.perf.setText(f"{s['fps']} fps - AI {s['matte_ms']:.0f} ms per run" + (" - PC is busy: AI runs every "
                          f"{s['matte_skip']} frames to keep the video smooth" if s["matte_skip"] > 1 else ""))
        self.status.setText(f"LIVE {secs // 60:02d}:{secs % 60:02d}{extra}")
        cap = lv.get("captioner")
        if cap and lv.get("cap_msg"):
            self.cap_state.setText(lv.pop("cap_msg"))
        err = s["error"] or (st.error() if st else "") or (lv["source"].error if lv["source"].error else "")
        if err:
            self.stop_live(error=err)

    # ------------------------------------------------------------------ stopping + after-the-show work
    def _teardown(self, lv):
        for k in ("pipe",):
            if lv.get(k):
                lv[k].stop()
        if lv.get("source"):
            lv["source"].stop()
        if lv.get("mic"):
            lv["mic"].stop()
        tail = None
        if lv.get("captioner"):
            lv["captioner"].stop()
        if lv.get("server"):
            lv["server"].stop()
        if lv.get("vcam"):
            lv["vcam"].stop()
        if lv.get("streamer"):
            lv["streamer"].stop()

    def stop_live(self, error: str = ""):
        lv = self.live
        if not lv:
            return
        self.live = None
        self._teardown(lv)
        self.go.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.preview.setText("Your live picture appears here")
        rec = lv.get("rec_path")
        msg = "Stopped." + (f"  ({error})" if error else "")
        if not rec or not os.path.exists(rec):
            self.status.setText(msg)
            return
        cap = lv.get("captioner")
        live_caps = list(cap.captions) if cap else []
        degraded = bool(cap and cap.degraded)
        want_final = self.final_cap.isChecked()
        model_key = self.model.currentData()
        mp = asr.find_model(model_key)
        lang = asr.LANGUAGES[self.lang.currentText()]
        if not asr.MODELS[model_key][3] and lang not in ("en", "auto"):
            lang = "en"
        engine_hook = self.hooks.get("final_engine")

        def job(w):
            w.status.emit("Saving the recording...")
            mp4 = remux_to_mp4(rec)
            out = {"video": mp4, "srt": "", "note": ""}
            if want_final and (mp or engine_hook):
                w.status.emit("Creating full captions from the recording (runs on your PC)...")
                res = make_final_captions(mp4, mp or "", lang, "", w.status.emit, w.progress.emit, engine=engine_hook)
                out["srt"] = res["srt"]
                out["note"] = "Full captions created." + (" (Live captions had fallen behind, so these replace them.)" if degraded else "")
            elif live_caps:
                p = os.path.splitext(mp4)[0] + ".live.srt"
                with open(p, "w", encoding="utf-8-sig") as f:
                    f.write(live_srt(live_caps))
                out["srt"], out["note"] = p, "Saved the live captions (no speech model for full captions)."
            return out
        self.last_output = rec
        self._busy(msg + "  Finishing the recording...")
        self.worker = _Finish(job)
        self.worker.status.connect(self.status.setText)
        self.worker.progress.connect(lambda p: self.bar.setValue(int(p)))
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _finished(self, out):
        self.worker = None
        self.bar.hide()
        self.last_output = out["video"]
        self.last_srt = out.get("srt", "")
        self.status.setText(f"Saved: {out['video']}" + (f"\nCaptions: {out['srt']}  -  {out['note']}" if out.get("srt") else ""))
        for b in (self.play_btn, self.folder_btn):
            b.show()
        self.burn_btn.setVisible(bool(self.last_srt))

    def burn_copy(self):
        if self.worker or not getattr(self, "last_srt", ""):
            return
        video, srt = self.last_output, self.last_srt
        out = os.path.splitext(video)[0] + " (captioned).mp4"
        info = media_info(video)
        self._busy("Adding captions to a copy of the video...")

        def job(w):
            with open(srt, "r", encoding="utf-8-sig") as f:
                cues = asr.parse_srt(f.read())
            work = tempfile.mkdtemp(prefix="ideawood_live_")
            try:
                with open(os.path.join(work, "subs.ass"), "w", encoding="utf-8") as f:
                    f.write(asr.to_ass(cues, info["w"], info["h"], "Bottom", "Medium", "White", True, False))
                run_ffmpeg(asr.burn_cmd(video, "subs.ass", out), info["duration"], w.progress.emit, cwd=work)
            finally:
                import shutil
                shutil.rmtree(work, ignore_errors=True)
            return {"video": out, "srt": srt, "note": "Captioned copy created."}
        self.worker = _Finish(job)
        self.worker.progress.connect(lambda p: self.bar.setValue(int(p)))
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def closeEvent(self, e):
        if self.live:
            self.stop_live()
        if self.worker:
            self.worker.wait(60000)
        super().closeEvent(e)


def _cli_transcribe(engine: "asr.WhisperEngine", pcm: np.ndarray, lang: str) -> str:
    """Fallback when whisper-server is missing: one whisper-cli run per phrase (slower)."""
    from .live_captions import wav_bytes
    d = tempfile.mkdtemp(prefix="ideawood_live_")
    try:
        p = os.path.join(d, "p.wav")
        with open(p, "wb") as f:
            f.write(wav_bytes(pcm))
        return engine.transcribe(p, lang).text()
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
