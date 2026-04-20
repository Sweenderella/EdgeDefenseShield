#!/usr/bin/env python3
"""
EdgeShield — Laptop Monitor (V2 — DNN + SNN Side by Side)
==========================================================
Receives detections from Raspberry Pi V2 and displays
DNN and SNN results side by side on a live dashboard.
Authors: Sweety Ramnani, Zarin Shejuti
Department of Computer Science, University of South Carolina

Usage:
    python3 laptop_monitor.py

Requirements:
    pip install opencv-python pillow matplotlib
"""

import socket
import json
import cv2
import numpy as np
import threading
from datetime import datetime
from collections import deque
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import logging


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    HOST         = '0.0.0.0'
    PORT         = 5555
    WINDOW_WIDTH  = 1600
    WINDOW_HEIGHT = 950
    VIDEO_WIDTH   = 580
    VIDEO_HEIGHT  = 380
    MAX_DETECTIONS = 200
    ALERT_CLASSES  = ['gun_shot', 'siren', 'car_horn']
    CLASSES = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
        'siren', 'street_music'
    ]
    CLASS_DISPLAY = {
        'air_conditioner': 'Air Conditioner', 'car_horn': 'Car Horn',
        'children_playing': 'Children Playing', 'dog_bark': 'Dog Bark',
        'drilling': 'Drilling', 'engine_idling': 'Engine Idling',
        'gun_shot': 'GUN SHOT', 'jackhammer': 'Jackhammer',
        'siren': 'Siren', 'street_music': 'Street Music',
    }
    BG       = '#1A1A2E'
    PANEL_BG = '#16213E'
    DNN_COLOR  = '#4CC9F0'
    SNN_COLOR  = '#F72585'
    THREAT_COLOR = '#FF4444'
    SAFE_COLOR   = '#44FF88'
    TEXT_COLOR   = '#FFFFFF'
    MUTED_COLOR  = '#AAAAAA'


# ============================================================================
# DASHBOARD
# ============================================================================

class MonitoringDashboard:

    def __init__(self, config):
        self.config  = config
        self.detections = deque(maxlen=config.MAX_DETECTIONS)
        self.current_frame = None
        self.running = True

        self.stats = {
            'total': 0, 'threats': 0,
            'dnn': {'avg_latency': 0, 'threats': 0,
                    'per_class': {c: 0 for c in config.CLASSES}},
            'snn': {'avg_latency': 0, 'threats': 0,
                    'per_class': {c: 0 for c in config.CLASSES}},
            'agreement': {'total': 0, 'agreed': 0}
        }

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self._build_gui()

        t = threading.Thread(target=self._listen, daemon=True)
        t.start()
        self.logger.info(f"Listening on port {config.PORT}")

    # -------------------------------------------------------------------------
    # GUI
    # -------------------------------------------------------------------------

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("EdgeDefenseShield — Realtime Threat Detection Monitor Tool")
        self.root.geometry(f"{self.config.WINDOW_WIDTH}x{self.config.WINDOW_HEIGHT}")
        self.root.minsize(1200,700)
        self.root.maxsize(self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT)
        self.root.resizable(True, True)
        self.root.configure(bg=self.config.BG)

        # ── Title bar ────────────────────────────────────────────────────────
        title = tk.Label(self.root, text="EdgeDefenseShield — Real-Time Threat Detection",
                         font=("Arial", 20, "bold"),
                         bg=self.config.BG, fg=self.config.TEXT_COLOR)
        title.pack(pady=(10, 0))

        subtitle = tk.Label(self.root, text="Deep Neural Network vs  Spiking Neural Network Multimodal Threat Detector",
                            font=("Arial", 14), bg=self.config.BG, fg=self.config.MUTED_COLOR)
        subtitle.pack(pady=(0, 8))
        dept_subtitle = tk.Label(self.root, text="Developed by Department of Computer Science | University of South Carolina. April 2026",
                         font=("Arial", 12, "italic"), bg=self.config.BG, fg=self.config.DNN_COLOR)
        dept_subtitle.pack(pady=(0, 10))

        # ── Main 3-column layout ─────────────────────────────────────────────
        main = tk.Frame(self.root, bg=self.config.BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10)

        left   = tk.Frame(main, bg=self.config.BG)
        center = tk.Frame(main, bg=self.config.BG)
        right  = tk.Frame(main, bg=self.config.BG)

        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # ── LEFT — video + DNN panel ─────────────────────────────────────────
        vid_lf = self._lf(left, "Live Camera Feed", self.config.TEXT_COLOR)
        vid_lf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.video_label = tk.Label(vid_lf, bg='black')
        self.video_label.pack(padx=6, pady=6)

        dnn_lf = self._lf(left, "DNN Multimodal V2", self.config.DNN_COLOR)
        dnn_lf.pack(fill=tk.X, pady=4)
        self.dnn_class_lbl = self._big_label(dnn_lf, "--", self.config.DNN_COLOR)
        self.dnn_conf_lbl  = self._small_label(dnn_lf, "Confidence: --")
        self.dnn_lat_lbl   = self._small_label(dnn_lf, "Latency: --")
        self.dnn_threat_lbl = self._small_label(dnn_lf, "Threat Level: --")

        # ── CENTER — SNN panel + agree + chart ───────────────────────────────
        snn_lf = self._lf(center, "SNN Multimodal V2", self.config.SNN_COLOR)
        snn_lf.pack(fill=tk.X, pady=4)
        self.snn_class_lbl = self._big_label(snn_lf, "--", self.config.SNN_COLOR)
        self.snn_conf_lbl  = self._small_label(snn_lf, "Confidence: --")
        self.snn_lat_lbl   = self._small_label(snn_lf, "Latency: --")
        self.snn_threat_lbl = self._small_label(snn_lf, "Threat Level: --")
        self.snn_status    = self._small_label(snn_lf, "Status: waiting for SNN model...")

        agree_lf = self._lf(center, "Model Agreement", self.config.TEXT_COLOR)
        agree_lf.pack(fill=tk.X, pady=4)
        self.agree_lbl = self._big_label(agree_lf, "--", self.config.SAFE_COLOR, size=14)
        self.agree_pct_lbl = self._small_label(agree_lf, "Agreement rate: --")

        stats_lf = self._lf(center, "Statistics", self.config.TEXT_COLOR)
        stats_lf.pack(fill=tk.X, pady=4)
        self.stats_lbl = self._small_label(stats_lf,
            "Total: 0  |  Threats: 0  |  DNN avg: -- ms  |  SNN avg: -- ms")

        chart_lf = self._lf(center, "Detection Distribution", self.config.TEXT_COLOR)
        chart_lf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.fig = Figure(figsize=(4, 3), facecolor=self.config.PANEL_BG)
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0D1117')
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_lf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ── RIGHT — alert log + detection log ────────────────────────────────
        alert_lf = self._lf(right, "THREAT ALERTS", self.config.THREAT_COLOR)
        alert_lf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.alert_log = scrolledtext.ScrolledText(
            alert_lf, height=10, font=("Courier", 9),
            bg='#0D1117', fg=self.config.THREAT_COLOR, insertbackground='white')
        self.alert_log.pack(padx=6, pady=6, fill=tk.BOTH, expand=True)

        det_lf = self._lf(right, "Detection Log", self.config.TEXT_COLOR)
        det_lf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.det_log = scrolledtext.ScrolledText(
            det_lf, height=20, font=("Courier", 8),
            bg='#0D1117', fg='#CCCCCC', insertbackground='white')
        self.det_log.pack(padx=6, pady=6, fill=tk.BOTH, expand=True)

        self._update_gui()

    def _lf(self, parent, text, fg):
        return tk.LabelFrame(parent, text=text, font=("Arial", 11, "bold"),
                             bg=self.config.PANEL_BG, fg=fg,
                             bd=1, relief=tk.GROOVE)

    def _big_label(self, parent, text, color, size=16):
        lbl = tk.Label(parent, text=text, font=("Arial", size, "bold"),
                       bg=self.config.PANEL_BG, fg=color)
        lbl.pack(pady=4)
        return lbl

    def _small_label(self, parent, text):
        lbl = tk.Label(parent, text=text, font=("Arial", 10),
                       bg=self.config.PANEL_BG, fg=self.config.MUTED_COLOR)
        lbl.pack(pady=2)
        return lbl

    # -------------------------------------------------------------------------
    # NETWORK
    # -------------------------------------------------------------------------

    def _listen(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.config.HOST, self.config.PORT))
        srv.listen(1)
        self.logger.info(f"Waiting for Raspberry Pi on port {self.config.PORT}...")

        while self.running:
            try:
                conn, addr = srv.accept()
                self.logger.info(f"Pi connected from {addr}")
                buf = ""
                while self.running:
                    data = conn.recv(65536).decode('utf-8', errors='ignore')
                    if not data:
                        break
                    buf += data
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        try:
                            self._process(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                conn.close()
                self.logger.info("Pi disconnected")
            except Exception as e:
                if self.running:
                    self.logger.error(f"Listener error: {e}")

    def _process(self, packet):
        if packet.get('type') != 'detection':
            return

        data = packet['data']

        # Decode frame
        if 'frame' in packet:
            try:
                fb = bytes.fromhex(packet['frame'])
                arr = np.frombuffer(fb, dtype=np.uint8)
                self.current_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                pass

        self.detections.append(data)

        dnn = data.get('dnn', {})
        snn = data.get('snn')

        # Update stats
        n = self.stats['total'] + 1
        self.stats['total'] = n

        if dnn:
            dm = self.stats['dnn']
            dm['per_class'][dnn['class']] = dm['per_class'].get(dnn['class'], 0) + 1
            dm['avg_latency'] = (dm['avg_latency'] * (n-1) + dnn.get('inference_ms', 0)) / n
            if dnn.get('is_threat'):
                dm['threats'] += 1
                self.stats['threats'] += 1
                self._log_threat(data)

        if snn:
            sm = self.stats['snn']
            sm['per_class'][snn['class']] = sm['per_class'].get(snn['class'], 0) + 1
            sm['avg_latency'] = (sm['avg_latency'] * (n-1) + snn.get('inference_ms', 0)) / n
            if snn.get('is_threat'):
                sm['threats'] += 1

        if data.get('agreed') is not None:
            self.stats['agreement']['total'] += 1
            if data['agreed']:
                self.stats['agreement']['agreed'] += 1

        self._log_detection(data)

    def _log_threat(self, data):
        dnn = data.get('dnn', {})
        ts  = datetime.fromisoformat(data['timestamp']).strftime('%H:%M:%S')
        msg = f"[{ts}] *** THREAT: {dnn['class'].upper()} ({dnn['confidence']:.1%}) ***\n"
        self.alert_log.insert(tk.END, msg)
        self.alert_log.see(tk.END)

    def _log_detection(self, data):
        dnn = data.get('dnn', {})
        snn = data.get('snn')
        n   = data.get('detection_number', 0)
        ts  = datetime.fromisoformat(data['timestamp']).strftime('%H:%M:%S')

        dnn_str = f"DNN: {dnn.get('class','--'):15s} {dnn.get('confidence',0):5.1%} {dnn.get('inference_ms',0):6.1f}ms"
        if snn:
            agreed = "AGREE   " if data.get('agreed') else "MISMATCH"
            snn_str = f"SNN: {snn.get('class','--'):15s} {snn.get('confidence',0):5.1%} {snn.get('inference_ms',0):6.1f}ms"
            msg = f"[{ts}] #{n:04d} | {dnn_str} | {snn_str} | {agreed}\n"
        else:
            msg = f"[{ts}] #{n:04d} | {dnn_str}\n"

        self.det_log.insert(tk.END, msg)
        self.det_log.see(tk.END)

    # -------------------------------------------------------------------------
    # GUI UPDATE LOOP
    # -------------------------------------------------------------------------

    def _update_gui(self):
        if not self.running:
            return

        # Video frame
        if self.current_frame is not None:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            frame_rsz = cv2.resize(frame_rgb, (self.config.VIDEO_WIDTH, self.config.VIDEO_HEIGHT))

            if self.detections:
                d   = self.detections[-1]
                dnn = d.get('dnn', {})
                col = (255, 60, 60) if dnn.get('is_threat') else (60, 255, 120)
                cv2.putText(frame_rsz,
                            f"DNN: {dnn.get('class','--')} {dnn.get('confidence',0):.0%}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                snn = d.get('snn')
                if snn:
                    scol = (255, 60, 60) if snn.get('is_threat') else (80, 160, 255)
                    cv2.putText(frame_rsz,
                                f"SNN: {snn.get('class','--')} {snn.get('confidence',0):.0%}",
                                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, scol, 2)

            img   = Image.fromarray(frame_rsz)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # DNN / SNN panels
        if self.detections:
            d   = self.detections[-1]
            dnn = d.get('dnn', {})
            snn = d.get('snn')
            cfg = self.config

            # DNN
            dcol = cfg.THREAT_COLOR if dnn.get('is_threat') else cfg.DNN_COLOR
            self.dnn_class_lbl.configure(
                text=cfg.CLASS_DISPLAY.get(dnn.get('class','--'), dnn.get('class','--')).upper(),
                fg=dcol)
            self.dnn_conf_lbl.configure(text=f"Confidence: {dnn.get('confidence',0):.1%}")
            self.dnn_lat_lbl.configure(text=f"Latency: {dnn.get('inference_ms',0):.1f} ms")
            self.dnn_threat_lbl.configure(
                text=f"Threat Level: {dnn.get('threat','--')}",
                fg=cfg.THREAT_COLOR if dnn.get('is_threat') else cfg.SAFE_COLOR)

            # SNN
            if snn:
                scol = cfg.THREAT_COLOR if snn.get('is_threat') else cfg.SNN_COLOR
                self.snn_class_lbl.configure(
                    text=cfg.CLASS_DISPLAY.get(snn.get('class','--'), snn.get('class','--')).upper(),
                    fg=scol)
                self.snn_conf_lbl.configure(text=f"Confidence: {snn.get('confidence',0):.1%}")
                self.snn_lat_lbl.configure(text=f"Latency: {snn.get('inference_ms',0):.1f} ms")
                self.snn_threat_lbl.configure(
                    text=f"Threat Level: {snn.get('threat','--')}",
                    fg=cfg.THREAT_COLOR if snn.get('is_threat') else cfg.SAFE_COLOR)
                self.snn_status.configure(text="Status: ACTIVE", fg=cfg.SAFE_COLOR)

                # Agreement
                agreed = d.get('agreed')
                if agreed is True:
                    self.agree_lbl.configure(text="AGREE", fg=cfg.SAFE_COLOR)
                elif agreed is False:
                    self.agree_lbl.configure(text="MISMATCH", fg=cfg.THREAT_COLOR)
            else:
                self.snn_class_lbl.configure(text="NOT AVAILABLE", fg=cfg.MUTED_COLOR)
                self.snn_status.configure(text="Status: SNN model not loaded on Pi",
                                          fg=cfg.MUTED_COLOR)
                self.agree_lbl.configure(text="DNN ONLY", fg=cfg.DNN_COLOR)

        # Agreement rate
        ag = self.stats['agreement']
        if ag['total'] > 0:
            pct = 100 * ag['agreed'] / ag['total']
            self.agree_pct_lbl.configure(
                text=f"Agreement rate: {ag['agreed']}/{ag['total']} ({pct:.1f}%)")

        # Stats bar
        self.stats_lbl.configure(
            text=(f"Total: {self.stats['total']}  |  "
                  f"Threats: {self.stats['threats']}  |  "
                  f"DNN avg: {self.stats['dnn']['avg_latency']:.1f}ms  |  "
                  f"SNN avg: {self.stats['snn']['avg_latency']:.1f}ms"))

        # Chart every 5 detections
        if self.stats['total'] % 5 == 0 and self.stats['total'] > 0:
            self._update_chart()

        self.root.after(150, self._update_gui)

    def _update_chart(self):
        self.ax.clear()
        self.ax.set_facecolor('#0D1117')

        dnn_counts = [(c, v) for c, v in self.stats['dnn']['per_class'].items() if v > 0]
        snn_counts  = {c: v for c, v in self.stats['snn']['per_class'].items()}

        if not dnn_counts:
            return

        dnn_counts.sort(key=lambda x: x[1], reverse=True)
        labels = [c.replace('_', '\n') for c, _ in dnn_counts[:8]]
        dvals  = [v for _, v in dnn_counts[:8]]
        svals  = [snn_counts.get(c, 0) for c, _ in dnn_counts[:8]]

        x = np.arange(len(labels))
        w = 0.35
        self.ax.bar(x - w/2, dvals, w, label='DNN', color=self.config.DNN_COLOR, alpha=0.85)
        self.ax.bar(x + w/2, svals, w, label='SNN', color=self.config.SNN_COLOR, alpha=0.85)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, color=self.config.TEXT_COLOR, fontsize=7)
        self.ax.tick_params(colors=self.config.TEXT_COLOR, labelsize=7)
        self.ax.set_title('DNN vs SNN Detections', color=self.config.TEXT_COLOR, fontsize=9)
        self.ax.legend(fontsize=7, facecolor='#0D1117', labelcolor=self.config.TEXT_COLOR)
        for spine in self.ax.spines.values():
            spine.set_color('#444')
        self.canvas.draw()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.mainloop()

    def _close(self):
        self.running = False
        self.root.destroy()


# ============================================================================
# MAIN
# ============================================================================

def main():
    MonitoringDashboard(Config).run()

if __name__ == "__main__":
    main()
