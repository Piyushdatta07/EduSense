"""
dashboard.py
============
Real-time emotion analytics dashboard.
Polls the emotion_server.py API and displays:
  - Live emotion label + confidence
  - Live bar chart of all 7 emotion scores
  - Timeline graph (emotion over session)
  - Session stats panel
  - Export to PDF button

USAGE:
    python dashboard.py --server http://localhost:5000

REQUIREMENTS:
    pip install requests matplotlib pillow reportlab
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import requests
import json
from collections import deque
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
POLL_INTERVAL_MS = 200  # poll server every 200ms

EMOTION_COLORS = {
    'Happy':     '#FFD700',
    'Sad':       '#4A90D9',
    'Angry':     '#E74C3C',
    'Fearful':   '#9B59B6',
    'Surprised': '#E67E22',
    'Disgusted': '#27AE60',
    'Neutral':   '#95A5A6',
}

EMOTION_EMOJIS = {
    'Happy': '😊', 'Sad': '😢', 'Angry': '😠',
    'Fearful': '😨', 'Surprised': '😲', 'Disgusted': '🤢', 'Neutral': '😐'
}

BG_COLOR     = '#1A1A2E'
CARD_COLOR   = '#16213E'
TEXT_COLOR   = '#E0E0E0'
ACCENT_COLOR = '#0F3460'
GREEN        = '#00C896'


class EmotionDashboard:
    def __init__(self, root, server_url):
        self.root       = root
        self.server_url = server_url.rstrip('/')
        self.running    = True

        # Data buffers
        self.timeline_emotions    = deque(maxlen=300)  # ~60 seconds at 200ms
        self.timeline_times       = deque(maxlen=300)
        self.timeline_confidences = deque(maxlen=300)
        self.session_start        = time.time()

        self._setup_ui()
        self._start_polling()

    # ── UI setup ──────────────────────────────────────────────────────────────
    def _setup_ui(self):
        self.root.title("EmotionVR — Real-Time Emotion Dashboard")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1280x800")
        self.root.resizable(True, True)

        # ── Title bar ──────────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg=ACCENT_COLOR, pady=8)
        title_frame.pack(fill='x')
        tk.Label(title_frame, text="🧠  EmotionVR — Adaptive Environment Dashboard",
                 font=('Segoe UI', 16, 'bold'), bg=ACCENT_COLOR, fg=TEXT_COLOR).pack()

        # ── Main layout: left panel | right charts ─────────────────────────
        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        left = tk.Frame(main_frame, bg=BG_COLOR, width=340)
        left.pack(side='left', fill='y', padx=(0,10))
        left.pack_propagate(False)

        right = tk.Frame(main_frame, bg=BG_COLOR)
        right.pack(side='right', fill='both', expand=True)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _card(self, parent, title, height=None):
        frame = tk.Frame(parent, bg=CARD_COLOR, bd=0, relief='flat')
        frame.pack(fill='x', pady=5, ipady=8, ipadx=8)
        if height:
            frame.configure(height=height)
        tk.Label(frame, text=title, font=('Segoe UI', 10, 'bold'),
                 bg=CARD_COLOR, fg='#7F8C8D').pack(anchor='w', padx=10, pady=(8,2))
        return frame

    def _build_left_panel(self, parent):
        # ── Current Emotion Card ───────────────────────────────────────────
        card = self._card(parent, "CURRENT EMOTION")
        self.emotion_label = tk.Label(card, text="😐  Neutral",
                                      font=('Segoe UI', 28, 'bold'),
                                      bg=CARD_COLOR, fg=TEXT_COLOR)
        self.emotion_label.pack(pady=4)

        self.confidence_label = tk.Label(card, text="Confidence: 0.0%",
                                         font=('Segoe UI', 12),
                                         bg=CARD_COLOR, fg='#7F8C8D')
        self.confidence_label.pack()

        self.confidence_bar = ttk.Progressbar(card, length=280, mode='determinate')
        self.confidence_bar.pack(pady=6, padx=10)

        # ── Face Status ────────────────────────────────────────────────────
        self.face_status = tk.Label(card, text="👤 Face: Not Detected",
                                    font=('Segoe UI', 10), bg=CARD_COLOR, fg='#E74C3C')
        self.face_status.pack(pady=2)

        # ── Environment Profile Card ───────────────────────────────────────
        env_card = self._card(parent, "UNITY ENVIRONMENT PROFILE")
        self.env_labels = {}
        for key in ['lighting', 'music', 'weather']:
            row = tk.Frame(env_card, bg=CARD_COLOR)
            row.pack(fill='x', padx=10, pady=2)
            tk.Label(row, text=key.capitalize()+":", font=('Segoe UI', 10, 'bold'),
                     bg=CARD_COLOR, fg='#7F8C8D', width=10, anchor='w').pack(side='left')
            lbl = tk.Label(row, text="—", font=('Segoe UI', 10),
                           bg=CARD_COLOR, fg=TEXT_COLOR)
            lbl.pack(side='left')
            self.env_labels[key] = lbl

        # ── Session Stats Card ─────────────────────────────────────────────
        stats_card = self._card(parent, "SESSION STATS")
        self.stats_labels = {}
        for key in ['Duration', 'FPS', 'Dominant Emotion', 'Total Frames']:
            row = tk.Frame(stats_card, bg=CARD_COLOR)
            row.pack(fill='x', padx=10, pady=2)
            tk.Label(row, text=key+":", font=('Segoe UI', 9, 'bold'),
                     bg=CARD_COLOR, fg='#7F8C8D', width=16, anchor='w').pack(side='left')
            lbl = tk.Label(row, text="—", font=('Segoe UI', 9),
                           bg=CARD_COLOR, fg=TEXT_COLOR)
            lbl.pack(side='left')
            self.stats_labels[key] = lbl

        # ── Server Status ──────────────────────────────────────────────────
        self.server_status = tk.Label(parent, text="⚪ Connecting...",
                                      font=('Segoe UI', 9), bg=BG_COLOR, fg='#7F8C8D')
        self.server_status.pack(pady=4)

        # ── Buttons ────────────────────────────────────────────────────────
        btn_frame = tk.Frame(parent, bg=BG_COLOR)
        btn_frame.pack(fill='x', pady=8)

        tk.Button(btn_frame, text="📊 Export Report",
                  font=('Segoe UI', 10, 'bold'), bg='#0F3460', fg=TEXT_COLOR,
                  relief='flat', padx=10, pady=6,
                  command=self._export_report).pack(side='left', padx=4)

        tk.Button(btn_frame, text="🔄 Reset Session",
                  font=('Segoe UI', 10, 'bold'), bg='#E74C3C', fg=TEXT_COLOR,
                  relief='flat', padx=10, pady=6,
                  command=self._reset_session).pack(side='left', padx=4)

    def _build_right_panel(self, parent):
        # ── Top: Bar chart ─────────────────────────────────────────────────
        top = tk.Frame(parent, bg=BG_COLOR)
        top.pack(fill='both', expand=True)

        # Bar chart
        bar_frame = tk.Frame(top, bg=CARD_COLOR)
        bar_frame.pack(side='left', fill='both', expand=True, padx=(0,5))
        tk.Label(bar_frame, text="EMOTION SCORES (LIVE)",
                 font=('Segoe UI', 10, 'bold'), bg=CARD_COLOR, fg='#7F8C8D').pack(pady=4)

        self.bar_fig = Figure(figsize=(5, 3.5), facecolor=CARD_COLOR)
        self.bar_ax  = self.bar_fig.add_subplot(111)
        self.bar_ax.set_facecolor(CARD_COLOR)
        self.bar_canvas = FigureCanvasTkAgg(self.bar_fig, master=bar_frame)
        self.bar_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Pie chart
        pie_frame = tk.Frame(top, bg=CARD_COLOR)
        pie_frame.pack(side='right', fill='both', expand=True)
        tk.Label(pie_frame, text="SESSION BREAKDOWN",
                 font=('Segoe UI', 10, 'bold'), bg=CARD_COLOR, fg='#7F8C8D').pack(pady=4)

        self.pie_fig = Figure(figsize=(4, 3.5), facecolor=CARD_COLOR)
        self.pie_ax  = self.pie_fig.add_subplot(111)
        self.pie_ax.set_facecolor(CARD_COLOR)
        self.pie_canvas = FigureCanvasTkAgg(self.pie_fig, master=pie_frame)
        self.pie_canvas.get_tk_widget().pack(fill='both', expand=True)

        # ── Bottom: Timeline ───────────────────────────────────────────────
        tl_frame = tk.Frame(parent, bg=CARD_COLOR)
        tl_frame.pack(fill='both', expand=True, pady=(5,0))
        tk.Label(tl_frame, text="EMOTION TIMELINE",
                 font=('Segoe UI', 10, 'bold'), bg=CARD_COLOR, fg='#7F8C8D').pack(pady=4)

        self.tl_fig = Figure(figsize=(10, 2.5), facecolor=CARD_COLOR)
        self.tl_ax  = self.tl_fig.add_subplot(111)
        self.tl_ax.set_facecolor(CARD_COLOR)
        self.tl_canvas = FigureCanvasTkAgg(self.tl_fig, master=tl_frame)
        self.tl_canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── Polling ───────────────────────────────────────────────────────────────
    def _start_polling(self):
        self._poll()

    def _poll(self):
        if not self.running:
            return
        threading.Thread(target=self._fetch_and_update, daemon=True).start()
        self.root.after(POLL_INTERVAL_MS, self._poll)

    def _fetch_and_update(self):
        try:
            r  = requests.get(f"{self.server_url}/emotion", timeout=1)
            rs = requests.get(f"{self.server_url}/status",  timeout=1)
            rsum = requests.get(f"{self.server_url}/session_summary", timeout=1)

            if r.status_code == 200:
                data    = r.json()
                sdata   = rs.json() if rs.status_code == 200 else {}
                sumdata = rsum.json() if rsum.status_code == 200 else {}
                self.root.after(0, lambda: self._update_ui(data, sdata, sumdata))
                self.root.after(0, lambda: self.server_status.config(
                    text="🟢 Server Connected", fg=GREEN))
        except Exception as e:
            self.root.after(0, lambda: self.server_status.config(
                text=f"🔴 Server Offline: {str(e)[:40]}", fg='#E74C3C'))

    def _update_ui(self, data, sdata, sumdata):
        emotion    = data.get('emotion', 'Neutral')
        confidence = data.get('confidence', 0.0)
        all_scores = data.get('all_scores', {})
        face_found = data.get('face_found', False)
        profile    = data.get('profile', {})

        # Timeline
        self.timeline_emotions.append(emotion)
        self.timeline_times.append(time.time() - self.session_start)
        self.timeline_confidences.append(confidence)

        # Main label
        emoji = EMOTION_EMOJIS.get(emotion, '😐')
        color = EMOTION_COLORS.get(emotion, TEXT_COLOR)
        self.emotion_label.config(text=f"{emoji}  {emotion}", fg=color)
        self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
        self.confidence_bar['value'] = confidence * 100

        # Face status
        if face_found:
            self.face_status.config(text="👤 Face: Detected ✓", fg=GREEN)
        else:
            self.face_status.config(text="👤 Face: Not Detected", fg='#E74C3C')

        # Environment profile
        for key in ['lighting', 'music', 'weather']:
            val = profile.get(key, '—').replace('_', ' ').title()
            self.env_labels[key].config(text=val, fg=color)

        # Stats
        duration = int(time.time() - self.session_start)
        self.stats_labels['Duration'].config(text=f"{duration//60}m {duration%60}s")
        self.stats_labels['FPS'].config(text=f"{sdata.get('fps', 0):.1f}")
        self.stats_labels['Dominant Emotion'].config(
            text=sumdata.get('dominant_emotion', '—'))
        self.stats_labels['Total Frames'].config(
            text=str(sumdata.get('total_frames', '—')))

        # Charts
        self._update_bar_chart(all_scores, emotion)
        self._update_timeline()
        if sumdata.get('emotion_breakdown_pct'):
            self._update_pie(sumdata['emotion_breakdown_pct'])

    def _update_bar_chart(self, all_scores, current_emotion):
        ax = self.bar_ax
        ax.clear()
        ax.set_facecolor(CARD_COLOR)

        scores = [all_scores.get(e, 0) * 100 for e in EMOTIONS]
        colors = [EMOTION_COLORS.get(e, '#95A5A6') for e in EMOTIONS]
        bars   = ax.barh(EMOTIONS, scores, color=colors, edgecolor='none')

        # Highlight current
        for i, e in enumerate(EMOTIONS):
            if e == current_emotion:
                bars[i].set_linewidth(2)
                bars[i].set_edgecolor('white')

        ax.set_xlim(0, 100)
        ax.set_xlabel('%', color=TEXT_COLOR, fontsize=8)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.spines[:].set_visible(False)
        ax.set_facecolor(CARD_COLOR)
        self.bar_fig.patch.set_facecolor(CARD_COLOR)

        for spine in ax.spines.values():
            spine.set_color('#444')

        self.bar_canvas.draw()

    def _update_timeline(self):
        ax = self.tl_ax
        ax.clear()
        ax.set_facecolor(CARD_COLOR)

        if len(self.timeline_times) < 2:
            return

        times    = list(self.timeline_times)
        emotions = list(self.timeline_emotions)
        emotion_idx = [EMOTIONS.index(e) if e in EMOTIONS else 4 for e in emotions]

        # Scatter with emotion colors
        scatter_colors = [EMOTION_COLORS.get(e, '#95A5A6') for e in emotions]
        ax.scatter(times, emotion_idx, c=scatter_colors, s=15, zorder=3)
        ax.plot(times, emotion_idx, color='#444', linewidth=0.5, zorder=2)

        ax.set_yticks(range(len(EMOTIONS)))
        ax.set_yticklabels(EMOTIONS, fontsize=7, color=TEXT_COLOR)
        ax.set_xlabel('Time (s)', color=TEXT_COLOR, fontsize=8)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax.spines[:].set_color('#444')
        self.tl_fig.patch.set_facecolor(CARD_COLOR)
        self.tl_canvas.draw()

    def _update_pie(self, breakdown):
        ax = self.pie_ax
        ax.clear()
        ax.set_facecolor(CARD_COLOR)

        labels  = [e for e, v in breakdown.items() if v > 0]
        sizes   = [v for v in breakdown.values() if v > 0]
        colors  = [EMOTION_COLORS.get(e, '#95A5A6') for e in labels]

        if not sizes:
            return

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.0f%%', startangle=90,
            textprops={'color': TEXT_COLOR, 'fontsize': 7}
        )
        for at in autotexts:
            at.set_fontsize(7)

        self.pie_fig.patch.set_facecolor(CARD_COLOR)
        self.pie_canvas.draw()

    # ── Actions ───────────────────────────────────────────────────────────────
    def _export_report(self):
        try:
            r = requests.get(f"{self.server_url}/session_summary", timeout=2)
            if r.status_code != 200:
                messagebox.showerror("Error", "Could not fetch session data")
                return
            data     = r.json()
            filename = f"EmotionVR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write("=" * 50 + "\n")
                f.write("  EmotionVR Session Report\n")
                f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Duration        : {data.get('duration_seconds', 0):.0f} seconds\n")
                f.write(f"Total Frames    : {data.get('total_frames', 0)}\n")
                f.write(f"Dominant Emotion: {data.get('dominant_emotion', 'N/A')}\n\n")
                f.write("Emotion Breakdown:\n")
                for e, pct in data.get('emotion_breakdown_pct', {}).items():
                    bar = '█' * int(pct / 5)
                    f.write(f"  {e:<12} {bar:<20} {pct:.1f}%\n")
            messagebox.showinfo("Exported", f"Report saved as:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_session(self):
        try:
            requests.post(f"{self.server_url}/reset", timeout=2)
            self.timeline_emotions.clear()
            self.timeline_times.clear()
            self.timeline_confidences.clear()
            self.session_start = time.time()
            messagebox.showinfo("Reset", "Session reset successfully")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_close(self):
        self.running = False
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='http://localhost:5000',
                        help='URL of emotion_server.py')
    args = parser.parse_args()

    root = tk.Tk()
    app  = EmotionDashboard(root, args.server)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
