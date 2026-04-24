"""
emotion_server.py
=================
Real-time emotion detection server.
- Captures webcam frames
- Detects faces using OpenCV Haar Cascade
- Classifies emotion using trained PyTorch model
- Serves results via Flask REST API on localhost:5000
- Logs session data to session_log.csv

USAGE:
    python emotion_server.py --model ./models/emotion_model.pt --port 5000

ENDPOINTS:
    GET  /emotion          → { "emotion": "Happy", "confidence": 0.92, "all_scores": {...} }
    GET  /session_summary  → { "dominant": "Happy", "timeline": [...], "duration": 120 }
    GET  /status           → { "running": true, "fps": 28.3 }
    POST /reset            → resets session log
"""

import os
import cv2
import time
import json
import threading
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS
from collections import deque, Counter
import csv
from datetime import datetime

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS   = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
IMG_SIZE   = 48
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Emotion → environment profile mapping (sent to Unity)
EMOTION_PROFILES = {
    'Happy':     {'lighting': 'bright_warm',  'music': 'upbeat',  'weather': 'sunny',    'color': '#FFD700'},
    'Sad':       {'lighting': 'warm_golden',  'music': 'soft_piano','weather': 'light_rain','color': '#4A90D9'},
    'Angry':     {'lighting': 'cool_blue',    'music': 'ambient',  'weather': 'calm_rain','color': '#4169E1'},
    'Fearful':   {'lighting': 'soft_white',   'music': 'calm',     'weather': 'fog',      'color': '#9B59B6'},
    'Surprised': {'lighting': 'dynamic',      'music': 'energetic','weather': 'clear',    'color': '#E67E22'},
    'Disgusted': {'lighting': 'cool_green',   'music': 'neutral',  'weather': 'breeze',   'color': '#27AE60'},
    'Neutral':   {'lighting': 'balanced',     'music': 'ambient',  'weather': 'default',  'color': '#95A5A6'},
}

app = Flask(__name__)
CORS(app)  # Allow Unity to call from localhost

# ── Global state ──────────────────────────────────────────────────────────────
state = {
    'emotion':     'Neutral',
    'confidence':  0.0,
    'all_scores':  {e: 0.0 for e in EMOTIONS},
    'fps':         0.0,
    'running':     False,
    'face_found':  False,
    'timeline':    [],          # [(timestamp, emotion, confidence), ...]
    'start_time':  None,
}
lock = threading.Lock()


# ── Model loader ──────────────────────────────────────────────────────────────
def build_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(EMOTIONS))
    )
    return model


def load_model(model_path):
    model = build_model().to(DEVICE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Run: python train_model.py --data_dir ./fer2013 first."
        )
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"[INFO] Model loaded from {model_path}")
    return model


# ── Image preprocessing ───────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def detect_and_classify(frame, model):
    """Detect face in frame and classify emotion. Returns (emotion, confidence, all_scores)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30))
    if len(faces) == 0:
        return None, 0.0, {e: 0.0 for e in EMOTIONS}

    # Use the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Crop + pad
    pad = int(0.1 * max(w, h))
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    face_img = frame[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    tensor = preprocess(face_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    emotion_idx = int(np.argmax(probs))
    emotion     = EMOTIONS[emotion_idx]
    confidence  = float(probs[emotion_idx])
    all_scores  = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}

    return emotion, confidence, all_scores


# ── Camera thread ─────────────────────────────────────────────────────────────
def camera_loop(model, camera_idx, log_path, smooth_window=5):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_idx}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # CSV logger
    log_file   = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['timestamp', 'emotion', 'confidence'] + EMOTIONS)

    emotion_buffer = deque(maxlen=smooth_window)  # temporal smoothing
    fps_times      = deque(maxlen=30)

    print(f"[INFO] Camera thread started. Press Ctrl+C to stop.")

    with lock:
        state['running']    = True
        state['start_time'] = time.time()

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed, retrying...")
            time.sleep(0.1)
            continue

        emotion, confidence, all_scores = detect_and_classify(frame, model)

        # Temporal smoothing — use most common emotion in last N frames
        if emotion is not None:
            emotion_buffer.append(emotion)
            smoothed = Counter(emotion_buffer).most_common(1)[0][0]
            face_found = True
        else:
            smoothed   = state['emotion']  # hold last known
            face_found = False

        # FPS
        fps_times.append(time.time() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

        # Update global state
        ts = time.time()
        with lock:
            state['emotion']    = smoothed
            state['confidence'] = confidence
            state['all_scores'] = all_scores
            state['fps']        = fps
            state['face_found'] = face_found
            state['timeline'].append({'time': ts, 'emotion': smoothed, 'confidence': confidence})

        # CSV log
        log_writer.writerow([
            datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f'),
            smoothed, f"{confidence:.4f}"
        ] + [f"{all_scores.get(e,0):.4f}" for e in EMOTIONS])
        log_file.flush()

        # Optional preview window (comment out if running headless)
        if emotion is not None:
            label = f"{smoothed} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("EmotionVR - Face Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    log_file.close()
    cv2.destroyAllWindows()
    with lock:
        state['running'] = False
    print("[INFO] Camera thread stopped.")


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route('/emotion', methods=['GET'])
def get_emotion():
    """
    Unity polls this every 100ms.
    Returns current emotion + environment profile.
    """
    with lock:
        emotion    = state['emotion']
        confidence = state['confidence']
        all_scores = state['all_scores']
        face_found = state['face_found']

    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES['Neutral'])
    return jsonify({
        'emotion':     emotion,
        'confidence':  round(confidence, 4),
        'face_found':  face_found,
        'all_scores':  {k: round(v, 4) for k, v in all_scores.items()},
        'profile':     profile,
        'timestamp':   time.time()
    })


@app.route('/status', methods=['GET'])
def get_status():
    with lock:
        return jsonify({
            'running':    state['running'],
            'fps':        round(state['fps'], 1),
            'face_found': state['face_found'],
            'uptime':     round(time.time() - state['start_time'], 1) if state['start_time'] else 0
        })


@app.route('/session_summary', methods=['GET'])
def session_summary():
    with lock:
        timeline  = list(state['timeline'])
        start     = state['start_time']

    if not timeline:
        return jsonify({'message': 'No data yet'})

    emotions  = [t['emotion'] for t in timeline]
    counts    = Counter(emotions)
    total     = len(emotions)
    dominant  = counts.most_common(1)[0][0]
    duration  = round(time.time() - start, 1) if start else 0
    breakdown = {e: round(counts.get(e, 0) / total * 100, 1) for e in EMOTIONS}

    return jsonify({
        'dominant_emotion': dominant,
        'duration_seconds': duration,
        'total_frames':     total,
        'emotion_breakdown_pct': breakdown,
        'timeline_sample':  timeline[-20:]  # last 20 entries
    })


@app.route('/reset', methods=['POST'])
def reset_session():
    with lock:
        state['timeline']   = []
        state['start_time'] = time.time()
    return jsonify({'message': 'Session reset'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='./models/emotion_model.pt')
    parser.add_argument('--port',   type=int, default=5000)
    parser.add_argument('--camera', type=int, default=0, help='Camera index (0=default webcam)')
    parser.add_argument('--log',    default='./session_log.csv')
    args = parser.parse_args()

    model = load_model(args.model)

    # Start camera in background thread
    cam_thread = threading.Thread(
        target=camera_loop,
        args=(model, args.camera, args.log),
        daemon=True
    )
    cam_thread.start()

    print(f"\n{'='*50}")
    print(f"  EmotionVR Server running on http://localhost:{args.port}")
    print(f"  Endpoints:")
    print(f"    GET  /emotion         → current emotion + profile")
    print(f"    GET  /status          → FPS + running status")
    print(f"    GET  /session_summary → analytics")
    print(f"    POST /reset           → reset session")
    print(f"{'='*50}\n")

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
