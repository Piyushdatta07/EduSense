import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import time
import csv
import os
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH      = "./models/emotion_model.pt"
EMOTIONS        = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
SESSION_FOLDER  = "./sessions"
CAMERA_INDEX    = 0
ALERT_THRESHOLD = 0.4   # If class avg drops below 40% → alert

# ─── ENGAGEMENT SCORES ────────────────────────────────────
ENGAGEMENT_SCORES = {
    'Happy':     1.0,
    'Surprised': 0.9,
    'Neutral':   0.5,
    'Fearful':   0.3,
    'Disgusted': 0.2,
    'Sad':       0.1,
    'Angry':     0.1,
}

# ─── COLORS PER EMOTION (BGR) ─────────────────────────────
EMOTION_COLORS = {
    'Happy':     (0, 255, 255),
    'Surprised': (0, 165, 255),
    'Neutral':   (200, 200, 200),
    'Fearful':   (255, 0, 255),
    'Disgusted': (0, 255, 0),
    'Sad':       (255, 100, 50),
    'Angry':     (0, 0, 255),
}

os.makedirs(SESSION_FOLDER, exist_ok=True)

# ─── LOAD MODEL ───────────────────────────────────────────
def load_model():
    print("[INFO] Loading emotion model...")
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 7)
    )
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("[INFO] Model loaded!")
    return model

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── PREDICT EMOTION FOR ONE FACE ─────────────────────────
def predict_emotion(model, face_img):
    try:
        pil_img   = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        tensor    = transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)[0]
        idx        = torch.argmax(probs).item()
        return EMOTIONS[idx], probs[idx].item(), probs.numpy()
    except Exception:
        return "Neutral", 0.5, np.ones(7) / 7

# ─── DRAW STUDENT BOX ─────────────────────────────────────
def draw_student(frame, x, y, w, h, student_id, emotion, conf, score):
    color = EMOTION_COLORS.get(emotion, (200, 200, 200))

    # Face bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background
    label     = f"S{student_id}: {emotion} {conf*100:.0f}%"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, y - lh - 10), (x + lw + 6, y), color, -1)
    cv2.putText(frame, label, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Engagement bar below face
    bar_w = int(w * score)
    cv2.rectangle(frame, (x, y + h + 2), (x + w, y + h + 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y + h + 2), (x + bar_w, y + h + 10), color, -1)

# ─── DRAW CLASS DASHBOARD (top left panel) ────────────────
def draw_dashboard(frame, student_data, fps, alert):
    panel_h = 140 + len(student_data) * 22
    cv2.rectangle(frame, (8, 8), (300, panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 8), (300, panel_h), (80, 80, 80), 1)

    cv2.putText(frame, "EduSense Classroom Monitor",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}   Students: {len(student_data)}",
                (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    if student_data:
        avg_score = sum(d['score'] for d in student_data) / len(student_data)
        engaged   = sum(1 for d in student_data if d['score'] >= 0.5)
        pct       = int(avg_score * 100)

        # Engagement color
        if avg_score >= 0.7:
            eng_color = (0, 255, 100)
            eng_text  = "CLASS ENGAGED"
        elif avg_score >= 0.4:
            eng_color = (0, 200, 255)
            eng_text  = "MODERATE"
        else:
            eng_color = (0, 0, 255)
            eng_text  = "LOW ENGAGEMENT"

        cv2.putText(frame, f"Class Avg: {pct}%  {eng_text}",
                    (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, eng_color, 1)
        cv2.putText(frame, f"Focused: {engaged}/{len(student_data)} students",
                    (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Avg engagement bar
        bar_w = int(270 * avg_score)
        cv2.rectangle(frame, (15, 105), (285, 118), (50, 50, 50), -1)
        cv2.rectangle(frame, (15, 105), (15 + bar_w, 118), eng_color, -1)

        # Per student rows
        for i, d in enumerate(student_data):
            y_pos = 138 + i * 22
            color = EMOTION_COLORS.get(d['emotion'], (200, 200, 200))
            row   = f"S{d['id']}: {d['emotion']:<12} {d['score']*100:.0f}%"
            cv2.putText(frame, row, (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # Alert banner
    if alert:
        ah = frame.shape[0]
        cv2.rectangle(frame, (0, ah - 45), (frame.shape[1], ah), (0, 0, 180), -1)
        cv2.putText(frame, "!! LOW CLASS ENGAGEMENT — Consider a break or activity change!",
                    (10, ah - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

# ─── MAIN ─────────────────────────────────────────────────
def main():
    model       = load_model()
    face_det    = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap         = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    session_start = datetime.now()
    session_id    = session_start.strftime("%Y%m%d_%H%M%S")
    csv_path      = f"{SESSION_FOLDER}/classroom_{session_id}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([
            "Timestamp", "Student_ID", "Emotion",
            "Confidence", "Engagement_Score", "Class_Avg"
        ])

    print("[INFO] Camera opened. Press Q to quit.")
    print(f"[INFO] Saving session to: {csv_path}")

    prev_time   = time.time()
    alert_flag  = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all faces
        faces = face_det.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(60, 60)
        )

        student_data = []

        for idx, (x, y, w, h) in enumerate(faces):
            face_img = frame[y:y+h, x:x+w]
            emotion, conf, probs = predict_emotion(model, face_img)
            score = ENGAGEMENT_SCORES.get(emotion, 0.5)

            student_data.append({
                'id': idx + 1,
                'emotion': emotion,
                'confidence': conf,
                'score': score
            })

            draw_student(frame, x, y, w, h, idx + 1, emotion, conf, score)

        # Class average
        class_avg = (
            sum(d['score'] for d in student_data) / len(student_data)
            if student_data else 0.0
        )
        alert_flag = class_avg < ALERT_THRESHOLD and len(student_data) > 0

        # FPS
        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        draw_dashboard(frame, student_data, fps, alert_flag)

        # Save to CSV every 30 frames
        if frame_count % 30 == 0 and student_data:
            now = datetime.now().strftime("%H:%M:%S")
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                for d in student_data:
                    writer.writerow([
                        now, d['id'], d['emotion'],
                        f"{d['confidence']:.2f}",
                        f"{d['score']:.2f}",
                        f"{class_avg:.2f}"
                    ])

        cv2.imshow("EduSense - Classroom Emotion Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Session saved to: {csv_path}")
    print("[INFO] Done!")

if __name__ == "__main__":
    main()
