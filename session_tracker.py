import requests
import time
import csv
import os
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────
SERVER_URL = "http://localhost:5000/emotion"
SAMPLE_INTERVAL = 5          # Check emotion every 5 seconds
ALERT_THRESHOLD = 30         # Seconds before disengagement alert
SESSION_FOLDER = "./sessions" # Where CSV reports are saved

# ─── ENGAGEMENT RULES ─────────────────────────────────────
ENGAGEMENT_SCORES = {
    'Happy':     1.0,   # Fully engaged
    'Surprised': 0.9,   # Highly engaged
    'Neutral':   0.5,   # Partially focused
    'Fearful':   0.3,   # Stressed / anxious
    'Sad':       0.1,   # Disengaged
    'Angry':     0.1,   # Disengaged
    'Disgusted': 0.2,   # Disengaged
}

DISENGAGED_EMOTIONS = ['Sad', 'Angry', 'Disgusted', 'Fearful']
ENGAGED_EMOTIONS    = ['Happy', 'Surprised']

# ─── SETUP ────────────────────────────────────────────────
os.makedirs(SESSION_FOLDER, exist_ok=True)

def get_emotion():
    """Fetch current emotion from Flask server"""
    try:
        response = requests.get(SERVER_URL, timeout=3)
        data = response.json()
        return data.get("emotion", "Neutral"), data.get("confidence", 0.0)
    except Exception:
        return "Neutral", 0.0

def engagement_label(score):
    """Convert score to human readable label"""
    if score >= 0.8:
        return "HIGHLY ENGAGED [HIGH]"
    elif score >= 0.5:
        return "MODERATE       [MED] "
    elif score >= 0.3:
        return "LOW FOCUS      [LOW] "
    else:
        return "DISENGAGED     [NONE]"

def print_banner():
    print("=" * 55)
    print("   EduSense -- Student Engagement Tracker")
    print("=" * 55)
    print(f"  Sampling every  : {SAMPLE_INTERVAL} seconds")
    print(f"  Alert threshold : {ALERT_THRESHOLD} seconds")
    print(f"  Reports saved to: {SESSION_FOLDER}/")
    print("=" * 55)
    print("  Press CTRL+C to end session and save report")
    print("=" * 55)
    print()

def run_session():
    print_banner()

    session_start = datetime.now()
    session_id    = session_start.strftime("%Y%m%d_%H%M%S")
    csv_path      = f"{SESSION_FOLDER}/session_{session_id}.csv"

    records           = []
    disengaged_since  = None
    total_score       = 0.0
    sample_count      = 0
    emotion_counts    = {e: 0 for e in ENGAGEMENT_SCORES}

    # Write CSV header
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Elapsed_Sec",
            "Emotion", "Confidence",
            "Engagement_Score", "Engagement_Label"
        ])

    print(f"  Session started: {session_start.strftime('%H:%M:%S')}\n")

    try:
        while True:
            now          = datetime.now()
            elapsed      = (now - session_start).seconds
            emotion, conf = get_emotion()

            score  = ENGAGEMENT_SCORES.get(emotion, 0.5)
            label  = engagement_label(score)

            total_score  += score
            sample_count += 1
            avg_score     = total_score / sample_count
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            # ── Print live status ──────────────────────────
            print(f"  [{now.strftime('%H:%M:%S')}]  "
                  f"Emotion: {emotion:<12} "
                  f"Conf: {conf*100:.0f}%  "
                  f"Score: {score:.1f}  "
                  f"{label}")

            # ── Save to CSV ────────────────────────────────
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    now.strftime("%H:%M:%S"),
                    elapsed, emotion,
                    f"{conf:.2f}", f"{score:.2f}", label
                ])

            records.append({
                "time": now, "elapsed": elapsed,
                "emotion": emotion, "confidence": conf,
                "score": score
            })

            # ── Disengagement Alert ────────────────────────
            if emotion in DISENGAGED_EMOTIONS:
                if disengaged_since is None:
                    disengaged_since = now
                else:
                    disengaged_duration = (now - disengaged_since).seconds
                    if disengaged_duration >= ALERT_THRESHOLD:
                        print()
                        print("  !! ALERT: Student disengaged for "
                              f"{disengaged_duration}s")
                        print("  >> Suggestion: Take a short break "
                              "or switch topics!")
                        print()
                        disengaged_since = None  # Reset after alert
            else:
                disengaged_since = None  # Reset if engaged again

            time.sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        print("\n")
        print("=" * 55)
        print("   SESSION SUMMARY")
        print("=" * 55)

        session_end      = datetime.now()
        total_minutes    = (session_end - session_start).seconds // 60
        total_seconds    = (session_end - session_start).seconds
        final_avg        = (total_score / sample_count) if sample_count > 0 else 0

        print(f"  Duration        : {total_minutes}m {total_seconds % 60}s")
        print(f"  Total samples   : {sample_count}")
        print(f"  Avg Engagement  : {final_avg:.2f} — "
              f"{engagement_label(final_avg)}")
        print()

        # ── Emotion breakdown ─────────────────────────────
        print("  Emotion Breakdown:")
        for emotion, count in sorted(
            emotion_counts.items(), key=lambda x: -x[1]
        ):
            if count > 0:
                pct = (count / sample_count) * 100
                bar = "█" * int(pct / 5)
                print(f"    {emotion:<12} {bar:<20} {pct:.0f}%")

        print()

        # ── Best and worst focus periods ──────────────────
        if records:
            best  = max(records, key=lambda x: x["score"])
            worst = min(records, key=lambda x: x["score"])
            print(f"  Best focus at   : {best['time'].strftime('%H:%M:%S')} "
                  f"({best['emotion']})")
            print(f"  Lowest focus at : {worst['time'].strftime('%H:%M:%S')} "
                  f"({worst['emotion']})")

        print()
        print(f"  ✓ Report saved : {csv_path}")
        print("=" * 55)

if __name__ == "__main__":
    run_session()
