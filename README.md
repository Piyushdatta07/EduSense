# EmotionVR — Complete Project Guide
## Emotion-Adaptive VR Environments Using Facial Expression Recognition and Unity

---

## PROJECT STRUCTURE

```
EmotionVR/
│
├── python/
│   ├── train_model.py          ← STEP 2: Train CNN on FER-2013
│   └── emotion_server.py       ← STEP 4: Run real-time detection server
│
├── dashboard/
│   └── dashboard.py            ← STEP 5: Live analytics dashboard
│
├── unity_scripts/
│   ├── EmotionClient.cs        ← Polls Python server (attach to GameObject)
│   ├── EmotionManager.cs       ← Routes emotion to all sub-systems
│   └── EnvironmentControllers.cs ← Lighting, Audio, Weather, Skybox, NPC, HUD
│
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## PREREQUISITES

### Python side
- Python 3.10 or 3.11
- pip
- Webcam connected (built-in or USB)
- GPU recommended (runs on CPU too, just slower)

### Unity side
- Unity 2022.3 LTS or 2023.x
- XR Plugin Management (for VR)
- TextMeshPro (Window → Package Manager → search TextMeshPro)
- Newtonsoft.Json (Window → Package Manager → search "Newtonsoft Json")
- OpenXR Plugin (for Meta Quest / SteamVR)

---

## STEP 1 — Install Python Dependencies

Open terminal in the EmotionVR folder:

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

---

## STEP 2 — Download Dataset & Train the Model

### Download FER-2013
Go to: https://www.kaggle.com/datasets/msambare/fer2013

Download and extract so the folder looks like:
```
fer2013/
    train/
        Angry/    (3995 images)
        Disgusted/(436 images)
        Fearful/  (4097 images)
        Happy/    (7215 images)
        Neutral/  (4965 images)
        Sad/      (4830 images)
        Surprised/(3171 images)
    test/
        Angry/ Disgusted/ Fearful/ Happy/ Neutral/ Sad/ Surprised/
```

OR if you downloaded the CSV version, place `fer2013.csv` inside a `fer2013/` folder.

### Train the model

```bash
cd python

# Full training (30 epochs, ~45 min on GPU, ~3 hours on CPU)
python train_model.py --data_dir ../fer2013 --epochs 30 --batch_size 64

# Quick test run (5 epochs, just to check everything works)
python train_model.py --data_dir ../fer2013 --epochs 5 --batch_size 32
```

**Output after training:**
```
models/
    emotion_model.pt         ← PyTorch model (used by server)
    emotion_model.onnx       ← ONNX model (for Unity Barracuda)
    training_curves.png      ← Loss & accuracy graphs
    confusion_matrix.png     ← Per-emotion performance
```

**Expected accuracy:** 82–87% on FER-2013 test set

---

## STEP 3 — Test the Model (Optional Quick Check)

```bash
cd python
python - << 'EOF'
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

EMOTIONS = ['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3), nn.Linear(model.last_channel, 256),
    nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 7))
ckpt = torch.load('./models/emotion_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("Model loaded successfully!")
print(f"Best val accuracy: {ckpt.get('val_acc', 'N/A'):.2f}%")
EOF
```

---

## STEP 4 — Run the Emotion Server

```bash
cd python

# Start the server (keep this terminal open the whole time)
python emotion_server.py --model ./models/emotion_model.pt --port 5000
```

**You will see:**
```
==================================================
  EmotionVR Server running on http://localhost:5000
  Endpoints:
    GET  /emotion         → current emotion + profile
    GET  /status          → FPS + running status
    GET  /session_summary → analytics
    POST /reset           → reset session
==================================================
```

A **webcam preview window** will open showing your face with the detected emotion label.

### Test the server in browser or curl:
```bash
# Test emotion endpoint
curl http://localhost:5000/emotion

# Expected response:
{
  "emotion": "Happy",
  "confidence": 0.9123,
  "face_found": true,
  "all_scores": {"Angry":0.01,"Disgusted":0.00,...,"Happy":0.91,...},
  "profile": {"lighting":"bright_warm","music":"upbeat","weather":"sunny","color":"#FFD700"},
  "timestamp": 1700000000.0
}
```

---

## STEP 5 — Run the Dashboard

Open a **second terminal** (keep Step 4 running):

```bash
cd dashboard
python dashboard.py --server http://localhost:5000
```

The **EmotionVR Dashboard** window opens showing:
- Live emotion label + confidence bar
- Live bar chart of all 7 emotion scores
- Emotion timeline graph
- Session breakdown pie chart
- Export Report + Reset Session buttons

---

## STEP 6 — Unity Setup

### 6.1 Create New Unity Project
- Open Unity Hub → New Project
- Template: **3D (URP)** or **VR (XR)**
- Name: `EmotionVR`

### 6.2 Install Required Packages
Window → Package Manager:
- ✅ **TextMeshPro** (import TMP Essentials when prompted)
- ✅ **Newtonsoft Json** (search "com.unity.nuget.newtonsoft-json")
- ✅ **XR Plugin Management** (for VR headset)
- ✅ **OpenXR Plugin** (for Meta Quest / SteamVR)

### 6.3 Copy Unity Scripts
Copy all `.cs` files from `unity_scripts/` into:
```
Assets/Scripts/EmotionVR/
```

### 6.4 Scene Setup

**Create these GameObjects in your scene Hierarchy:**

```
Scene
├── EmotionSystem (Empty GameObject)
│   ├── EmotionClient       ← attach EmotionClient.cs
│   └── EmotionManager      ← attach EmotionManager.cs
│
├── Environment
│   ├── LightingController  ← attach LightingController.cs
│   ├── AudioController     ← attach AudioController.cs
│   ├── WeatherController   ← attach WeatherController.cs
│   └── SkyboxController    ← attach SkyboxController.cs
│
├── NPC_Character           ← attach NPCController.cs (with Animator)
│
├── VR_Rig (XR Origin)
│   └── Camera Offset
│       └── Main Camera
│
└── Canvas (World Space)    ← HUD overlay
    └── UIOverlayController ← attach UIOverlayController.cs
```

### 6.5 Wire Up Events

1. Select **EmotionClient** GameObject in Inspector
2. Find **On Emotion Changed** event field
3. Click **+** → drag **EmotionManager** → select `EmotionManager.OnEmotionChanged`
4. Click **+** → drag **EmotionManager** → select `EmotionManager.OnEmotionReceived`

### 6.6 Assign References in EmotionManager Inspector

Drag each controller GameObject into the corresponding field:
- Lighting Controller → LightingController GameObject
- Audio Controller    → AudioController GameObject
- Weather Controller  → WeatherController GameObject
- Skybox Controller   → SkyboxController GameObject
- NPC Controller      → NPC_Character GameObject
- UI Controller       → Canvas GameObject

### 6.7 Add Music Assets

In **AudioController** Inspector → Music Library:
- Add entries: `upbeat`, `soft_piano`, `calm_ambient`, `energetic`, `ambient`, `neutral`, `calm`
- Drag `.mp3`/`.wav` files into each AudioClip slot
- Royalty-free music: https://freemusicarchive.org or https://pixabay.com/music

### 6.8 Add Weather Particle Systems

For each weather type, create a Particle System:
- **Rain**: long thin particles, falling direction
- **Snow**: slow, drifting particles  
- **Fog**: slow large transparent particles
- **Sunny**: bright sparkle/lens flare particles

Drag each into WeatherController → Presets

### 6.9 Add Skybox Materials

Create/import HDR skybox materials for each mood:
- `day_bright`, `day_balanced`, `overcast`, `evening_blue`, `soft_white`, `dynamic_sunset`, `day_green`

Free skyboxes: https://assetstore.unity.com (search "free skybox")

Drag into SkyboxController → Presets

---

## STEP 7 — Run Everything Together

**Terminal 1:** (already running)
```bash
python emotion_server.py --model ./models/emotion_model.pt
```

**Terminal 2:** (already running)
```bash
python dashboard.py
```

**Unity:**
- Press **Play** in Unity Editor
- Unity polls `http://localhost:5000/emotion` every 100ms
- Your emotion changes → server detects → Unity environment adapts
- Dashboard shows live analytics

---

## STEP 8 — Build for VR Headset (Meta Quest 2/3)

### Configure XR Settings
1. Edit → Project Settings → XR Plug-in Management
2. Enable **OpenXR** (Android tab for Quest)
3. Add **Meta Quest feature set**

### Build
1. File → Build Settings
2. Platform: **Android**
3. Click **Switch Platform**
4. Player Settings → Company Name, Package Name
5. XR Settings → Stereo Rendering: **Single Pass Instanced**
6. **Build and Run** (Quest connected via USB)

### For PC VR (Oculus Link / SteamVR)
- Platform: **Windows**
- Build and Run with headset connected

---

## COMPLETE FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────┐
│                    PYTHON SIDE                          │
│                                                         │
│  Webcam → OpenCV Face Detect → CNN Model (MobileNetV2) │
│                    ↓                                    │
│          emotion_server.py (Flask REST API)             │
│          localhost:5000/emotion                         │
│                    ↓                                    │
│          session_log.csv (all frames logged)            │
│                    ↓                                    │
│          dashboard.py (live analytics)                  │
└──────────────────┬──────────────────────────────────────┘
                   │  HTTP GET every 100ms
                   ↓
┌─────────────────────────────────────────────────────────┐
│                    UNITY SIDE                           │
│                                                         │
│  EmotionClient.cs (polls API)                           │
│          ↓ onEmotionChanged event                       │
│  EmotionManager.cs (routes to sub-systems)              │
│     ├── LightingController  → color / intensity / fog  │
│     ├── AudioController     → music crossfade           │
│     ├── WeatherController   → rain / snow / particles  │
│     ├── SkyboxController    → skybox blend              │
│     ├── NPCController       → animator mood             │
│     └── UIOverlayController → HUD emotion label        │
│                                                         │
│  VR Headset renders adaptive environment                │
└─────────────────────────────────────────────────────────┘
```

---

## TROUBLESHOOTING

| Problem | Fix |
|---|---|
| `Model not found` | Run `python train_model.py` first |
| `Cannot open camera 0` | Change `--camera 1` or check webcam in Device Manager |
| `No faces detected` | Ensure good lighting, face the camera directly |
| Unity: `Connection refused` | Make sure `emotion_server.py` is running first |
| Unity: `Newtonsoft not found` | Install via Package Manager → `com.unity.nuget.newtonsoft-json` |
| Low FPS in Unity | Reduce poll interval: `pollInterval = 0.2f` in EmotionClient |
| Model accuracy < 70% | Train for more epochs: `--epochs 50` |

---

## EXPECTED RESULTS

| Metric | Target |
|---|---|
| Emotion recognition accuracy | ≥ 85% on FER-2013 |
| Server response latency | < 50ms |
| Unity poll → environment change | < 200ms total |
| VR frame rate | ≥ 72 FPS (Quest 2) |
| Dashboard refresh rate | 5 FPS (every 200ms) |

---

## DATASET CITATION

Goodfellow, I.J., et al. "Challenges in Representation Learning: A Report on Three Machine Learning Contests." ICML 2013 Challenges in Representation Learning Workshop.

FER-2013: https://www.kaggle.com/datasets/msambare/fer2013
