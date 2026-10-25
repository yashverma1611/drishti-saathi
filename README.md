# Drishti Saathi | दृष्टि साथी
### Offline, Multi-Model Edge AI Voice Assistant for the Visually Impaired

> **"Your AI Companion for Independent, Safe Navigation"**  
> *100% On-Device Processing • Zero Cloud Latency • 4 Cooperative Edge AI Models + Master Fusion Arbiter*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/Platform-Web%20%7C%20PWA%20%7C%20Mobile%20%7C%20Desktop-cyan.svg)](#)
[![Privacy](https://img.shields.io/badge/Privacy-Zero%20Data%20Egress-emerald.svg)](#)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-TensorFlow.js%20%7C%20COCO--SSD%20%7C%20Optical%20Raycasting-purple.svg)](#)

---

## 🚀 Overview

**Drishti Saathi (दृष्टि साथी)** is an edge-native, privacy-preserving Progressive Web App (PWA) designed to empower blind and visually impaired individuals with real-time, directional obstacle detection, staircase and drop-off warnings, and hands-free bilingual voice navigation.

While traditional white canes only detect physical obstacles in immediate contact with the ground, Drishti Saathi looks ahead using the phone's camera and inertial sensors. It computes real-time distance, detects ground drop-offs and stairs, and warns the user in natural spoken Hindi and English with **zero cloud dependency and zero latency**.

---

## 🧠 Cooperative Multi-AI Ensemble Architecture (4 Models + 1 Master Arbiter)

In accordance with strict hackathon multi-model rules, Drishti Saathi distributes the perception and navigation workload across **4 specialized, cooperative AI models**, coordinated by an overarching **Master Sensor Fusion Arbiter**:

```mermaid
flowchart TD
    Cam["Live Camera Feed (60 FPS)"] --> M1["MODEL 1: Neural Perception AI\n(COCO-SSD 80 Cls + Teachable Vision)"]
    Cam --> M2["MODEL 2: Surface Hazard AI\n(Stairs & Pothole Void Detector)"]
    Cam --> M4["MODEL 4: Collision & Depth AI\n(3-Sector Perspective Raycaster + TTC)"]
    
    Sensors["Device IMU & Compass\n(Magnetometer & Accelerometer)"] --> M3["MODEL 3: Direction Guidance AI\n(Cardinal Heading & Everyday Steering)"]
    Sensors --> Steps["Dead-Reckoning Step Detector"]

    M1 -->|"Object Labels, Bounding Boxes, Optical Dimensions"| Fusion["MASTER SENSOR FUSION ARBITER\n(Priority Preemption & Conflict Resolver)"]
    M2 -->|"Stair Risers, Ground Depressions, Distance"| Fusion
    M3 -->|"Compass Degrees, Cardinal Sector, Safe Path"| Fusion
    M4 -->|"Raycast Distance, Ground Contact, Time-To-Collision"| Fusion

    Fusion --> Speech["Bilingual Voice Announcer\n(Natural Hindi & English TTS)"]
    Fusion --> Earcons["Synthetic Web Audio Earcons\n(Instant Acoustic Cues)"]
    Fusion --> Haptics["Tactile Haptic Engine\n(Pattern Vibration Feedback)"]
    Fusion --> HUD["High-Contrast Futuristic HUD\n(Real-Time Visual Telemetry)"]
```

---

### Detailed Breakdown of the 4 Cooperative AI Models:

#### 1. Model 1: Neural Object Perception & Classification AI
* **Engine**: Quantized TensorFlow.js MobileNetV2 Lite (COCO-SSD) + "Drishti Seekho" (दृष्टि सीखो) Teachable Vision.
* **Capabilities**:
  * Recognizes **80 real-world object classes** on-device (pedestrians, vehicles, furniture, bottles, cups, laptops, chairs, beds, bags, etc.).
  * **Proactive Automatic Voice Announcer**: As soon as an object enters the walking path ($< 2.2\text{m}$), Drishti Saathi automatically speaks the warning (e.g. *"सामने बिस्तर है, दूरी 1.2 मीटर। संभलकर चलें।"* / *"Caution: Laptop directly ahead, 0.7 meters. Step carefully."*).
  * **Teachable Vision (दृष्टि सीखो)**: Allows visually impaired users to train and recognize their private personal objects (medicine bottles, keychains, walking sticks, badges) locally using cosine similarity feature vectors.

#### 2. Model 2: Surface Hazard, Staircase & Pothole Detection AI
* **Engine**: Dual-Pass Optical Edge Gradient & Ground Void Anomaly Analyzer.
* **Capabilities**:
  * **Stairs & Steps Detection**: Scans the central corridor for parallel horizontal riser-tread transitions where vertical gradients ($\frac{\partial I}{\partial y}$) dominate horizontal gradients ($\frac{\partial I}{\partial x}$). Accurately distinguishes descending/ascending staircases.
  * **Potholes & Ground Voids**: Identifies localized drop-off depressions where ground-level luminance sharply plummets below the surrounding floor baseline by $> 30\text{ Lum}$, preventing perilous falls into road craters or open manholes.
  * Instantly triggers emergency audio stop sirens and rapid triple-pulse vibrations.

#### 3. Model 3: Everyday Natural Direction & Spatial Compass Guidance AI
* **Engine**: iOS CoreMotion WebKit Heading + Android Absolute Magnetometer Sensor Fusion.
* **Capabilities**:
  * Determines real-time magnetic heading ($0^\circ - 360^\circ$) fused with the user's facing direction.
  * Uses simple, intuitive, non-confusing directions (**"Right"**, **"Left"**, **"Straight Ahead"**, **"Turn Around"**) coupled with true cardinal directions (**North, South, East, West**).
  * Eliminates confusing aviation terms like "veer" in favor of natural conversational guidance in both Hindi and English.

#### 4. Model 4: Millimeter Collision Risk & Predictive Optical Depth AI
* **Engine**: 3-Sector Ground-Plane Contact Raycaster + Perspective Pinhole Metric Estimator.
* **Capabilities**:
  * **Accurate Metric Distance**: Dynamically computes proximity for Left ($10\% - 30\%$), Center ($35\% - 65\%$), and Right ($70\% - 90\%$) sectors from $0.25\text{m}$ to $5.0\text{m}$.
  * **Elevated Object Sizing**: Overcomes horizon distortion for tabletop items (laptops, bottles, cups) using optical focal sizing ($Z = \frac{f \times W_{\text{real}}}{w_{\text{norm}}}$), accurately reporting $0.5\text{m} - 0.8\text{m}$ distance instead of false horizon defaults.
  * **Time-to-Collision (TTC)**: Continuously evaluates rate-of-approach ($TTC = \frac{D_{\text{center}}}{V_{\text{walking}}}$) in seconds, warning users before impact occurs.

#### 5. Master Sensor Fusion Arbiter
* Synchronizes all 4 models at 60 FPS in a unified inference loop.
* Implements a **Deterministic Preemptive Safety Hierarchy**:
  1. Camera Occlusion / Hand on Lens ($0.0\text{m}$ Stop Alert)
  2. Staircase / Pothole Emergency Hazard ($< 0.85\text{m}$)
  3. Natural Wall Imminent Collision ($< 0.55\text{m}$)
  4. Moving In-Path Obstacle (Pedestrian / Vehicle / Furniture Steering)
  5. Clear Walking Corridor Navigation & Step Progress

---

## 🎙️ Full Voice Control System

Drishti Saathi features an end-to-end, bilingual **Voice Control & Auditory Feedback Architecture** designed specifically for blind users:

### 1. Hands-Free & Tactile Interaction
- **Touch-Anywhere Push-To-Talk**: The entire screen serves as a single massive tactile target. Users can tap anywhere to issue voice instructions without searching for buttons.
- **Continuous Background Auto-Listening**: Automatically listens for spoken instructions while walking.
- **Soft Microphone Ducking**: Eliminates speech synthesis feedback loops by ducking microphone recognition while TTS speaks, without killing the underlying Web Speech API connection on mobile devices.

### 2. Supported Voice Commands (Bilingual)

| User Intent | English Command Examples | Hindi Commands (हिंदी) | Hinglish Transliterations |
| :--- | :--- | :--- | :--- |
| **Scene Description** | *"What's in front of me?"*, *"Describe scene"*, *"Look around"* | *"सामने क्या है?"*, *"क्या दिख रहा है?"*, *"दृश्य बताओ"* | `samne kya hai`, `kya dikh raha hai` |
| **Navigation Start** | *"Start navigation"*, *"Begin walking"* | *"नेविगेशन शुरू करो"*, *"चालू करो"* | `navigation shuru karo`, `start karo` |
| **Navigation Stop** | *"Stop navigation"*, *"Turn off"* | *"नेविगेशन बंद करो"*, *"रोको"* | `navigation band karo`, `roko` |
| **Path Clearance** | *"Is path clear?"*, *"Any obstacles ahead?"* | *"क्या रास्ता साफ है?"*, *"कोई रुकावट है?"* | `kya rasta saaf hai`, `koi rukawat` |
| **Object Finder** | *"Where is my laptop?"*, *"Where is my bottle?"*, *"Find bed"* | *"मेरा लैपटॉप कहाँ है?"*, *"बोतल कहाँ है?"*, *"बिस्तर ढूँढो"* | `laptop kahan hai`, `bottle dhundo` |
| **Wall Distance** | *"How far is the wall?"*, *"Distance to wall"* | *"दीवार कितनी दूर है?"*, *"दीवार बताओ"* | `deewar kitni door hai` |
| **Retrace Route** | *"Take me back"*, *"Retrace steps"*, *"Return to origin"* | *"वापस चलो"*, *"पीछे घूमो"*, *"शुरुआत में ले चलो"* | `vapas chalo`, `piche ghumo` |
| **Route Summary** | *"How far did I walk?"*, *"Route summary"* | *"कितना चले?"*, *"दूरी बताओ"*, *"रास्ता बताओ"* | `kitna chale`, `summary batao` |
| **Language Switch** | *"Speak Hindi"*, *"Switch to English"* | *"हिंदी में बोलो"*, *"अंग्रेजी में बोलो"* | `hindi me bolo`, `english me bolo` |

---

## 🗺️ Topological Breadcrumb Dead-Reckoning Engine ("स्मृति मार्ग")

Drishti Saathi incorporates a 100% offline **Dead-Reckoning Trajectory & Retrace Engine**:
* Uses device accelerometers to detect physical footfall impacts ($> 2.4\text{ m/s}^2$ at $> 360\text{ms}$ intervals).
* Plots a 2D topological vector map of the user's journey in local device memory.
* When the user commands **"वापस चलो" / "Take me back"**, the system inverts the turn sequence and provides turn-by-turn guidance back to the starting point.

---

## 🔔 Synthetic Web Audio Earcons

Zero pre-recorded MP3 files are used; all auditory cues are dynamically synthesized via the browser's native **Web Audio API**:
- **Listen Start (Rising Chime)**: $440\text{Hz} \to 660\text{Hz}$ tone confirming mic is listening.
- **Command Acknowledged (Bright Ping)**: $587\text{Hz} \to 880\text{Hz}$ confirming speech intent matched.
- **Steer Right Chord**: Dual $523\text{Hz} + 659\text{Hz}$ ascending harmony.
- **Steer Left Chord**: Dual $659\text{Hz} + 523\text{Hz}$ descending harmony.
- **Critical Hazard Siren**: High-frequency emergency alarm ($1046\text{Hz} \to 1318\text{Hz}$) paired with continuous haptic vibration.
- **Path Clear Chime**: Gentle major-third chord signaling path is open.

---

## 🔒 Privacy & Zero-Data-Egress Guarantee

1. **100% On-Device Inference**: All camera frames, speech recognition transcripts, and sensor measurements are processed exclusively in RAM.
2. **Zero Cloud Ingestion**: No external cloud APIs, tracking beacons, analytics scripts, or remote servers.
3. **Immediate Buffer Purge**: Every processed camera frame is immediately overwritten in canvas memory; zero images are saved to disk or storage.

---

## 🛠️ Quickstart

### Prerequisites
- Modern browser supporting WebGL/WebGPU, Web Audio API, and Web Speech API (Google Chrome, Microsoft Edge, Safari).
- Works directly on mobile smartphones (Android / iOS) and laptops/desktops.

### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/yashverma1611/drishti-saathi.git
   cd drishti-saathi
   ```
2. Serve the directory with any local HTTP server (required for camera & microphone permissions):
   ```bash
   # Using Python 3
   python -m http.server 8000
   
   # Or using Node npx
   npx serve .
   ```
3. Open `http://localhost:8000` (or `http://<your-pc-ip>:8000` on your mobile phone connected to the same Wi-Fi).
4. Tap anywhere on the screen to initialize audio, camera, and voice guidance.

---

## 🏆 Hackathon Multi-Model Compliance Checklist

| Feature Requirement | Drishti Saathi Implementation | Status |
| :--- | :--- | :---: |
| **Model 1: Object Recognition** | TensorFlow.js COCO-SSD (80 classes) + Teachable Vision engine | ✅ Verified |
| **Model 2: Stairs / Potholes** | Dual-pass gradient edge horizon analysis + ground depression void detector | ✅ Verified |
| **Model 3: Direction Guidance** | Magnetometer true North compass fusion + everyday 4-quadrant turn instructions | ✅ Verified |
| **Model 4: Collision & Distance** | 3-Sector ground raycasting + optical metric pinhole estimator + Time-to-Collision (TTC) | ✅ Verified |
| **Master Fusion Arbiter** | Real-time multi-model synchronization with preemptive safety hierarchy | ✅ Verified |
| **Hands-Free Bilingual Voice** | Full Hindi & English Web Speech voice navigation with soft mic ducking | ✅ Verified |
| **Mobile Form Factor** | Responsive HUD, touch-anywhere push-to-talk, iOS/Android sensor permissions | ✅ Verified |

---

## ⚠️ Assistive Disclaimer
*Drishti Saathi is an assistive navigation prototype designed to supplement, not replace, traditional mobility aids such as white canes or trained guide dogs. Users should always exercise personal caution and use physical mobility aids in unfamiliar or hazardous environments.*
