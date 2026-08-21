# TrafficVision 🚘📹

**TrafficVision** is an end-to-end Computer Vision & Deep Learning pipeline designed for real-time traffic monitoring, vehicle detection, persistent trajectory tracking, counting, and traffic violation analytics.

---

## 🌟 Key Features

- **Multi-Class Vehicle Detection**: Detects cars, trucks, buses, and motorcycles using pretrained YOLO11 models.
- **Class-Agnostic NMS (`agnostic_nms=True`)**: Resolves overlapping multi-class candidate bounding boxes for distant or visually ambiguous vehicles.
- **Persistent Trajectory Tracking**: Tracks vehicle movement across video streams using **ByteTrack** with motion and appearance consistency.
- **Trajectory Majority-Vote Class Resolution**: Prevents per-class double-counting by maintaining global track histories and assigning each unique vehicle exclusively to its majority-voted class.
- **Traffic Analytics & Violations**: Detects lane intrusions, emergency vehicle heuristics, and records tracking & violation event logs (`outputs/tracking_log.csv`).
- **Interactive Dashboard**: Streamlit-powered Web UI for real-time video playback, interactive data filters, and visual counts breakdown.

---

## 📁 Repository Structure

```
TrafficVision/
├── src/
│   ├── detection/
│   │   ├── detector.py          # YOLO11 Vehicle Detector with Class-Agnostic NMS
│   │   ├── tracker.py           # ByteTrack Object Tracker
│   │   ├── counter.py           # Unique Vehicle Counter with Majority-Vote Logic
│   │   └── run_pipeline.py      # Main Pipeline Runner
│   ├── violations/
│   │   ├── lane_intrusion.py    # Lane Intrusion & Zone Monitoring
│   │   └── emergency_heuristic.py# Emergency Vehicle Detection
│   ├── classification/         # Vehicle Classification Models & Training
│   └── dashboard/
│       └── app.py               # Streamlit Dashboard UI
├── docs/
│   └── known_issues.md          # Technical Challenges & Empirical Diagnostic Findings
├── data/                        # Sample Traffic Video Feeds & Dataset Placeholders
└── outputs/                     # Output Logs & Annotated Video Files
```

---

## 🛠️ Quick Start

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/ReshmanthSai/TrafficVision.git
cd TrafficVision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Pipeline & Analytics

Run detection, tracking, and counter processing on a sample traffic video feed:

```bash
python -m src.detection.run_pipeline --video data/sample_traffic_multi.mp4
```

### 3. Launch Dashboard

Launch the interactive Streamlit web dashboard:

```bash
streamlit run src/dashboard/app.py
```

---

## 💡 Technical Challenges & Solutions

Detailed diagnostic analysis and empirical findings are documented in [`docs/known_issues.md`](docs/known_issues.md):

1. **Class-Agnostic NMS (`agnostic_nms=True`)**: Default per-class NMS caused visually ambiguous distant vehicles to emit dual overlapping boxes (`car` + `truck`), fragmenting tracks into multiple IDs. Enabling class-agnostic NMS ensures a single highest-confidence box per physical vehicle.
2. **Trajectory Majority-Vote Resolution**: Trajectory class predictions flipping between classes (e.g. 57 frames `car` / 3 frames `truck`) previously double-counted vehicles across multiple class counters. Implemented trajectory majority voting to assign each persistent `track_id` exclusively to a single class.
3. **Aerial Top-Down View Domain Mismatch**: Identified COCO model class aliasing on directly overhead parking-lot camera angles (`sample_traffic.mp4`), where rectangular vehicle roofs received up to **0.962 confidence** as `cell phone`. Retained standard `0.35` threshold for validated street-level cameras and documented overhead fine-tuning as a future improvement.

---

## 🚀 Future Roadmap

- [ ] Fine-tune YOLO models on dedicated overhead / drone traffic datasets (VisDrone, UAVDT).
- [ ] Add camera perspective presets (`camera_profile='street'` vs `'overhead'`).
- [ ] Speed estimation & license plate recognition (ALPR) integration.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.
