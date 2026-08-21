# TrafficVision — AI Traffic Police (Computer Vision Track)

**MIC AIML Department Recruitment Challenge — Round 2 Submission**
Naidu Reshmanth Sai | 2nd Year, CSE Core | VIT Chennai
Repository: [github.com/reshmanth-sai/TrafficVision](https://github.com/reshmanth-sai/TrafficVision)

---

## Project Overview

TrafficVision is a two-stage computer vision system for traffic monitoring, built for the **Computer Vision track (AI Traffic Police)** of the MIC AIML recruitment challenge. It detects, tracks, and counts vehicles in traffic footage, classifies scene-level vehicle composition using a custom-trained model on the mandatory dataset, and flags two categories of traffic violations: candidate emergency vehicles and wrong-way / lane-intrusion driving.

The system is exposed through both a CLI pipeline and an interactive Streamlit dashboard, and satisfies the 2nd-year requirement of **Part 1 (classification + counting) + Part 2 (emergency vehicle detection + violation detection)**.

## Problem Statement

Given traffic video or image input, the system must:
1. Classify vehicle types present (car, bus, motorcycle, truck) and count unique vehicles — **Part 1**
2. Detect candidate emergency vehicles and basic traffic violations — **Part 2**

A key early finding shaped this project's architecture: the mandatory dataset (`Vehicles-coco`) is annotated in **scene-level multi-label classification format**, not object-detection format — there are no bounding boxes, and a single image can carry multiple positive vehicle-type labels simultaneously (35%+ of images have 2 or more classes marked). This meant a single-model, crop-and-classify approach was not directly supported by the data as provided, and shaped the two-stage design described below.

## Installation Instructions

```bash
git clone https://github.com/reshmanth-sai/TrafficVision.git
cd TrafficVision

python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Run the CLI detection + tracking + counting pipeline:
```bash
python src/detection/run_pipeline.py --input data/sample_traffic.mp4 --output outputs/
```

Run the violation detection engine:
```bash
python src/violations/run_violations.py --video data/sample_traffic.mp4 --output outputs/
```

Launch the interactive dashboard:
```bash
streamlit run src/dashboard/app.py
```
Then open `http://localhost:8501` in your browser.

**Hardware note:** the pipeline uses Apple Silicon MPS acceleration where available (`torch.device('mps')`), confirmed working during development on a MacBook Pro M-series (16GB). On other hardware it will fall back to CPU/CUDA automatically via PyTorch's standard device resolution.

## Dataset Used

**Primary (mandatory) dataset:** [Vehicles-coco](https://universe.roboflow.com/vehicle-mscoco/vehicles-coco) (Roboflow, CC BY 4.0)
- 18,998 images total — 13,300 train / 3,799 valid / 1,901 test (post-verification counts)
- Multi-label scene classification format: `bus`, `car`, `motorcycle`, `truck`
- Class distribution (train split): car 67.5%, truck 33.5%, bus 21.9%, motorcycle 19.5% — used to compute `pos_weight` for weighted BCE loss

**Secondary dataset:** [Vehicles](https://universe.roboflow.com/exjobb-dq06p/vehicles-k83q3) (Roboflow, CC BY 4.0)
- 4,549 images (4,312 train / 152 valid / 88 test)
- Same 4 classes, plus a residual `"0"` artifact column (0.14% positive rate) identified as a Roboflow export artifact and excluded from training
- Used as a secondary reference; not the primary training set given its smaller, noisier split sizes

**Sample test footage:** Intel IoT DevKit sample videos (`car-detection.mp4`, `person-bicycle-car-detection.mp4`) — public sample traffic footage used for pipeline validation, tracking verification, and dashboard demonstration.

## Methodology

### Two-stage architecture

**Stage 1 — Detection, Tracking, Counting (Part 1)**
Uses COCO-pretrained YOLO11n restricted to native COCO vehicle classes (`car`, `truck`, `bus`, `motorcycle`), integrated with ByteTrack for persistent multi-frame tracking, producing unique per-class vehicle counts (not per-frame detection counts). No fine-tuning was needed for this stage since the target classes are natively present in COCO's training distribution.

**Stage 2 — Scene-Level Multi-Label Classifier (mandatory dataset compliance)**
Since the mandatory dataset provides scene-level, not per-instance, labels, Stage 2 was deliberately scoped as an independent scene composition classifier ("which vehicle types are present in this image") rather than a per-crop classifier. This is a legitimate architectural response to the data's actual structure rather than a workaround — training a per-crop classifier on labels that were never assigned per-instance would have produced a model with a mismatched training signal. Two ResNet18 variants were trained and compared:
- **Fine-tuned**: ImageNet-pretrained backbone, initially frozen (2 epochs) then unfrozen with discriminative learning rates (backbone 1e-4, head 1e-3)
- **From-scratch**: identical architecture, random initialization, given ~2.5x the epoch budget to account for slower convergence

Both use `BCEWithLogitsLoss` with `pos_weight` computed directly from the training split's class distribution to counter class imbalance (motorcycle and bus being the minority classes).

**Part 2 — Violation Detection**
- **Emergency vehicle heuristic**: rather than training a classifier (no labeled emergency-vehicle data was available in the timeframe), a signal-processing heuristic detects flashing light-bar signatures. The top 30% of each tracked vehicle's bounding box is monitored for oscillating red/blue channel intensity, smoothed via a moving average, with zero-crossing frequency checked against **1.0–4.0 Hz**, the flash-rate range specified under SAE J595 for emergency vehicle warning devices.
- **Lane intrusion / wrong-way detection**: centroid trajectory direction per tracked vehicle is compared against a configured expected lane direction via cosine similarity, with a violation only flagged when the vehicle is >120° opposed (`cosine < -0.5`) and this persists for 5+ consecutive frames — avoiding false positives from ordinary turns, lane changes, or tracking jitter.

## Technologies Used

- **Detection & Tracking**: Ultralytics YOLO11n (COCO-pretrained), ByteTrack (custom-tuned config)
- **Classification**: PyTorch, torchvision ResNet18
- **Dashboard**: Streamlit, imageio-ffmpeg (H.264 re-encoding for browser playback)
- **Data handling**: pandas, NumPy, OpenCV
- **Evaluation**: scikit-learn (F1, precision/recall, Hamming loss)
- **Hardware acceleration**: PyTorch MPS backend (Apple Silicon)

## Results

### Stage 2 Classifier — Fine-tuned vs. From-scratch (validation split, threshold = 0.4)

| Class | Fine-tuned F1 | From-scratch F1 | Relative gain |
|---|---|---|---|
| car | 0.824 | 0.816 | +1.0% |
| motorcycle | 0.783 | 0.639 | +22.5% |
| bus | 0.727 | 0.574 | +26.7% |
| truck | 0.590 | 0.526 | +12.2% |
| **Macro F1** | **0.7307** | **0.6387** | **+14.4%** |

- Fine-tuning was also **2.27x faster** in total wall-clock training time (9.0 min vs 20.4 min).
- Pretrained ImageNet features gave the largest relative benefit to the minority classes (bus, motorcycle) — consistent with the expectation that transfer learning compensates most where task-specific data is scarcest.
- `truck` was the weakest-performing class on both models, showing a precision/recall imbalance (high recall, lower precision) likely from visual confusion with large vans/buses at scene level — noted honestly here rather than only reporting favorable numbers.

### Violation Detection — Validation

- **False-positive testing**: both emergency-vehicle and lane-intrusion detectors produced **zero false flags** across 1,024 combined frames of normal traffic footage.
- **True-positive testing**: since no real emergency-vehicle or wrong-way footage was available, both heuristics were validated against synthetic controlled signals (a clean 2.0 Hz alternating red/blue pattern; a straight sustained wrong-direction trajectory). Both correctly triggered — the emergency heuristic recovered the frequency to within 0.01 Hz, and the lane-intrusion detector correctly gated on the 5-frame persistence requirement before flagging. Real-world robustness against partial occlusion, non-ideal viewing angles, and ambient lighting remains untested pending suitable footage — stated here explicitly rather than implied to be resolved.

## Challenges Faced

**1. Class-agnostic NMS bug causing vehicle double-counting.** Investigation of a crossing-path test case revealed that per-class NMS was allowing YOLO to emit duplicate overlapping bounding boxes for a single physical vehicle under two different class labels (e.g. `car` and `truck` simultaneously) for visually ambiguous vehicles like SUVs/pickups. This caused ByteTrack to assign two separate track IDs to one real vehicle. Root cause was confirmed via raw bounding-box coordinate comparison, and resolved by enabling `agnostic_nms=True`, verified via before/after tracking log comparison showing stable single-track continuity across the previously-fragmented sequence.

**2. Per-class counter double-counting from label instability.** Even after the above fix, a single track's *class label* could still flip frame-to-frame (e.g. a track spending 57/60 frames as `car` and 3/60 as `truck`). The original counter tracked "seen track IDs" per class independently, so this one vehicle was counted as both a unique car and a unique truck. Fixed by switching to majority-vote class assignment per track ID, guaranteeing each physical vehicle contributes to exactly one class's count.

**3. Domain mismatch on aerial/top-down footage.** Stress-testing on an aerial parking-lot clip (outside the challenge's primary street-level use case) revealed that COCO-pretrained YOLO's vehicle-class confidence competes with visually similar classes — most notably `cell phone` — when viewing vehicles from directly overhead, since COCO's training distribution contains essentially no aerial vehicle imagery. In one case, a vehicle's roof/windshield was classified as `cell phone` with 0.962 confidence. This is documented as a known limitation rather than patched via threshold-lowering, since doing so would risk destabilizing false-positive rates already validated on the project's actual target footage (street-level traffic).

**4. Confidence threshold conflict with ByteTrack's tiered matching.** An initial configuration applied the same confidence threshold to both detection supply and display/counting, which starved ByteTrack's low-confidence recovery tier of the detections it needs for occlusion handling. Resolved by supplying detections to the tracker at a low threshold (0.15) while applying a separate, stricter threshold (0.35 default) only for display and counting downstream.

## Future Improvements

- Fine-tune YOLO on aerial/top-down vehicle datasets (e.g. VisDrone, VEDAI, UAVDT) to resolve the documented domain-mismatch limitation for overhead camera feeds
- Replace the emergency-vehicle heuristic with a properly trained classifier once labeled emergency-vehicle footage is available
- Extend violation detection with signal-state-aware signal-jumping detection (requires traffic-light state annotation, unavailable in current footage)
- Add re-identification (appearance-based) to ByteTrack to handle vehicles that briefly leave and re-enter frame, currently treated as new tracks
- Multi-camera support and traffic density heatmaps (stretch goals)

## Screenshots

See `docs/screenshots/` for dashboard views, sample annotated frames, and training curves.

---

*Note: all reported metrics, training logs, and bug investigations in this README are drawn directly from actual run outputs and logs (see `docs/known_issues.md`, `docs/training_curves.csv`) rather than estimated.*
