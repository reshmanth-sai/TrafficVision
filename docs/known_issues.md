# Known Issues & Technical Challenges (`traffic-cv`)

This document records key technical challenges, tracking edge cases, and design decisions encountered during development.

---

## 1. Class-Agnostic Non-Maximum Suppression (`agnostic_nms=True`)

### Problem & Diagnostic Findings
During Stage 1 video tracking verification (`data/sample_traffic_multi.mp4`), a track fragmentation / double-counting bug was identified around frames 555–566.
- **Symptom**: A single physical SUV/truck moving down the road emitted `car#4` at frame 557, `truck#4` at frame 558, `car#4; truck#5` at frame 559, and `truck#5` / `car#5` thereafter. Cumulative unique truck count artificially incremented from 1 to 2.
- **Root Cause**: By default, Ultralytics YOLO NMS runs per-class (`agnostic_nms=False`). For visually ambiguous distant vehicles, YOLO emitted two overlapping bounding boxes for the exact same physical vehicle (one as `car` with conf=0.43 and one as `truck` with conf=0.42). Because ByteTrack tracks objects per class category, receiving dual overlapping detections caused it to instantiate two separate track IDs (`#4` and `#5`).

### Solution Implemented
Set `agnostic_nms=True` permanently in both `VehicleDetector` ([`src/detection/detector.py`](file:///Users/sai/Desktop/traffic-cv/src/detection/detector.py)) and `VehicleTracker` ([`src/detection/tracker.py`](file:///Users/sai/Desktop/traffic-cv/src/detection/tracker.py)).
- **Effect**: NMS compares overlapping candidate boxes across all vehicle classes, keeping only the single highest-confidence box per physical vehicle.
- **Result**: Single physical vehicle maintained persistent identity `track_id = 1` continuously across frames 555–566 without ID switching or duplicate counts.

---

## 2. Per-Class Track Double Counting & Majority-Vote Resolution

### Problem Identified
In `VehicleCounter` ([`src/detection/counter.py`](file:///Users/sai/Desktop/traffic-cv/src/detection/counter.py)), unique vehicles were initially tracked using separate `set()` collections per class (`self.seen_track_ids['car']`, `self.seen_track_ids['truck']`).
When a vehicle (e.g. `track_id = 4`) had class predictions flip between `car` (57 frames) and `truck` (3 frames) over its 60-frame trajectory:
- `track_id = 4` was added to `seen_track_ids['car']` AND `seen_track_ids['truck']`.
- **Symptom**: `counts['car'] = 2`, `counts['truck'] = 1`, and `total_unique = 3` on footage containing only 2 physical vehicles (`data/sample_traffic_multi.mp4`).

### Fix Implemented
Updated `VehicleCounter` to maintain a single global trajectory class history `track_class_history[track_id] = [class_predictions...]`.
- **Majority-Vote Assignment**: `get_counts()` determines the single majority-vote class for each `track_id` (`Counter(history).most_common(1)[0][0]`).
- **Exclusive Counting**: Each persistent `track_id` is assigned EXCLUSIVELY to its majority-vote class and contributes EXACTLY ONCE to `total_unique`.
- **Result on `sample_traffic_multi.mp4`**:
  - `Track #01`: `car` (20/20 frames) -> `car`
  - `Track #04`: `car` (57/60 frames) -> `car`
  - **Final Counts**: `{'car': 2, 'truck': 0, 'total_unique': 2}` (Perfect match with 2 physical vehicles).

---

## 3. Aerial Top-Down View Domain Mismatch (`sample_traffic.mp4`)

### Diagnostic Findings & File Identity
Full verification confirmed [`data/sample_traffic.mp4`](file:///Users/sai/Desktop/traffic-cv/data/sample_traffic.mp4) is bit-for-bit identical to the official Intel IoT DevKit `car-detection.mp4` source (`MD5: e919de1193da5ceb8b0fd3cd998c2694`, 768×432 @ 12.5 FPS, 377 frames), depicting an aerial top-down parking lot view. Manual visual review identified 4 distinct physical vehicles appearing across the 30-second duration:
1. **Vehicle 1 (White Hatchback, ~0:04–0:09)**: Moving North up center aisle.
2. **Vehicle 2 (Red Sedan, ~0:15–0:20)**: Moving North up right-center aisle.
3. **Vehicle 3 (Silver/Grey Sedan, ~0:14–0:18)**: Moving South down left aisle (`car#5` in `tracking_log.csv`).
4. **Vehicle 4 (White Sedan, ~0:25–0:28)**: Moving North up left aisle (`car#7` in `tracking_log.csv`).

### Root Cause: COCO Class Alias Effect & Domain Mismatch
- **Domain Mismatch**: Standard COCO-pretrained YOLO models (e.g., `yolo11n.pt`) are trained almost exclusively on eye-level and street-level traffic camera angles. From a directly overhead aerial perspective, rectangular car roofs and glass windshields lack front-grille, headlight, side-profile, and wheel features.
- **Class Aliasing**: Rectangular overhead car bodies strongly match the feature distribution of COCO's `cell phone` class. Unconstrained low-confidence inference (`conf = 0.01` to `0.15`) revealed overhead vehicle bounding boxes receiving up to **0.962 confidence** as `cell phone`:
  - **Vehicle 1 (White Hatchback)**: Tracked as `cell phone` (frames 73–102, max conf **0.962**).
  - **Vehicle 2 (Red Sedan)**: Tracked as `cell phone` / `bus` (frames 198–234, max conf **0.946**).
  - **Vehicle 3 (Silver/Grey Car)**: Tracked as `cell phone` / `bottle` (frames 186–223, max conf **0.925**).
- **Track Fragmentation**: Because `VehicleCounter` ignores non-vehicle classes, valid `car` class confidence fluctuates between 0.05 and 0.40. Vehicle 1 emitted `car` confidence > 0.35 on only 2 isolated frames (F66 conf 0.40, F104 conf 0.38), failing ByteTrack's track initialization criteria (`min_hits`), leading to omission from cumulative vehicle counts under standard thresholds (`display_conf = 0.35`).

### Production Design Choice: Retain `display_conf = 0.35`
- The production `display_conf = 0.35` threshold remains unchanged. The primary target use case for `traffic-cv` is street-level traffic monitoring (e.g., `sample_traffic_multi.mp4`), where `display_conf = 0.35` is validated and performs cleanly.
- Lowering the global detection threshold risks reintroducing false positives or noisy tracks on validated street-level footage.

### Future Improvements & Architectural Fix Paths
1. **Fine-Tuning on Aerial Vehicle Datasets**: Fine-tune YOLO weights on specialized top-down / aerial vehicle datasets (such as VisDrone, VEDAI, or UAVDT) to establish robust overhead vehicle feature representations.
2. **Scene-Adaptive Camera Profiles**: Implement camera perspective profiles (e.g., `camera_profile='overhead'`) that dynamically include top-down class aliases or adjust detection thresholds specifically for overhead camera feeds.

