import os
import sys
import argparse
import time
import cv2
import pandas as pd
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.detection.detector import VehicleDetector
from src.detection.tracker import VehicleTracker
from src.detection.counter import VehicleCounter

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

def process_image(input_path, output_dir, display_conf=0.35, device=None):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    annotated_filename = f"annotated_{filename}"
    annotated_path = os.path.join(output_dir, annotated_filename)
    csv_path = os.path.join(output_dir, "image_counts.csv")

    detector = VehicleDetector(conf_threshold=0.15, device=device)
    detections, _ = detector.detect(input_path)

    # Filter for display and counting
    valid_dets = [d for d in detections if d['confidence'] >= display_conf]

    counts = {'car': 0, 'bus': 0, 'motorcycle': 0, 'truck': 0}
    for d in valid_dets:
        cls_name = d['class_name']
        if cls_name in counts:
            counts[cls_name] += 1
    counts['total'] = sum(counts.values())

    # Read image for annotation
    img = cv2.imread(input_path)
    if img is not None:
        for d in valid_dets:
            x1, y1, x2, y2 = [int(v) for v in d['box']]
            label = f"{d['class_name']} {d['confidence']:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Overlay summary counts box
        y_offset = 30
        cv2.rectangle(img, (10, 10), (220, 130), (0, 0, 0), -1)
        cv2.putText(img, "Detections Summary", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for k in ['car', 'bus', 'motorcycle', 'truck']:
            cv2.putText(img, f"{k}: {counts[k]}", (15, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_offset += 18

        cv2.imwrite(annotated_path, img)

    # Save CSV
    df_counts = pd.DataFrame([counts])
    df_counts.to_csv(csv_path, index=False)

    print("\n==========================================")
    print(f"IMAGE PROCESSING COMPLETE")
    print(f"Input: {input_path}")
    print(f"Device Used: {detector.device}")
    print(f"Display Threshold: {display_conf}")
    print(f"Annotated Image Saved: {annotated_path}")
    print(f"Counts CSV Saved: {csv_path}")
    print(f"Detections Summary: {counts}")
    print("==========================================\n")
    return counts

def process_video(input_path, output_dir, display_conf=0.35, device=None):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    annotated_filename = f"annotated_{filename}"
    annotated_path = os.path.join(output_dir, annotated_filename)
    log_path = os.path.join(output_dir, "tracking_log.csv")

    tracker = VehicleTracker(conf_threshold=0.15, device=device)
    counter = VehicleCounter(display_conf=display_conf)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video at {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

    print("\n==========================================")
    print(f"STARTING VIDEO PROCESSING & TRACKING")
    print(f"Input: {input_path}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} fps")
    print(f"Total Frames: {total_frames}")
    print(f"Device Used: {tracker.device}")
    print(f"Tracker Conf Supply: 0.15 | Display Threshold: {display_conf}")
    print("==========================================\n")

    frame_idx = 0
    tracking_logs = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        timestamp_sec = round(frame_idx / fps, 2)

        # ByteTrack update
        tracked_objects, _ = tracker.track(frame)

        # Counter update
        active_counts, active_ids = counter.update(tracked_objects)
        cum_counts = counter.get_counts()

        # Format active IDs string for log
        active_ids_str = "; ".join([f"{cls_name}#{tid}" for cls_name, tid in active_ids])

        # Record frame log
        log_entry = {
            'frame': frame_idx,
            'timestamp_sec': timestamp_sec,
            'active_car': active_counts['car'],
            'active_bus': active_counts['bus'],
            'active_motorcycle': active_counts['motorcycle'],
            'active_truck': active_counts['truck'],
            'active_track_ids': active_ids_str,
            'cum_unique_car': cum_counts['car'],
            'cum_unique_bus': cum_counts['bus'],
            'cum_unique_motorcycle': cum_counts['motorcycle'],
            'cum_unique_truck': cum_counts['truck'],
            'cum_total_unique': cum_counts['total_unique']
        }
        tracking_logs.append(log_entry)

        # Annotate frame
        for obj in tracked_objects:
            if obj['confidence'] >= display_conf:
                x1, y1, x2, y2 = [int(v) for v in obj['box']]
                tid = obj.get('track_id')
                tid_str = f"#{tid}" if tid is not None else ""
                label = f"{obj['class_name']}{tid_str} {obj['confidence']:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Overlay running statistics box
        cv2.rectangle(frame, (10, 10), (240, 135), (0, 0, 0), -1)
        cv2.putText(frame, f"Unique Counts (F:{frame_idx})", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y_off = 30
        for cls_name in ['car', 'bus', 'motorcycle', 'truck']:
            cnt = cum_counts[cls_name]
            cv2.putText(frame, f"{cls_name}: {cnt}", (15, y_off + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_off += 16
        cv2.putText(frame, f"Total Unique: {cum_counts['total_unique']}", (15, y_off + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        out.write(frame)

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            print(f"Processed Frame [{frame_idx:03d}/{total_frames:03d}] ({elapsed:.1f}s) -> Cumulative Unique Vehicles: {cum_counts['total_unique']}")

    cap.release()
    out.release()

    # Save log CSV
    df_logs = pd.DataFrame(tracking_logs)
    df_logs.to_csv(log_path, index=False)

    print("\n==========================================")
    print(f"VIDEO PROCESSING COMPLETE")
    print(f"Annotated Video Saved: {annotated_path}")
    print(f"Tracking Log Saved: {log_path}")
    print(f"FINAL UNIQUE VEHICLE COUNTS: {counter.get_counts()}")
    print("==========================================\n")
    return counter.get_counts()

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Vehicle Detection, Tracking, and Counting Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Path to input image or video file")
    parser.add_argument('--output', type=str, default='outputs/', help="Directory to save output artifacts")
    parser.add_argument('--conf', type=float, default=0.35, help="Display/counting confidence threshold (default: 0.35)")
    args = parser.parse_args()

    ext = os.path.splitext(args.input)[1].lower()
    if ext in IMAGE_EXTENSIONS or args.input.startswith('http'):
        process_image(args.input, args.output, display_conf=args.conf)
    elif ext in VIDEO_EXTENSIONS:
        process_video(args.input, args.output, display_conf=args.conf)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported image: {IMAGE_EXTENSIONS}, video: {VIDEO_EXTENSIONS}")

if __name__ == '__main__':
    main()
