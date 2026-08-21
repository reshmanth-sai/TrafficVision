import os
import sys
import argparse
import time
from collections import Counter, deque
import cv2
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.detection.tracker import VehicleTracker
from src.violations.emergency_heuristic import EmergencyVehicleHeuristic
from src.violations.lane_intrusion import LaneIntrusionDetector

def process_violations(video_path, output_dir, display_conf=0.35, expected_dir=(0.0, -1.0), device=None):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(video_path)
    annotated_filename = f"annotated_violations_{filename}"
    annotated_path = os.path.join(output_dir, annotated_filename)
    csv_path = os.path.join(output_dir, "violations_log.csv")

    tracker = VehicleTracker(conf_threshold=0.15, device=device)
    emergency_detector = EmergencyVehicleHeuristic(buffer_size=30)
    lane_detector = LaneIntrusionDetector(expected_dir=expected_dir, cosine_thresh=-0.5, persistence=5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 12.5
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

    print("\n==========================================")
    print("STARTING TRAFFIC VIOLATION ENGINE")
    print(f"Input Video: {video_path}")
    print(f"Specs: {width}x{height} @ {fps:.1f} fps ({total_frames} frames)")
    print(f"Expected Lane Vector: {expected_dir}")
    print(f"Device Used: {tracker.device}")
    print("==========================================\n")

    # Track class history for majority-vote class mapping per track_id
    track_class_history = {}
    violations_log = []
    frame_idx = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        timestamp_sec = round(frame_idx / fps, 2)

        # ByteTrack update
        tracked_objects, _ = tracker.track(frame)

        active_violations_in_frame = 0

        for obj in tracked_objects:
            conf = obj['confidence']
            tid = obj.get('track_id')
            box = obj['box']

            if tid is None:
                continue

            # Update class history for majority voting
            if tid not in track_class_history:
                track_class_history[tid] = []
            track_class_history[tid].append(obj['class_name'])

            # Compute majority-vote class
            majority_class = Counter(track_class_history[tid]).most_common(1)[0][0]

            # Update violation modules with frame data
            emergency_detector.update_track(tid, frame, box)
            lane_detector.update_track(tid, box)

            # Evaluate heuristics only for objects above display threshold
            if conf >= display_conf:
                is_emerg, freq_hz, amp = emergency_detector.is_emergency(tid, fps=fps)
                is_lane_v, cos_sim, p_count = lane_detector.check_violation(tid)

                box_color = (0, 255, 0) # Green for normal vehicles
                label_prefix = f"{majority_class}#{tid} ({conf:.2f})"

                if is_emerg:
                    active_violations_in_frame += 1
                    box_color = (255, 255, 0) # Cyan/Amber for Emergency Vehicle
                    label_prefix = f"[EMERGENCY] {majority_class}#{tid} ({freq_hz}Hz)"
                    violations_log.append({
                        'frame': frame_idx,
                        'timestamp_sec': timestamp_sec,
                        'track_id': tid,
                        'majority_class': majority_class,
                        'violation_type': 'EMERGENCY_VEHICLE',
                        'evidence_details': f"Flash Freq: {freq_hz} Hz, Amp: {amp}"
                    })

                if is_lane_v:
                    active_violations_in_frame += 1
                    box_color = (0, 0, 255) # Red for Wrong-Way Lane Intrusion
                    label_prefix = f"[WRONG-WAY] {majority_class}#{tid} (cos={cos_sim})"
                    violations_log.append({
                        'frame': frame_idx,
                        'timestamp_sec': timestamp_sec,
                        'track_id': tid,
                        'majority_class': majority_class,
                        'violation_type': 'LANE_INTRUSION_WRONG_WAY',
                        'evidence_details': f"Cos Sim: {cos_sim}, Persist: {p_count} frames"
                    })

                # Draw annotated bounding box
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label_prefix, (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

        # Overlay violation summary box
        cv2.rectangle(frame, (10, 10), (260, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames} ({timestamp_sec:.1f}s)", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Total Violation Records: {len(violations_log)}", (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        out.write(frame)

        if frame_idx % 100 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            print(f"Processed Frame [{frame_idx:03d}/{total_frames:03d}] ({elapsed:.1f}s) -> Total Violation Logs: {len(violations_log)}")

    cap.release()
    out.release()

    # Save CSV
    df_v = pd.DataFrame(violations_log)
    df_v.to_csv(csv_path, index=False)

    print("\n==========================================")
    print("VIOLATION PROCESSING COMPLETE")
    print(f"Annotated Video Saved: {annotated_path}")
    print(f"Violations Log Saved: {csv_path}")
    print(f"TOTAL VIOLATION RECORDS FLAGGED: {len(violations_log)}")
    print("==========================================\n")
    return df_v

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Traffic Violation Detection Engine")
    parser.add_argument('--video', type=str, required=True, help="Path to input video file")
    parser.add_argument('--output', type=str, default='outputs/', help="Directory to save output artifacts")
    parser.add_argument('--conf', type=float, default=0.35, help="Display confidence threshold")
    parser.add_argument('--dir_x', type=float, default=0.0, help="Expected lane travel vector X component")
    parser.add_argument('--dir_y', type=float, default=-1.0, help="Expected lane travel vector Y component")
    args = parser.parse_args()

    process_violations(args.video, args.output, display_conf=args.conf, expected_dir=(args.dir_x, args.dir_y))

if __name__ == '__main__':
    main()
