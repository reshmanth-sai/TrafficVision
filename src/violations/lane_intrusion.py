import math
import numpy as np
from collections import deque

TRAJECTORY_WINDOW = 10     # Number of frames to compute movement velocity vector
COSINE_THRESHOLD = -0.5    # Tightened angle threshold: cosine < -0.5 (> 120° opposed travel)
PERSISTENCE_FRAMES = 5     # Minimum consecutive frames of opposed travel required to flag violation
MIN_MOVEMENT_PIXELS = 12.0 # Minimum total displacement (pixels) over window to avoid static jitter

class LaneIntrusionDetector:
    """
    Detects wrong-way driving and lane intrusion violations by comparing vehicle 
    centroid trajectory velocity vectors against expected lane direction vectors.
    """
    def __init__(self, expected_dir=(0.0, -1.0), cosine_thresh=COSINE_THRESHOLD, persistence=PERSISTENCE_FRAMES):
        """
        Args:
            expected_dir (tuple): Expected normalized (dx, dy) travel direction vector for the lane.
            cosine_thresh (float): Cosine similarity threshold (default: -0.5, >120° opposed).
            persistence (int): Minimum consecutive frames required (default: 5).
        """
        # Normalize expected direction vector
        norm = math.hypot(expected_dir[0], expected_dir[1])
        if norm > 0:
            self.expected_dir = (expected_dir[0] / norm, expected_dir[1] / norm)
        else:
            self.expected_dir = (0.0, -1.0)

        self.cosine_thresh = cosine_thresh
        self.persistence = persistence

        # Trajectory centroid history per track_id: deque of (x_centroid, y_centroid)
        self.centroid_histories = {}
        # Consecutive violation counter per track_id
        self.violation_counters = {}

    def update_track(self, track_id, box):
        """
        Records bounding box centroid for trajectory calculation.
        
        Args:
            track_id (int): Persistent ByteTrack ID.
            box (list): Bounding box [x1, y1, x2, y2].
        """
        if track_id not in self.centroid_histories:
            self.centroid_histories[track_id] = deque(maxlen=TRAJECTORY_WINDOW)
            self.violation_counters[track_id] = 0

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.centroid_histories[track_id].append((cx, cy))

    def check_violation(self, track_id):
        """
        Evaluates track velocity vector against expected lane direction vector.
        
        Returns:
            is_violation (bool): True if vehicle is travelling in wrong direction for >= 5 consecutive frames.
            cos_sim (float): Cosine similarity between velocity vector and expected lane vector.
            consecutive_frames (int): Current count of consecutive violation frames.
        """
        if track_id not in self.centroid_histories:
            return False, 1.0, 0

        history = list(self.centroid_histories[track_id])
        if len(history) < TRAJECTORY_WINDOW:
            return False, 1.0, 0

        start_pt = history[0]
        end_pt = history[-1]

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        displacement = math.hypot(dx, dy)

        if displacement < MIN_MOVEMENT_PIXELS:
            # Vehicle stationary or moving too slowly for reliable direction vector
            self.violation_counters[track_id] = 0
            return False, 1.0, 0

        # Vehicle velocity vector
        v_veh = (dx / displacement, dy / displacement)

        # Dot product = Cosine similarity
        cos_sim = v_veh[0] * self.expected_dir[0] + v_veh[1] * self.expected_dir[1]

        if cos_sim < self.cosine_thresh:
            self.violation_counters[track_id] += 1
        else:
            self.violation_counters[track_id] = 0

        consecutive_count = self.violation_counters[track_id]
        is_violation_flag = (consecutive_count >= self.persistence)

        return is_violation_flag, round(cos_sim, 3), consecutive_count
