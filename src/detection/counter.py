from collections import Counter

class VehicleCounter:
    """
    Counts unique vehicle instances using persistent ByteTrack IDs and majority-vote class assignment.
    Filters display and counting based on a display confidence threshold (default: 0.35).
    Guarantees that each track_id contributes to EXCLUSIVELY ONE vehicle class and EXACTLY ONCE to total_unique.
    """
    TARGET_CLASSES = ['car', 'bus', 'motorcycle', 'truck']

    def __init__(self, display_conf=0.35):
        self.display_conf = display_conf
        # Track label history per track_id: {track_id: [list of valid class predictions]}
        self.track_class_history = {}
        # Count of un-tracked detections meeting display_conf
        self.untracked_detection_counts = {cls_name: 0 for cls_name in self.TARGET_CLASSES}

    def update(self, tracked_objects):
        """
        Updates counts for a frame given a list of tracked objects.
        
        Args:
            tracked_objects: list of dict with keys ['class_name', 'confidence', 'track_id', ...]
            
        Returns:
            active_counts (dict): Active visible counts per class in current frame.
            active_tracked_ids (list): List of (majority_class, track_id) tuples in current frame.
        """
        active_counts = {cls_name: 0 for cls_name in self.TARGET_CLASSES}
        active_tracked_ids = []

        for obj in tracked_objects:
            cls_name = obj['class_name']
            conf = obj['confidence']
            track_id = obj.get('track_id')

            # Enforce display / counting confidence threshold
            if conf >= self.display_conf and cls_name in self.TARGET_CLASSES:
                active_counts[cls_name] += 1
                if track_id is not None:
                    if track_id not in self.track_class_history:
                        self.track_class_history[track_id] = []
                    self.track_class_history[track_id].append(cls_name)

                    # Majority vote class for this track_id so far
                    maj_class = Counter(self.track_class_history[track_id]).most_common(1)[0][0]
                    active_tracked_ids.append((maj_class, track_id))
                else:
                    self.untracked_detection_counts[cls_name] += 1

        return active_counts, active_tracked_ids

    def get_counts(self):
        """
        Returns cumulative unique vehicle counts per class based on majority-vote track assignment.
        Each track_id is assigned EXCLUSIVELY to its majority-vote class.
        """
        counts = {cls_name: 0 for cls_name in self.TARGET_CLASSES}
        for tid, history in self.track_class_history.items():
            if len(history) > 0:
                maj_class = Counter(history).most_common(1)[0][0]
                if maj_class in counts:
                    counts[maj_class] += 1

        counts['total_unique'] = len(self.track_class_history)
        return counts

    def reset(self):
        """
        Resets unique track ID memory.
        """
        self.track_class_history = {}
        self.untracked_detection_counts = {cls_name: 0 for cls_name in self.TARGET_CLASSES}
