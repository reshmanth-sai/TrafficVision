import numpy as np
from collections import deque

# Emergency Vehicle Light-Bar Heuristic Constants
# Cited: Emergency lighting standards (SAE J595 / NFPA 1901) mandate flash rates of 1.0 Hz - 4.0 Hz (60-240 flashes/min)
LIGHTBAR_CROP_RATIO = 0.30  # Top 30% of vehicle bounding box height
BUFFER_SIZE = 30           # 30 frames (~2.4s at 12.5 fps), covering >= 2.4 cycles of a 1.0 Hz signal
SMOOTHING_WINDOW = 3       # 3-frame moving average low-pass filter to suppress motion blur and reflections
MIN_CHROMATIC_DIFF = 15.0  # Minimum peak-to-peak Red-vs-Blue intensity difference above noise
MIN_FLASH_FREQ_HZ = 1.0    # Minimum emergency strobe flash rate (Hz)
MAX_FLASH_FREQ_HZ = 4.0    # Maximum emergency strobe flash rate (Hz)

class EmergencyVehicleHeuristic:
    """
    Detects emergency vehicles (ambulance, police, fire truck) by analyzing 
    chromatic Red-vs-Blue light-bar oscillation frequency in upper bounding box crops.
    """
    def __init__(self, buffer_size=BUFFER_SIZE, crop_ratio=LIGHTBAR_CROP_RATIO):
        self.buffer_size = buffer_size
        self.crop_ratio = crop_ratio
        # Store rolling history of raw chromatic diff (Mean_R - Mean_B) per track_id
        self.track_buffers = {}

    def update_track(self, track_id, frame_bgr, box):
        """
        Extracts upper 30% bounding box crop and records chromatic intensity difference (Mean_R - Mean_B).
        
        Args:
            track_id (int): Persistent ByteTrack ID.
            frame_bgr (np.ndarray): Full BGR image frame.
            box (list): Bounding box [x1, y1, x2, y2].
        """
        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = deque(maxlen=self.buffer_size)

        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame_bgr.shape[:2]
        
        # Clamp coordinates to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        box_h = y2 - y1
        if box_h < 10 or (x2 - x1) < 10:
            # Box too small for reliable light-bar extraction
            return

        # Crop top 30% of box height
        crop_y2 = y1 + max(int(box_h * self.crop_ratio), 5)
        crop = frame_bgr[y1:crop_y2, x1:x2]

        if crop.size == 0:
            return

        # Compute Mean Red and Mean Blue intensity
        mean_b = np.mean(crop[:, :, 0])
        mean_r = np.mean(crop[:, :, 2])
        chromatic_diff = float(mean_r - mean_b)

        self.track_buffers[track_id].append(chromatic_diff)

    def is_emergency(self, track_id, fps=12.5):
        """
        Evaluates rolling buffer for emergency light-bar chromatic oscillation.
        
        Returns:
            is_emergency (bool): True if chromatic oscillation frequency is in [1.0 Hz, 4.0 Hz].
            freq_hz (float): Estimated flash frequency (Hz).
            amplitude (float): Peak-to-peak chromatic difference.
        """
        if track_id not in self.track_buffers:
            return False, 0.0, 0.0

        buffer = list(self.track_buffers[track_id])
        if len(buffer) < self.buffer_size:
            # Need full buffer (30 frames) for reliable frequency estimation
            return False, 0.0, 0.0

        # 1. Apply 3-frame moving average low-pass filter to smooth signal
        smoothed_signal = np.convolve(buffer, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode='valid')

        # 2. Check peak-to-peak amplitude against noise threshold
        amplitude = float(np.max(smoothed_signal) - np.min(smoothed_signal))
        if amplitude < MIN_CHROMATIC_DIFF:
            return False, 0.0, amplitude

        # 3. Compute zero-crossings of mean-centered signal
        mean_centered = smoothed_signal - np.mean(smoothed_signal)
        zero_crossings = np.where(np.diff(np.signbit(mean_centered)))[0]
        num_crossings = len(zero_crossings)

        # 4. Calculate frequency in Hz
        duration_sec = len(smoothed_signal) / fps
        freq_hz = (num_crossings / 2.0) / duration_sec if duration_sec > 0 else 0.0

        # 5. Check if frequency falls within emergency strobe range [1.0 Hz, 4.0 Hz]
        is_emergency_flag = (MIN_FLASH_FREQ_HZ <= freq_hz <= MAX_FLASH_FREQ_HZ)

        return is_emergency_flag, round(freq_hz, 2), round(amplitude, 2)
