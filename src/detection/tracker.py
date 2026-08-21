import os
import torch
from ultralytics import YOLO

from src.detection.detector import VEHICLE_CLASS_MAP, VEHICLE_CLASSES

class VehicleTracker:
    """
    Vehicle tracker integrating ByteTrack via Ultralytics model.track().
    Maintains persistent track IDs across consecutive video frames.
    
    Uses agnostic_nms=True to suppress duplicate multi-class bounding boxes on the same physical vehicle,
    preventing ByteTrack from creating false split tracks (e.g. car#4 and truck#5 on 1 vehicle).
    """
    def __init__(self, model_name='yolo11n.pt', tracker_config=None, conf_threshold=0.15, device=None):
        if device is None:
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
            
        if tracker_config is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            tracker_config = os.path.join(base_dir, 'bytetrack_traffic.yaml')
            
        print(f"Initializing VehicleTracker with {model_name} and tracker config '{tracker_config}' on '{self.device}'...")
        self.model = YOLO(model_name)
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.classes = VEHICLE_CLASSES

    def track(self, frame, conf=None):
        """
        Processes a single video frame and updates ByteTrack state.
        """
        th = conf if conf is not None else self.conf_threshold
        
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            classes=self.classes,
            conf=th,
            agnostic_nms=True, # Enforce class-agnostic NMS for tracking consistency
            device=self.device,
            verbose=False
        )
        
        tracked_objects = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            has_ids = boxes.id is not None
            
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                cls_name = VEHICLE_CLASS_MAP.get(cls_id, self.model.names.get(cls_id, 'vehicle'))
                track_id = int(boxes.id[i].item()) if has_ids else None
                
                tracked_objects.append({
                    'box': [round(v, 2) for v in xyxy],
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': round(confidence, 4),
                    'track_id': track_id
                })
                
        return tracked_objects, results[0]
