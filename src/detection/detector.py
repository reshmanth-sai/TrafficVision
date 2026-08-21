import torch
from ultralytics import YOLO

# COCO Vehicle Class Indices
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_MAP = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}
VEHICLE_CLASSES = list(VEHICLE_CLASS_MAP.keys())

class VehicleDetector:
    """
    Primary vehicle detector wrapping Ultralytics YOLO11n.
    Restricted at inference time to native COCO vehicle classes [2, 3, 5, 7].
    
    Uses agnostic_nms=True to suppress duplicate overlapping multi-class bounding boxes
    (e.g., car vs truck predicted on the exact same physical vehicle), preventing track fragmentation.
    """
    def __init__(self, model_name='yolo11n.pt', conf_threshold=0.15, device=None):
        if device is None:
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing VehicleDetector with {model_name} on device '{self.device}'...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.classes = VEHICLE_CLASSES

    def detect(self, image_source, conf=None):
        """
        Runs detection on an image path or BGR image frame array.
        """
        th = conf if conf is not None else self.conf_threshold
        
        results = self.model(
            image_source,
            classes=self.classes,
            conf=th,
            agnostic_nms=True, # Class-agnostic NMS to eliminate duplicate multi-class bounding boxes
            device=self.device,
            verbose=False
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                cls_name = VEHICLE_CLASS_MAP.get(cls_id, self.model.names.get(cls_id, 'vehicle'))
                
                detections.append({
                    'box': [round(v, 2) for v in xyxy],
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': round(confidence, 4)
                })
                
        return detections, results[0]
