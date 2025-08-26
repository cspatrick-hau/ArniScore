import os
import sys
import cv2
from ultralytics import YOLO

def get_model_path(model_name):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, 'models', model_name)

class ArnisStrikeDetector:
    def __init__(self, min_conf=0.5, allowed_labels=None):
        self.yolo_model = YOLO(get_model_path('object_detection.pt'))
        self.convlstm_model_path = get_model_path('convlstm_v3.h5')

        self.class_names = self.yolo_model.names if hasattr(self.yolo_model, 'names') else {}
        self.min_conf = float(min_conf)
        self.allowed_labels = set(allowed_labels) if allowed_labels else None

    def detect(self, frame, debug=False):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(rgb_frame)

        best_label = None
        best_conf = 0.0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.class_names.get(cls, str(cls))

                if self.allowed_labels is not None and label not in self.allowed_labels:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_label = label

        if best_label is None:
            return rgb_frame, False, None, None

        valid = best_conf >= self.min_conf
        return rgb_frame, valid, best_conf, best_label
