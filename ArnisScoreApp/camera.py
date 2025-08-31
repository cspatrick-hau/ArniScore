from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, Qt, pyqtSlot
import cv2
import numpy as np

def list_cameras(max_devices=10):
    available = []
    backends = [
        cv2.CAP_DSHOW,
        cv2.CAP_MSMF,
        cv2.CAP_ANY
    ]
    
    for i in range(max_devices):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                        cap.release()
                        break
                    cap.release()
            except:
                continue
                
    return available if available else [0]

class CameraThread(QThread):
    log_signal = pyqtSignal(bool, float, str)
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, detector=None, camera_index=0, use_prediction=False, prediction_instance=None, camera_number=1):
        super().__init__()
        self.detector = detector
        self.camera_index = camera_index
        self.detection_enabled = False
        self.running = False
        self.use_prediction = use_prediction
        self.prediction = prediction_instance
        self.camera_number = camera_number
        self.mutex = QMutex()
        self.last_frame = None

    def run(self):
        self.mutex.lock()
        self.running = True
        self.mutex.unlock()
        
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        cap = None
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(self.camera_index, backend)
                if cap.isOpened():
                    break
            except:
                continue
        
        if not cap or not cap.isOpened():
            print(f"Failed to open camera {self.camera_index}")
            return
            
        try:
            while self.is_running():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                    
                with QMutexLocker(self.mutex):
                    self.last_frame = frame.copy()
                    
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.detection_enabled and self.use_prediction and self.prediction:
                    try:
                        processed_frame = self.prediction.process_frame(frame, self.camera_number)
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        
                self.frame_signal.emit(rgb_frame)
        finally:
            if cap:
                cap.release()
            print(f"Camera {self.camera_index} stopped")

    def is_running(self):
        with QMutexLocker(self.mutex):
            return self.running

    @pyqtSlot()
    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False
        self.wait(500)

    def get_last_frame(self):
        with QMutexLocker(self.mutex):
            return self.last_frame.copy() if hasattr(self, 'last_frame') and self.last_frame is not None else None
