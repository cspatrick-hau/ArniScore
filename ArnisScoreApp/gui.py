from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QComboBox, QApplication, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from camera import CameraThread, list_cameras
from detection import ArnisStrikeDetector
from utils import export_to_csv, export_to_pdf
import os
import datetime
import sys
import traceback
import cv2
from threading import Lock
from ultralytics import YOLO

sys.excepthook = lambda exc_type, exc_value, exc_traceback: traceback.print_exception(
    exc_type, exc_value, exc_traceback
)

from PyQt5.QtWidgets import QProgressBar

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loading...")
        self.setFixedSize(400, 200)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        layout = QVBoxLayout()

        self.label = QLabel("Loading Arnis Strike Detection System...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 10px;
                margin: 0.5px;
            }
        """)

        self.progress_label = QLabel("Initializing...")
        self.progress_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        self.setLayout(layout)

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        QApplication.processEvents()

# ------------ CNN+LSTM Match Logs Window ------------
class MatchLogsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN-LSTM Realtime Detection")
        layout = QVBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Confidence", "Body Part", "Camera"]
        )
        layout.addWidget(self.table)
        buttons_layout = QHBoxLayout()

        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_table)
        buttons_layout.addWidget(self.clear_btn)

        # Export CSV button
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(self.export_logs_csv)
        buttons_layout.addWidget(self.export_csv_btn)

        # Export PDF button
        self.export_pdf_btn = QPushButton("Export PDF")
        self.export_pdf_btn.clicked.connect(self.export_logs_pdf)
        buttons_layout.addWidget(self.export_pdf_btn)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        self.resize(600, 350)

    def add_log(self, time_str, valid, confidence, scored_by, body_part, camera):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(time_str))
        self.table.setItem(row, 1, QTableWidgetItem(f"{confidence:.2f}"))
        self.table.setItem(row, 2, QTableWidgetItem(str(body_part)))
        self.table.setItem(row, 3, QTableWidgetItem(str(camera)))
        self.table.scrollToBottom()

    def clear_table(self):
        self.table.setRowCount(0)

    def export_logs_csv(self):
        from utils import export_cnn_lstm_logs_csv
        logs = []
        for row in range(self.table.rowCount()):
            time = self.table.item(row, 0).text()
            confidence = self.table.item(row, 1).text()
            body_part = self.table.item(row, 2).text()
            camera = self.table.item(row, 3).text()
            logs.append((time, confidence, body_part, camera))
        export_cnn_lstm_logs_csv("CNN-LSTM Detection Logs", logs)

    def export_logs_pdf(self):
        from utils import export_cnn_lstm_logs_pdf
        logs = []
        for row in range(self.table.rowCount()):
            time = self.table.item(row, 0).text()
            confidence = self.table.item(row, 1).text()
            body_part = self.table.item(row, 2).text()
            camera = self.table.item(row, 3).text()
            logs.append((time, confidence, body_part, camera))
        export_cnn_lstm_logs_pdf("CNN-LSTM Detection Logs", logs)

# ---------------- Prediction Class ----------------
class Prediction:
    VALID_PARTS = [
        "Head", "Body", "Legs", "Chest & Abdomen",
        "Side of the Body", "Upper Extremities", "Lower Extremities"
    ]
    INVALID_PARTS = ["Back of the Head", "Throat", "Hitting the Groin", "Back"]

    def __init__(self, log_callback=None):
        from detection import get_model_path
        self.model = YOLO(get_model_path("object_detection.pt"))
        self.convlstm_model_path = get_model_path("convlstm_v3.h5")
        self.lock = Lock()
        self.scores = {"Blue": 0, "Red": 0}
        self.winner = None
        self.last_hit = None
        self.confidence_threshold = 0.6
        self.log_callback = log_callback

    def calculate_iou(self, box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def classify_hit(self, player_box, stick_box):
        px1, py1, px2, py2 = player_box
        stick_center_y = (stick_box[1] + stick_box[3]) / 2
        height = py2 - py1
        if stick_center_y < py1 + height * 0.30:
            return "Head"
        elif stick_center_y < py1 + height * 0.70:
            return "Body"
        elif stick_center_y < py1 + height * 0.90:
            return "Legs"
        else:
            return "Invalid"

    def map_invalid_hit(self, classified_part, player_box):
        if classified_part == "Head":
            px1, py1, px2, py2 = player_box
            if py1 < py1 + (py2 - py1) * -0.25:
                return "Back of the Head"
        return classified_part

    def get_player_confidences(self, frame):
        """Returns the confidence of detected players."""
        with self.lock:
            result = self.model(frame, verbose=False, conf=self.confidence_threshold)[0]
            blue_conf = 0.0
            red_conf = 0.0
            for box in result.boxes:
                if box.conf < self.confidence_threshold:
                    continue
                class_id = int(box.cls)
                conf = float(box.conf)
                if class_id == 0:  # Blue player
                    blue_conf = max(blue_conf, conf)
                elif class_id == 2:  # Red player
                    red_conf = max(red_conf, conf)
            return blue_conf, red_conf

    def process_frame(self, frame, camera_number):
        with self.lock:
            result = self.model(frame, verbose=False, conf=self.confidence_threshold)[0]
            blue_players, blue_sticks, red_players, red_sticks = [], [], [], []
            for box in result.boxes:
                if box.conf < self.confidence_threshold:
                    continue
                class_id = int(box.cls)
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                if class_id == 0:
                    blue_players.append((coords, conf))
                elif class_id == 1:
                    blue_sticks.append((coords, conf))
                elif class_id == 2:
                    red_players.append((coords, conf))
                elif class_id == 3:
                    red_sticks.append((coords, conf))
            self.winner = None

        hit_registered = False
        scored_by_red = False
        body_part = ""
        valid = False

        # Red Stick hits Blue Player
        for red_stick, conf in red_sticks:
            for blue_player, _ in blue_players:
                iou = self.calculate_iou(red_stick, blue_player)
                if iou > 0.0 and self.last_hit != "Red":
                    body_part = self.classify_hit(blue_player, red_stick)
                    body_part = self.map_invalid_hit(body_part, blue_player)
                    valid = body_part not in self.INVALID_PARTS
                    if valid:
                        self.scores["Red"] += 1
                        scored_by_red = True
                    self.winner = "Red" if valid else None
                    self.last_hit = "Red"
                    hit_registered = True
                    break
            if hit_registered:
                break

        # Blue Stick hits Red Player
        if not hit_registered:
            for blue_stick, conf in blue_sticks:
                for red_player, _ in red_players:
                    iou = self.calculate_iou(blue_stick, red_player)
                    if iou > 0.0 and self.last_hit != "Blue":
                        body_part = self.classify_hit(red_player, blue_stick)
                        body_part = self.map_invalid_hit(body_part, red_player)
                        valid = body_part not in self.INVALID_PARTS
                        if valid:
                            self.scores["Blue"] += 1
                        self.winner = "Blue" if valid else None
                        self.last_hit = "Blue"
                        hit_registered = True
                        break
                if hit_registered:
                    break

        if hit_registered and self.log_callback:
            self.log_callback(
                valid,
                conf * 100.0,
                f"{'Red' if scored_by_red else 'Blue'} - {body_part}",
                camera_number
            )
        if not hit_registered:
            self.last_hit = None
        return result.plot()

# ---------------- ArnisApp GUI ----------------
class ArnisApp(QMainWindow):
    CNNLSTM = 1.5

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArniScore: Arnis Strike Detection System")
        self.setGeometry(200, 200, 1800, 900)
        self.statusBar().showMessage("Ready")
        self.detector = ArnisStrikeDetector()
        self.match_logs_window = MatchLogsWindow()
        self.cnnlstmshot_timer = None
        self.confidence_timer = QTimer()
        self.confidence_timer.timeout.connect(self.log_confidence_values)
        self.pending_event_1 = None
        self.pending_event_2 = None
        self.pending_event_3 = None
        # Cameras
        self.cameras = list_cameras() or [0]
        self.camera_thread_1 = None
        self.camera_thread_2 = None
        self.camera_thread_3 = None
        # Separate prediction instances for each camera
        self.predictions = {
            1: Prediction(log_callback=lambda valid, conf, body_part, cam_num=1:
                         self.update_log(valid, conf, body_part, cam_num)),
            2: Prediction(log_callback=lambda valid, conf, body_part, cam_num=2:
                         self.update_log(valid, conf, body_part, cam_num)),
            3: Prediction(log_callback=lambda valid, conf, body_part, cam_num=3:
                         self.update_log(valid, conf, body_part, cam_num))
        }
        # Track scores per camera
        self.scores = {
            1: {"Blue": 0, "Red": 0},
            2: {"Blue": 0, "Red": 0},
            3: {"Blue": 0, "Red": 0}
        }
        # Logs & Queues
        self.logs_1, self.logs_2, self.logs_3 = [], [], []
        self.log_queue_1, self.log_queue_2, self.log_queue_3 = [], [], []
        self.init_ui()
        # ---------------- Safety: Disable detection buttons by default ----------------
        self.start_btn_1.setEnabled(False)
        self.start_btn_2.setEnabled(False)
        self.start_btn_3.setEnabled(False)

    # ---------------- UI ----------------
    def init_ui(self):
        container = QWidget()
        main_layout = QVBoxLayout()
        # Camera selectors
        cam_select_layout = QHBoxLayout()
        self.camera_selector_1 = QComboBox()
        self.camera_selector_2 = QComboBox()
        self.camera_selector_3 = QComboBox()
        for cam in self.cameras:
            self.camera_selector_1.addItem(f"Camera {cam}")
            self.camera_selector_2.addItem(f"Camera {cam}")
            self.camera_selector_3.addItem(f"Camera {cam}")
        cam_select_layout.addWidget(self.camera_selector_1)
        cam_select_layout.addWidget(self.camera_selector_2)
        cam_select_layout.addWidget(self.camera_selector_3)
        
        self.refresh_cameras_btn = QPushButton("Refresh Camera List")
        self.refresh_cameras_btn.clicked.connect(self.refresh_camera_list)
        cam_select_layout.addWidget(self.refresh_cameras_btn)
        main_layout.addLayout(cam_select_layout)
        # Camera Feeds
        feed_layout = QHBoxLayout()
        self.camera_label_1 = QLabel("Camera Feed 1")
        self.camera_label_2 = QLabel("Camera Feed 2")
        self.camera_label_3 = QLabel("Camera Feed 3")
        for label in [self.camera_label_1, self.camera_label_2, self.camera_label_3]:
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(480, 360)
        feed_layout.addWidget(self.camera_label_1)
        feed_layout.addWidget(self.camera_label_2)
        feed_layout.addWidget(self.camera_label_3)
        main_layout.addLayout(feed_layout)
        # Scores
        score_layout = QHBoxLayout()
        self.blue_score_label_1 = QLabel("Blue: 0")
        self.red_score_label_1 = QLabel("Red: 0")
        self.blue_score_label_2 = QLabel("Blue: 0")
        self.red_score_label_2 = QLabel("Red: 0")
        self.blue_score_label_3 = QLabel("Blue: 0")
        self.red_score_label_3 = QLabel("Red: 0")
        for lbl in [
            self.blue_score_label_1, self.red_score_label_1,
            self.blue_score_label_2, self.red_score_label_2,
            self.blue_score_label_3, self.red_score_label_3,
        ]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size:16pt;font-weight:bold;")
        score_layout.addWidget(self.blue_score_label_1)
        score_layout.addWidget(self.red_score_label_1)
        score_layout.addWidget(self.blue_score_label_2)
        score_layout.addWidget(self.red_score_label_2)
        score_layout.addWidget(self.blue_score_label_3)
        score_layout.addWidget(self.red_score_label_3)
        main_layout.addLayout(score_layout)
        # Buttons
        button_layout = QHBoxLayout()
        self.preview_btn_1 = QPushButton("Preview Camera 1")
        self.start_btn_1 = QPushButton("Start Detection 1")
        self.stop_btn_1 = QPushButton("Stop Detection 1")
        self.preview_btn_2 = QPushButton("Preview Camera 2")
        self.start_btn_2 = QPushButton("Start Detection 2")
        self.stop_btn_2 = QPushButton("Stop Detection 2")
        self.preview_btn_3 = QPushButton("Preview Camera 3")
        self.start_btn_3 = QPushButton("Start Detection 3")
        self.stop_btn_3 = QPushButton("Stop Detection 3")
        self.reset_btn = QPushButton("Reset All")
        self.export_csv_btn = QPushButton("Export CSV All")
        self.export_pdf_btn = QPushButton("Export PDF All")
        self.match_logs_btn = QPushButton("CNN-LSTM Realtime Detection")
        self.preview_btn_1.clicked.connect(self.start_preview_1)
        self.start_btn_1.clicked.connect(lambda: self.start_detection(1))
        self.stop_btn_1.clicked.connect(lambda: self.stop_detection(1))
        self.preview_btn_2.clicked.connect(self.start_preview_2)
        self.start_btn_2.clicked.connect(lambda: self.start_detection(2))
        self.stop_btn_2.clicked.connect(lambda: self.stop_detection(2))
        self.preview_btn_3.clicked.connect(self.start_preview_3)
        self.start_btn_3.clicked.connect(lambda: self.start_detection(3))
        self.stop_btn_3.clicked.connect(lambda: self.stop_detection(3))
        self.stop_preview_btn_1 = QPushButton("Stop Preview 1")
        self.stop_preview_btn_1.clicked.connect(self.stop_preview_1)
        self.stop_preview_btn_2 = QPushButton("Stop Preview 2")
        self.stop_preview_btn_2.clicked.connect(self.stop_preview_2)
        self.stop_preview_btn_3 = QPushButton("Stop Preview 3")
        self.stop_preview_btn_3.clicked.connect(self.stop_preview_3)
        self.reset_btn.clicked.connect(self.reset_all)
        self.export_csv_btn.clicked.connect(self.export_all_csv)
        self.export_pdf_btn.clicked.connect(self.export_all_pdf)
        self.match_logs_btn.clicked.connect(self.open_match_logs)
        for btn in [
            self.preview_btn_1, self.stop_preview_btn_1, self.start_btn_1, self.stop_btn_1,
            self.preview_btn_2, self.stop_preview_btn_2, self.start_btn_2, self.stop_btn_2,
            self.preview_btn_3, self.stop_preview_btn_3, self.start_btn_3, self.stop_btn_3,
            self.reset_btn, self.export_csv_btn, self.export_pdf_btn,
        ]:
            button_layout.addWidget(btn)
        # ---- CNN+LSTM Match Logs button ----
        self.match_logs_btn = QPushButton("CNN-LSTM Realtime Detection")
        self.match_logs_btn.clicked.connect(self.open_match_logs)
        button_layout.addWidget(self.match_logs_btn)
        main_layout.addLayout(button_layout)
        # Logs Side by Side
        logs_layout = QHBoxLayout()
        log_container_1 = QVBoxLayout()
        log_container_1.addWidget(QLabel("Camera 1 Logs"))
        self.table_1 = QTableWidget(0, 5)
        self.table_1.setHorizontalHeaderLabels(
            ["Time", "Valid", "Confidence", "Scored By", "Body Part"]
        )
        log_container_1.addWidget(self.table_1)
        logs_layout.addLayout(log_container_1)
        log_container_2 = QVBoxLayout()
        log_container_2.addWidget(QLabel("Camera 2 Logs"))
        self.table_2 = QTableWidget(0, 5)
        self.table_2.setHorizontalHeaderLabels(
            ["Time", "Valid", "Confidence", "Scored By", "Body Part"]
        )
        log_container_2.addWidget(self.table_2)
        logs_layout.addLayout(log_container_2)
        log_container_3 = QVBoxLayout()
        log_container_3.addWidget(QLabel("Camera 3 Logs"))
        self.table_3 = QTableWidget(0, 5)
        self.table_3.setHorizontalHeaderLabels(
            ["Time", "Valid", "Confidence", "Scored By", "Body Part"]
        )
        log_container_3.addWidget(self.table_3)
        logs_layout.addLayout(log_container_3)
        main_layout.addLayout(logs_layout)
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # --------- Refresh Camera List ---------
    def refresh_camera_list(self):
        self.cameras = list_cameras() or [0]
        # Clear and repopulate camera selectors
        self.camera_selector_1.clear()
        self.camera_selector_2.clear()
        self.camera_selector_3.clear()
        for cam in self.cameras:
            self.camera_selector_1.addItem(f"Camera {cam}")
            self.camera_selector_2.addItem(f"Camera {cam}")
            self.camera_selector_3.addItem(f"Camera {cam}")
        self.statusBar().showMessage(f"Camera list refreshed. Found {len(self.cameras)} cameras.", 3000)

    # --------- CNN+LSTM Match Logs Controls ---------
    def open_match_logs(self):
        self.match_logs_window.show()

    def start_cnnlstmshot_timer(self):
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        for cam in (1, 2, 3):
            pending = getattr(self, f"pending_event_{cam}")
            if pending:
                valid, conf, scored_by, part = pending
                setattr(self, f"pending_event_{cam}", None)
            else:
                valid, conf, scored_by, part = False, 0.0, "", ""
            self.match_logs_window.add_log(
                now_str, valid, conf, scored_by, part, cam
            )
        self.cnnlstmshot_timer = QTimer()
        self.cnnlstmshot_timer.timeout.connect(self.start_cnnlstmshot_timer)
        self.cnnlstmshot_timer.start(int(self.CNNLSTM * 1000))

    def stop_cnnlstmshot_timer(self):
        if self.cnnlstmshot_timer:
            self.cnnlstmshot_timer.stop()
            self.cnnlstmshot_timer = None

    def log_confidence_values(self):
        """Log confidence values for both players every second."""
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        for cam_num in (1, 2, 3):
            thread = {1: self.camera_thread_1, 2: self.camera_thread_2, 3: self.camera_thread_3}.get(cam_num)
            if thread and thread.isRunning() and thread.detection_enabled:
                frame = thread.get_last_frame()
                if frame is not None:
                    blue_conf, red_conf = self.predictions[cam_num].get_player_confidences(frame)
                    self.match_logs_window.add_log(
                        now_str, "N/A", blue_conf * 100.0, "Blue", "Classifying", cam_num
                    )
                    self.match_logs_window.add_log(
                        now_str, "N/A", red_conf * 100.0, "Red", "Classifying", cam_num
                    )

    # ---------------- Camera Methods ----------------
    def start_preview_1(self):
        if self.camera_thread_1:
            self.camera_thread_1.stop()
            self.camera_thread_1.wait()
        cam_index = self.cameras[self.camera_selector_1.currentIndex()]
        self.camera_thread_1 = CameraThread(
            detector=None,
            camera_index=cam_index,
            use_prediction=True,
            prediction_instance=self.predictions[1],
            camera_number=1,
        )
        self.camera_thread_1.frame_signal.connect(
            lambda frame: self.update_camera_feed(frame, 1),
            Qt.QueuedConnection
        )
        self.camera_thread_1.detection_enabled = False
        self.camera_thread_1.start()
        self.statusBar().showMessage(f"Camera {cam_index} preview 1 started", 3000)
        self.start_btn_1.setEnabled(True)

    def start_preview_2(self):
        if self.camera_thread_2:
            self.camera_thread_2.stop()
            self.camera_thread_2.wait()
        cam_index = self.cameras[self.camera_selector_2.currentIndex()]
        self.camera_thread_2 = CameraThread(
            detector=None,
            camera_index=cam_index,
            use_prediction=True,
            prediction_instance=self.predictions[2],
            camera_number=2,
        )
        self.camera_thread_2.frame_signal.connect(
            lambda frame: self.update_camera_feed(frame, 2),
            Qt.QueuedConnection
        )
        self.camera_thread_2.detection_enabled = False
        self.camera_thread_2.start()
        self.statusBar().showMessage(f"Camera {cam_index} preview 2 started", 3000)
        self.start_btn_2.setEnabled(True)

    def start_preview_3(self):
        if self.camera_thread_3:
            self.camera_thread_3.stop()
            self.camera_thread_3.wait()
        cam_index = self.cameras[self.camera_selector_3.currentIndex()]
        self.camera_thread_3 = CameraThread(
            detector=None,
            camera_index=cam_index,
            use_prediction=True,
            prediction_instance=self.predictions[3],
            camera_number=3,
        )
        self.camera_thread_3.frame_signal.connect(
            lambda frame: self.update_camera_feed(frame, 3),
            Qt.QueuedConnection
        )
        self.camera_thread_3.detection_enabled = False
        self.camera_thread_3.start()
        self.statusBar().showMessage(f"Camera {cam_index} preview 3 started", 3000)
        self.start_btn_3.setEnabled(True)

    def stop_preview_1(self):
        if self.camera_thread_1:
            self.camera_thread_1.stop()
            self.camera_thread_1.wait()
            self.camera_label_1.clear()
            self.camera_label_1.setText("Camera Feed 1")
            self.statusBar().showMessage("Camera 1 preview stopped", 3000)
            self.start_btn_1.setEnabled(False)

    def stop_preview_2(self):
        if self.camera_thread_2:
            self.camera_thread_2.stop()
            self.camera_thread_2.wait()
            self.camera_label_2.clear()
            self.camera_label_2.setText("Camera Feed 2")
            self.statusBar().showMessage("Camera 2 preview stopped", 3000)
            self.start_btn_2.setEnabled(False)

    def stop_preview_3(self):
        if self.camera_thread_3:
            self.camera_thread_3.stop()
            self.camera_thread_3.wait()
            self.camera_label_3.clear()
            self.camera_label_3.setText("Camera Feed 3")
            self.statusBar().showMessage("Camera 3 preview stopped", 3000)
            self.start_btn_3.setEnabled(False)

    # ---------------- Start / Stop Detection ----------------
    def start_detection(self, cam_number):
        thread = {1: self.camera_thread_1, 2: self.camera_thread_2, 3: self.camera_thread_3}.get(cam_number)
        if not thread or not thread.isRunning():
            self.statusBar().showMessage(f"Start preview Camera {cam_number} first!", 5000)
            return
        thread.detection_enabled = True
        self.statusBar().showMessage(f"Detection {cam_number} started", 3000)
        if not self.confidence_timer.isActive():
            self.confidence_timer.start(1000)
        if not self.cnnlstmshot_timer:
            self.start_cnnlstmshot_timer()

    def stop_detection(self, cam_number):
        thread = {1: self.camera_thread_1, 2: self.camera_thread_2, 3: self.camera_thread_3}.get(cam_number)
        if thread:
            thread.detection_enabled = False
        self.statusBar().showMessage(f"Detection {cam_number} stopped", 3000)
        if not any(t and t.detection_enabled for t in [self.camera_thread_1, self.camera_thread_2, self.camera_thread_3]):
            self.stop_cnnlstmshot_timer()
            self.confidence_timer.stop()

    # ---------------- Camera Feed ----------------
    def update_camera_feed(self, frame, cam_number):
        if frame is not None:
            lbl = {1: self.camera_label_1, 2: self.camera_label_2, 3: self.camera_label_3}.get(cam_number)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            rgb_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            lbl.setPixmap(QPixmap.fromImage(rgb_image))

    # ---------------- Logs ----------------
    def update_log(self, valid, confidence, body_part, cam_number):
        if isinstance(body_part, str) and "-" in body_part:
            scored_by, part = [x.strip() for x in body_part.split("-", 1)]
        else:
            scored_by, part = body_part if isinstance(body_part, tuple) else ("", body_part)
        setattr(self, f"pending_event_{cam_number}", (valid, confidence, scored_by, part))
        table = {1: self.table_1, 2: self.table_2, 3: self.table_3}[cam_number]
        log_queue = {1: self.log_queue_1, 2: self.log_queue_2, 3: self.log_queue_3}[cam_number]
        logs = {1: self.logs_1, 2: self.logs_2, 3: self.logs_3}[cam_number]
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_queue.append((timestamp, valid, confidence, scored_by, part))
        if valid and scored_by in ("Red", "Blue"):
            opposing = "Blue" if scored_by == "Red" else "Red"
            log_queue.append((timestamp, False, 0.0, opposing, part))
        for log_entry in log_queue:
            logs.append(log_entry)
            row = table.rowCount()
            table.insertRow(row)
            for i, val in enumerate(log_entry):
                table.setItem(row, i, QTableWidgetItem(str(val)))
        table.scrollToBottom()
        log_queue.clear()
        self.scores[cam_number]["Blue"] = self.predictions[cam_number].scores["Blue"]
        self.scores[cam_number]["Red"] = self.predictions[cam_number].scores["Red"]
        self.update_scores(cam_number)

    # ---------------- Update Scores ----------------
    def update_scores(self, cam_number):
        if cam_number == 1:
            self.blue_score_label_1.setText(f"Blue: {self.scores[1]['Blue']}")
            self.red_score_label_1.setText(f"Red: {self.scores[1]['Red']}")
        elif cam_number == 2:
            self.blue_score_label_2.setText(f"Blue: {self.scores[2]['Blue']}")
            self.red_score_label_2.setText(f"Red: {self.scores[2]['Red']}")
        else:
            self.blue_score_label_3.setText(f"Blue: {self.scores[3]['Blue']}")
            self.red_score_label_3.setText(f"Red: {self.scores[3]['Red']}")

    # ---------------- Reset & Export ----------------
    def reset_all(self):
        for table in [self.table_1, self.table_2, self.table_3]:
            table.setRowCount(0)
        self.logs_1.clear()
        self.logs_2.clear()
        self.logs_3.clear()
        self.log_queue_1.clear()
        self.log_queue_2.clear()
        self.log_queue_3.clear()
        for cam_num in [1, 2, 3]:
            self.scores[cam_num] = {"Blue": 0, "Red": 0}
            self.predictions[cam_num].scores = {"Blue": 0, "Red": 0}
            self.predictions[cam_num].last_hit = None
            self.predictions[cam_num].winner = None
        self.update_scores(1)
        self.update_scores(2)
        self.update_scores(3)
        self.statusBar().showMessage("All logs and scores reset", 3000)

    def export_all_csv(self):
        export_to_csv(
            [
                ("Camera 1 Logs", self.logs_1),
                ("Camera 2 Logs", self.logs_2),
                ("Camera 3 Logs", self.logs_3),
            ]
        )

    def export_all_pdf(self):
        export_to_pdf(
            [
                ("Camera 1 Logs", self.logs_1),
                ("Camera 2 Logs", self.logs_2),
                ("Camera 3 Logs", self.logs_3),
            ]
        )


