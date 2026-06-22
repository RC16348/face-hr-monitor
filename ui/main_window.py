from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon
import cv2
import numpy as np
import time
from pathlib import Path

from ui.camera_widget import CameraWidget
from ui.result_widget import ResultWidget
from core.detector import FaceDetector
from core.collector import FaceBuffer, crop_face, FACE_TARGET
from core.inferencer import FacePhysInferencer
from core.analyzer import compute_hrv
from core.advice import get_advice


INTRO = 'background: #1a1a2e;'
BTN_BASE = (
    'QPushButton { background: #1e2a3e; color: #7f8fa6; border: 1px solid #2c3e6b; '
    'border-radius: 6px; padding: 8px 20px; font-size: 13px; }'
    'QPushButton:hover { background: #2c3e6b; color: #c8d6e5; border-color: #4a6a9a; }'
)
BTN_START = (
    'QPushButton { background: #0fbc9c; color: #fff; border: none; border-radius: 6px; '
    'padding: 10px 28px; font-size: 14px; font-weight: bold; }'
    'QPushButton:hover { background: #1ad1b0; }'
    'QPushButton:disabled { background: #1e2a3e; color: #4a5a6a; }'
)
BTN_STOP = (
    'QPushButton { background: #e74c3c; color: #fff; border: none; border-radius: 6px; '
    'padding: 10px 28px; font-size: 14px; font-weight: bold; }'
    'QPushButton:hover { background: #c0392b; }'
    'QPushButton:disabled { background: #1e2a3e; color: #4a5a6a; }'
)
BTN_RESET = (
    'QPushButton { background: transparent; color: #7f8fa6; border: 1px solid #2c3e6b; '
    'border-radius: 6px; padding: 8px 16px; font-size: 12px; }'
    'QPushButton:hover { color: #c8d6e5; border-color: #4a6a9a; }'
)


class CameraWorker(QThread):
    frame_ready = Signal(object, object, object, object, object, object, object, object, object)
    collection_done = Signal(object)
    camera_error = Signal(str)

    def __init__(self, detector, buffer):
        super().__init__()
        self.detector = detector
        self.buffer = buffer
        self.running = False
        self.collecting = False
        self.detecting = False
        self.fps_timer = time.perf_counter()
        self.fps_counter = 0
        self.display_fps = 0
        self.display_count = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            self.camera_error.emit('无法打开摄像头')
            return

        self.running = True
        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                if self.detecting:
                    boxes, scores, kpts = self.detector.detect(frame)
                    no_face = len(boxes) == 0
                else:
                    boxes, scores, kpts = [], [], []
                    no_face = False

                if self.collecting and self.detecting and not no_face:
                    best = np.argmax(scores)
                    face = crop_face(frame, boxes[best])
                    if face is not None and self.buffer.add(face):
                        self.display_count = self.buffer.progress

                self.fps_counter += 1
                elapsed = time.perf_counter() - self.fps_timer
                if elapsed >= 1.0:
                    self.display_fps = self.fps_counter / elapsed
                    self.fps_counter = 0
                    self.fps_timer = time.perf_counter()

                kpts_copy = [kp.copy() for kp in kpts] if len(kpts) > 0 else []
                self.frame_ready.emit(
                    display_frame, boxes, scores, kpts_copy,
                    self.display_count, self.display_fps,
                    no_face, self.collecting, self.detecting
                )
                if self.collecting and self.buffer.is_full():
                    self.collecting = False
                    self.detecting = False
                    self.collection_done.emit(self.buffer.get_data())
            except Exception as e:
                print(f'[CameraWorker error] {e}')
        cap.release()

    def start_detecting(self):
        self.detecting = True

    def stop_detecting(self):
        self.detecting = False

    def start_collecting(self):
        self.collecting = True

    def stop_collecting(self):
        self.collecting = False

    def reset_count(self):
        self.display_count = 0

    def stop(self):
        self.running = False


class InferenceWorker(QThread):
    finished = Signal(dict, object, object, object)
    error = Signal(str)

    def __init__(self, model_path, state_path):
        super().__init__()
        self.model_path = model_path
        self.state_path = state_path
        self.faces = None

    def run(self):
        try:
            inf = FacePhysInferencer(self.model_path, self.state_path)
            bvp, _, _ = inf.inference(self.faces, fps=30)
            results = compute_hrv(bvp, fs=30)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(results['bvp_filtered'], distance=15)
            self.finished.emit(results, bvp, results['bvp_filtered'], peaks)
        except Exception as e:
            self.error.emit(str(e))


class AdviceWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, results):
        super().__init__()
        self.results = results

    def run(self):
        try:
            advice = get_advice(self.results)
            self.finished.emit(advice)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('面部心率监测')
        self.setMinimumSize(960, 720)
        self.setStyleSheet(INTRO)

        root = Path(__file__).parent.parent
        self.detector = FaceDetector(root / 'models' / 'yolov8n-face.onnx')
        self.buffer = FaceBuffer()
        self.phys_model = root / 'models' / 'model.onnx'
        self.phys_state = root / 'models' / 'state.gz'

        self._setup_ui()

        self.cam_worker = CameraWorker(self.detector, self.buffer)
        self.cam_worker.frame_ready.connect(self._on_frame)
        self.cam_worker.collection_done.connect(self._on_collection_done)
        self.cam_worker.camera_error.connect(self._on_camera_error)
        self.cam_worker.start()

        self.infer_worker = None
        self.advice_worker = None
        self._update_buttons(False)
        self.status.setText('就绪')

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        title_bar = QWidget()
        title_bar.setStyleSheet('background: #16213e; padding: 6px 0;')
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 6, 20, 6)

        self.title = QLabel('❤  面部心率监测')
        self.title.setStyleSheet('font-size: 16px; font-weight: bold; color: #c8d6e5;')
        title_layout.addWidget(self.title)
        title_layout.addStretch()

        self.status = QLabel()
        self.status.setStyleSheet('font-size: 12px; color: #7f8fa6;')
        title_layout.addWidget(self.status)
        root_layout.addWidget(title_bar)

        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack, stretch=1)

        cam_page = QWidget()
        cam_page.setStyleSheet(INTRO)
        cam_layout = QVBoxLayout(cam_page)
        cam_layout.setContentsMargins(20, 12, 20, 12)

        cam_frame = QWidget()
        cam_frame.setStyleSheet(
            'background: #0d1520; border: 1px solid #1e2a4a; border-radius: 10px; padding: 4px;')
        cam_frame_layout = QVBoxLayout(cam_frame)
        cam_frame_layout.setContentsMargins(0, 0, 0, 0)
        cam_frame_layout.setSpacing(3)
        cam_frame_layout.setAlignment(Qt.AlignCenter)

        cam_label = QLabel('摄像头预览')
        cam_label.setStyleSheet('font-size: 13px; color: #7f8fa6;')
        cam_label.setAlignment(Qt.AlignCenter)
        cam_frame_layout.addWidget(cam_label)

        self.cam_widget = CameraWidget()
        self.cam_widget.setStyleSheet('border-radius: 8px;')
        self.cam_widget.setFixedSize(960, 720)
        cam_frame_layout.addWidget(self.cam_widget)
        cam_layout.addWidget(cam_frame, stretch=1)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(12)
        ctrl_layout.addStretch()

        self.start_btn = QPushButton('开始采集')
        self.start_btn.setStyleSheet(BTN_START)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start_collection)
        ctrl_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton('停止')
        self.stop_btn.setStyleSheet(BTN_STOP)
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_collection)
        ctrl_layout.addWidget(self.stop_btn)

        self.reset_btn = QPushButton('重新开始')
        self.reset_btn.setStyleSheet(BTN_RESET)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self._reset_collection)
        ctrl_layout.addWidget(self.reset_btn)

        ctrl_layout.addStretch()
        cam_layout.addLayout(ctrl_layout)
        self.stack.addWidget(cam_page)

        self.result_widget = ResultWidget()
        self.result_widget.back_requested.connect(self._go_home)
        self.stack.addWidget(self.result_widget)

    def _update_buttons(self, collecting):
        self.start_btn.setEnabled(not collecting)
        self.stop_btn.setEnabled(collecting)

    def _on_frame(self, frame, boxes, scores, kpts, face_count, fps,
                  no_face, collecting, detecting):
        self.cam_widget.update_frame(
            frame.copy(), boxes, scores, kpts, face_count, fps,
            no_face, collecting, detecting)

    def _on_camera_error(self, msg):
        QMessageBox.critical(self, '错误', msg)
        self.status.setText(f'错误: {msg}')

    def _go_home(self):
        if self.infer_worker and self.infer_worker.isRunning():
            self.infer_worker.quit()
            self.infer_worker.wait(1000)
        if self.advice_worker and self.advice_worker.isRunning():
            self.advice_worker.quit()
            self.advice_worker.wait(1000)
        self.buffer.reset()
        self.cam_worker.reset_count()
        self.cam_worker.stop_detecting()
        self.cam_worker.collecting = False
        if not self.cam_worker.isRunning():
            self.cam_worker.start()
        self._update_buttons(False)
        self.start_btn.setText('开始采集')
        self.status.setText('就绪')
        self.stack.setCurrentIndex(0)

    def _start_collection(self):
        if self.buffer.is_full():
            self.buffer.reset()
            self.cam_worker.reset_count()
        self.cam_worker.start_detecting()
        self.cam_worker.start_collecting()
        self._update_buttons(True)
        self.stack.setCurrentIndex(0)
        self.status.setText('采集中...')
        self.start_btn.setText('采集中')

    def _stop_collection(self):
        self.cam_worker.stop_detecting()
        self.cam_worker.stop_collecting()
        self._update_buttons(False)
        self.start_btn.setText('继续采集')
        self.status.setText(f'已暂停  ({self.cam_worker.display_count}/{FACE_TARGET})')

    def _reset_collection(self):
        self.cam_worker.stop_detecting()
        self.cam_worker.stop_collecting()
        self.buffer.reset()
        self.cam_worker.reset_count()
        self._update_buttons(False)
        self.start_btn.setText('开始采集')
        self.status.setText('已重置')

    def _on_collection_done(self, faces):
        self._update_buttons(False)
        self.start_btn.setText('开始采集')
        self.stack.setCurrentIndex(1)
        self.result_widget.show_analyzing()
        self.status.setText(f'采集完成 ({len(faces)}帧)，正在分析...')

        self.infer_worker = InferenceWorker(self.phys_model, self.phys_state)
        self.infer_worker.faces = faces
        self.infer_worker.finished.connect(self._on_inference_done)
        self.infer_worker.error.connect(
            lambda e: self.status.setText(f'分析失败: {e}'))
        self.infer_worker.start()

    def _on_inference_done(self, results, bvp_raw, bvp_filt, peaks):
        self.result_widget.plot_results(
            bvp_raw, bvp_filt, peaks,
            results['freqs'], results['fft_vals'], results)
        self.status.setText(
            f'BPM: {results["bpm_peaks"]}  |  LF/HF: {results["lf_hf_ratio"]}')

        self.advice_worker = AdviceWorker(results)
        self.advice_worker.finished.connect(self.result_widget.show_advice)
        self.advice_worker.error.connect(
            lambda e: self.result_widget.show_advice(f'⚠️ 建议获取失败: {e}'))
        self.advice_worker.start()

    def closeEvent(self, event):
        if self.cam_worker:
            self.cam_worker.stop()
            self.cam_worker.wait(2000)
        for w in [self.infer_worker, self.advice_worker]:
            if w and w.isRunning():
                w.quit()
                w.wait(2000)
        event.accept()
