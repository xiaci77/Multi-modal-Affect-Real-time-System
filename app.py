"""
基于Transformer跨模态对齐的多模态情感识别系统桌面应用
模式切换: 实时识别 / 录制识别 / 上传视频识别
"""
import os
import sys
import time
import queue
import threading
import subprocess
import numpy as np
from datetime import datetime

import cv2

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("[警告] sounddevice 未安装，麦克风录音不可用: pip install sounddevice")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QTextEdit,
                              QProgressBar, QGroupBox, QSpinBox, QFileDialog,
                              QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QPointF, QSize, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QBrush

import torch

from config import MultimodalConfig as Config
from inference import MultimodalEmotionRecognizer

# ================================================================
# 全局加载模型（模块级加载 FusionTransformer 和 ASR）
# ================================================================

print("=" * 50)
print("正在加载多模态情感识别模型...")
print("=" * 50)

recognizer = MultimodalEmotionRecognizer(model_type='transformer')

asr_model = None
try:
    print("[ASR] 正在加载 FunASR Paraformer 模型...")
    from funasr import AutoModel
    asr_model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc",
        punc_model_revision="v2.0.4",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("[ASR] Paraformer 模型加载完成")
except Exception as e:
    print(f"[ASR] 加载失败，文本模态将不可用: {e}")

print("=" * 50)
print("模型加载完成，启动 GUI...")
print("=" * 50)

# ================================================================
# Qt 颜色
# ================================================================

EMOTION_COLORS_Q = [
    QColor(144, 164, 174),
    QColor(255, 213, 79),
    QColor(66, 165, 245),
    QColor(239, 83, 80),
    QColor(171, 71, 188),
    QColor(120, 144, 156),
    QColor(38, 198, 218),
]

# ================================================================
# 自绘控件
# ================================================================

class ProbBarWidget(QWidget):
    def __init__(self, color=QColor(76, 175, 80), parent=None):
        super().__init__(parent)
        self._prob = 0.0
        self._color = color
        self.setMinimumHeight(16)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set_prob(self, prob):
        self._prob = max(0.0, min(1.0, prob))
        self.update()

    def clear(self):
        self._prob = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        radius = h / 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(230, 230, 230))
        painter.drawRoundedRect(0, 0, w, h, radius, radius)
        if self._prob > 0.005:
            bar_w = max(h, w * self._prob)
            painter.setBrush(self._color)
            painter.drawRoundedRect(0, 0, int(bar_w), h, radius, radius)
        painter.end()


class RadarChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._probs = np.zeros(7, dtype=np.float32)
        self._names = Config.CLASS_NAMES
        self._ring_count = 5
        self._fill_color = QColor(76, 175, 80, 60)
        self._line_color = QColor(76, 175, 80, 220)
        self._point_color = QColor(56, 142, 60, 255)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(220, 220)

    def sizeHint(self):
        return QSize(360, 360)

    def set_data(self, probs):
        self._probs = np.array(probs, dtype=np.float32)
        self.update()

    def clear(self):
        self._probs[:] = 0
        self.update()

    def _polar(self, cx, cy, r, idx, n):
        angle = 2 * np.pi * idx / n - np.pi / 2
        return cx + r * np.cos(angle), cy + r * np.sin(angle)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        side = min(w, h)
        cx, cy = w / 2, h / 2
        margin = side * 0.10
        max_r = (side - 2 * margin) / 2
        n = 7

        painter.setPen(QPen(QColor(200, 200, 200), 0.8))
        for ring in range(1, self._ring_count + 1):
            r = max_r * ring / self._ring_count
            pts = [self._polar(cx, cy, r, i, n) for i in range(n)]
            for i in range(n):
                j = (i + 1) % n
                painter.drawLine(int(pts[i][0]), int(pts[i][1]),
                                 int(pts[j][0]), int(pts[j][1]))

        font = QFont()
        font.setPointSize(max(7, int(side * 0.035)))
        painter.setFont(font)
        painter.setPen(QColor(150, 150, 150))
        for ring in range(1, self._ring_count + 1):
            val = ring / self._ring_count
            r = max_r * ring / self._ring_count
            painter.drawText(int(cx + 4), int(cy - r + 3), f"{val:.1f}")

        painter.setPen(QPen(QColor(180, 180, 180), 0.8))
        label_font = QFont()
        label_font.setPointSize(max(9, int(side * 0.048)))
        label_font.setBold(True)
        for i in range(n):
            ex, ey = self._polar(cx, cy, max_r, i, n)
            painter.drawLine(int(cx), int(cy), int(ex), int(ey))
            lx, ly = self._polar(cx, cy, max_r * 1.16, i, n)
            painter.setFont(label_font)
            painter.setPen(QColor(60, 60, 60))
            painter.drawText(int(lx - 30), int(ly - 12), 60, 24, Qt.AlignCenter, self._names[i])
            painter.setPen(QPen(QColor(180, 180, 180), 0.8))

        data = self._probs
        if data.max() < 1e-6:
            painter.end()
            return

        pts = [self._polar(cx, cy, max_r * min(1.0, data[i]), i, n) for i in range(n)]
        poly = [QPointF(x, y) for x, y in pts]
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._fill_color)
        painter.drawPolygon(*poly)
        painter.setPen(QPen(self._line_color, 2.5))
        painter.setBrush(Qt.NoBrush)
        for i in range(n):
            j = (i + 1) % n
            painter.drawLine(int(pts[i][0]), int(pts[i][1]), int(pts[j][0]), int(pts[j][1]))

        dot_r = max(3, int(side * 0.018))
        val_font = QFont()
        val_font.setPointSize(max(8, int(side * 0.038)))
        val_font.setBold(True)
        for i in range(n):
            x, y = pts[i]
            painter.setPen(Qt.NoPen)
            painter.setBrush(self._point_color)
            painter.drawEllipse(int(x - dot_r), int(y - dot_r), dot_r * 2, dot_r * 2)
            lx, ly = self._polar(cx, cy, max_r * min(1.0, data[i]) + max_r * 0.12, i, n)
            painter.setFont(val_font)
            painter.setPen(QColor(50, 50, 50))
            painter.drawText(int(lx - 20), int(ly - 10), 40, 20, Qt.AlignCenter, f"{data[i]:.0%}")

        painter.end()


class PieChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._probs = np.zeros(7, dtype=np.float32)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(150, 150)

    def sizeHint(self):
        return QSize(250, 250)

    def set_probs(self, probs):
        self._probs = np.array(probs, dtype=np.float32)
        self.update()

    def clear(self):
        self._probs = np.zeros(7, dtype=np.float32)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        side = min(w, h)
        margin = int(side * 0.06)
        outer_rect = QRectF(margin, margin, side - 2 * margin, side - 2 * margin)
        cx, cy = side / 2, side / 2
        inner_ratio = 0.42
        inner_w = (side - 2 * margin) * inner_ratio
        inner_h = (side - 2 * margin) * inner_ratio
        inner_rect = QRectF(cx - inner_w / 2, cy - inner_h / 2, inner_w, inner_h)

        total = self._probs.sum()
        if total < 1e-6:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(224, 224, 224))
            painter.drawEllipse(outer_rect)
            painter.setBrush(self.palette().window().color())
            painter.drawEllipse(inner_rect)
            painter.end()
            return

        start_angle = 90 * 16
        for i in range(7):
            if self._probs[i] < 1e-4:
                continue
            span = int(-self._probs[i] / total * 360 * 16)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            painter.setBrush(QBrush(EMOTION_COLORS_Q[i]))
            painter.drawPie(outer_rect, start_angle, span)
            start_angle += span

        painter.setPen(Qt.NoPen)
        bg = self.palette().window().color() if self.parent() else QColor(245, 245, 245)
        painter.setBrush(bg)
        painter.drawEllipse(inner_rect)

        max_idx = int(np.argmax(self._probs))
        max_prob = self._probs[max_idx]
        if max_prob > 0:
            painter.setPen(QColor(50, 50, 50))
            font = QFont()
            font.setPointSize(max(9, int(side * 0.08)))
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(inner_rect, Qt.AlignCenter, f"{max_prob:.0%}")

        painter.end()


class EmotionResultWidget(QWidget):
    def __init__(self, title, pie_size=0, use_radar=False, parent=None):
        super().__init__(parent)
        self.title = title
        self.radar = RadarChartWidget() if use_radar else None
        self.pie = None if use_radar else (PieChartWidget() if pie_size > 0 else None)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)

        header_row = QVBoxLayout()
        self.emotion_label = QLabel("待识别")
        ef = QFont()
        ef.setPointSize(18)
        ef.setBold(True)
        self.emotion_label.setFont(ef)
        self.emotion_label.setStyleSheet("color: #1976D2; padding: 2px 0;")

        self.confidence_label = QLabel("置信度: -")
        self.confidence_label.setStyleSheet("color: #666; font-size: 14px;")

        header_row.addWidget(self.emotion_label)
        header_row.addWidget(self.confidence_label)
        layout.addLayout(header_row)

        body_row = QHBoxLayout()
        body_row.setSpacing(10)

        prob_layout = QVBoxLayout()
        prob_layout.setSpacing(3)
        self.prob_labels = {}
        self.prob_bars = {}

        for i, emotion in enumerate(Config.CLASS_NAMES):
            row = QHBoxLayout()
            row.setSpacing(6)
            label = QLabel(f"{emotion}:")
            label.setMinimumWidth(55)
            label.setStyleSheet("font-size: 13px;")
            bar = ProbBarWidget(color=EMOTION_COLORS_Q[i])
            row.addWidget(label)
            row.addWidget(bar, 1)
            prob_layout.addLayout(row)
            self.prob_labels[emotion] = label
            self.prob_bars[emotion] = bar

        body_row.addLayout(prob_layout, 1)

        if self.pie:
            self.pie.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            body_row.addWidget(self.pie, 1, Qt.AlignCenter)
        if self.radar:
            self.radar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            body_row.addWidget(self.radar, 1)

        layout.addLayout(body_row)

        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(10)
        legend_layout.setContentsMargins(0, 4, 0, 0)
        for i, name in enumerate(Config.CLASS_NAMES):
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {EMOTION_COLORS_Q[i].name()}; font-size: 13px;")
            dot.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl = QLabel(name)
            lbl.setStyleSheet("font-size: 13px; color: #555;")
            legend_layout.addWidget(dot)
            legend_layout.addWidget(lbl)
        legend_layout.addStretch()
        layout.addLayout(legend_layout)

        self.setLayout(layout)

    def update_result(self, label, confidence, probabilities):
        self.emotion_label.setText(label)
        self.confidence_label.setText(f"置信度: {confidence:.1%}")
        for emotion, prob in probabilities.items():
            if emotion in self.prob_labels:
                self.prob_labels[emotion].setText(f"{emotion}: {prob:.1%}")
                self.prob_bars[emotion].set_prob(prob)
        if self.pie:
            probs = [probabilities.get(n, 0) for n in Config.CLASS_NAMES]
            self.pie.set_probs(np.array(probs, dtype=np.float32))
        if self.radar:
            probs = [probabilities.get(n, 0) for n in Config.CLASS_NAMES]
            self.radar.set_data(np.array(probs, dtype=np.float32))

    def clear(self):
        self.emotion_label.setText("待识别")
        self.confidence_label.setText("置信度: -")
        for emotion in Config.CLASS_NAMES:
            if emotion in self.prob_labels:
                self.prob_labels[emotion].setText(f"{emotion}: 0.0%")
                self.prob_bars[emotion].clear()
        if self.pie:
            self.pie.clear()
        if self.radar:
            self.radar.clear()

# ================================================================
# 后台推理工作线程（实时模式）
# ================================================================

class InferenceWorker(QThread):
    """摄像头开启时持续运行的实时推理线程，录制期间也不中断。"""
    visual_result_ready = pyqtSignal(np.ndarray)
    face_bbox_ready = pyqtSignal(tuple)
    speech_result_ready = pyqtSignal(np.ndarray)
    text_result_ready = pyqtSignal(np.ndarray)
    fused_result_ready = pyqtSignal(np.ndarray)
    asr_text_ready = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.frame_queue = queue.Queue(maxsize=4)
        self.audio_queue = queue.Queue(maxsize=4)
        self._visual_frame_skip = 15
        self._visual_counter = 0
        self._visual_ema_alpha = 0.3
        self._visual_probs = np.zeros(7, dtype=np.float32)
        self._visual_probs_ready = False
        self._speech_probs = np.zeros(7, dtype=np.float32)
        self._speech_probs_ready = False
        self._text_probs = np.zeros(7, dtype=np.float32)
        self._text_probs_ready = False

    def start_inference(self):
        self.running = True
        self.start()

    def stop_inference(self):
        self.running = False
        self._drain(self.frame_queue)
        self._drain(self.audio_queue)

    @staticmethod
    def _drain(q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def run(self):
        while self.running:
            try:
                self._process_frame(self.frame_queue.get_nowait())
            except queue.Empty:
                pass
            try:
                self._process_audio(self.audio_queue.get_nowait())
            except queue.Empty:
                pass
            time.sleep(0.005)

    def _process_frame(self, frame):
        self._visual_counter += 1
        if self._visual_counter % self._visual_frame_skip != 0:
            return
        try:
            ve = recognizer.feature_extractor.visual_extractor
            if ve and ve.model_loaded:
                feat, bbox = ve.extract_with_bbox(frame)
                self.face_bbox_ready.emit(bbox)
                if feat.max() > (1.0 / 7.0 + 0.01):
                    if not self._visual_probs_ready:
                        self._visual_probs = feat
                        self._visual_probs_ready = True
                    else:
                        self._visual_probs = (self._visual_ema_alpha * feat
                                              + (1 - self._visual_ema_alpha) * self._visual_probs)
                    self.visual_result_ready.emit(self._visual_probs.copy())
                    self._run_fusion()
        except Exception as e:
            print(f"[工作线程-视觉] {e}")

    def _process_audio(self, audio_data):
        if len(audio_data) < 0.5 * 16000:
            return
        if asr_model is not None:
            try:
                asr_result = asr_model.generate(input=audio_data.astype(np.float32))
                if asr_result and len(asr_result) > 0:
                    text = asr_result[0].get('text', '').strip()
                    if text:
                        self.asr_text_ready.emit(text)
                        te = recognizer.feature_extractor.text_extractor
                        if te and te.available:
                            self._text_probs = te.extract_from_text(text)
                            self._text_probs_ready = True
                            self.text_result_ready.emit(self._text_probs.copy())
            except Exception as e:
                print(f"[ASR] {e}")
        try:
            se = recognizer.feature_extractor.speech_extractor
            if se and se.model and se.processor:
                probs = self._extract_speech_from_array(se, audio_data)
                self._speech_probs = probs
                self._speech_probs_ready = True
                self.speech_result_ready.emit(probs.copy())
        except Exception as e:
            print(f"[语音情感] {e}")
        self._run_fusion()

    @staticmethod
    def _extract_speech_from_array(se, audio_data, sample_rate=16000):
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        audio_data = audio_data.astype(np.float32)
        max_samples = sample_rate * 5
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        inputs = se.processor(audio_data, sampling_rate=sample_rate,
                              return_tensors="pt", padding=True,
                              max_length=max_samples, truncation=True)
        input_values = inputs.input_values.to(se.device)
        attention_mask = (inputs.attention_mask.to(se.device)
                         if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None
                         else torch.ones_like(input_values).to(se.device))
        with torch.no_grad():
            p = torch.softmax(se.model(input_values, attention_mask), dim=1).cpu().numpy()[0]
        merged = np.zeros(7, dtype=np.float32)
        merged[0] = p[0] + p[1]    # 中性（含平静）
        merged[1:7] = p[2:8]       # 快乐, 悲伤, 愤怒, 恐惧, 厌恶, 惊讶
        merged /= (merged.sum() + 1e-8)
        return merged

    def _run_fusion(self):
        v = torch.tensor(self._visual_probs, dtype=torch.float32).unsqueeze(0).to(recognizer.device)
        s = torch.tensor(self._speech_probs, dtype=torch.float32).unsqueeze(0).to(recognizer.device)
        t = torch.tensor(self._text_probs, dtype=torch.float32).unsqueeze(0).to(recognizer.device)
        with torch.no_grad():
            self.fused_result_ready.emit(
                torch.softmax(recognizer.model(v, s, t), dim=1).cpu().numpy()[0])

# ================================================================
# 主窗口
# ================================================================

class MultimodalEmotionApp(QMainWindow):

    MODE_REALTIME = 'realtime'
    MODE_RECORD = 'record'
    MODE_UPLOAD = 'upload'

    def __init__(self):
        super().__init__()

        self.capture = None
        self.is_camera_running = False
        self.current_mode = None

        # 录制状态
        self.is_recording = False
        self.recorded_frames = []
        self._recorded_audio_chunks = []
        self._record_start = None
        self.recorded_video_path = None
        self.recorded_audio_path = None

        # 上传状态
        self.uploaded_video_path = None
        self.uploaded_audio_path = None

        # 实时推理工作线程
        self.worker = InferenceWorker()
        self._connect_worker_signals()

        # 实时音频采集（实时模式）
        self._audio_stream = None
        self._audio_samplerate = 16000
        self._audio_segment_sec = 5.0
        self._audio_buffer = []
        self._audio_accumulated_samples = 0

        # 录制模式专用麦克风（仅录制时开启）
        self._record_audio_stream = None

        # 推理结果缓存（线程安全）
        self._visual_probs = np.zeros(7, dtype=np.float32)
        self._visual_probs_ready = False
        self._speech_probs = np.zeros(7, dtype=np.float32)
        self._speech_probs_ready = False
        self._text_probs = np.zeros(7, dtype=np.float32)
        self._text_probs_ready = False
        self._fused_probs = np.zeros(7, dtype=np.float32)
        self._fused_probs_ready = False
        self._face_bbox = None
        self._result_lock = threading.Lock()

        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.init_ui()

        # 同步加载子模型（避免多线程 GPU context 冲突导致 0xC0000409）
        self._load_models()

    # ================================================================
    # 同步加载子模型（原 _ModelPreloadThread 改为同步）
    # ================================================================

    def _load_models(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        fe = recognizer.feature_extractor

        for label, fn in [
            ("视觉模型 (YOLO)", fe._ensure_visual),
            ("语音模型 (Wav2Vec2)", fe._ensure_speech),
            ("文本模型 (TF-IDF+SVM)", fe._ensure_text),
        ]:
            self.status_label.setText(f"正在加载 {label}...")
            QApplication.processEvents()
            try:
                fn()
            except Exception as e:
                print(f"[加载] {label} 失败: {e}")

        self.progress_bar.setVisible(False)
        self.status_label.setText("模型就绪 — 请选择识别模式")

    # ================================================================
    # 工作线程信号
    # ================================================================

    def _connect_worker_signals(self):
        self.worker.visual_result_ready.connect(self._on_visual_result)
        self.worker.face_bbox_ready.connect(self._on_face_bbox)
        self.worker.speech_result_ready.connect(self._on_speech_result)
        self.worker.text_result_ready.connect(self._on_text_result)
        self.worker.fused_result_ready.connect(self._on_fused_result)
        self.worker.asr_text_ready.connect(self._on_asr_text)

    def _on_visual_result(self, probs):
        if self.current_mode == self.MODE_RECORD:
            return  # 录制模式不更新视觉面板
        with self._result_lock:
            self._visual_probs, self._visual_probs_ready = probs, True
        self._update_visual_display(probs)

    def _on_face_bbox(self, bbox):
        with self._result_lock:
            self._face_bbox = bbox

    def _on_speech_result(self, probs):
        with self._result_lock:
            self._speech_probs, self._speech_probs_ready = probs, True
        self._update_speech_display(probs)

    def _on_text_result(self, probs):
        with self._result_lock:
            self._text_probs, self._text_probs_ready = probs, True
        self._update_text_display(probs)

    def _on_fused_result(self, probs):
        if self.current_mode == self.MODE_RECORD:
            return  # 录制模式不更新融合面板
        with self._result_lock:
            self._fused_probs, self._fused_probs_ready = probs, True
        self._update_fused_display(probs)

    def _on_asr_text(self, text):
        self.text_input.setPlainText(text)

    # ================================================================
    # UI
    # ================================================================

    def init_ui(self):
        self.setWindowTitle("多模态情感识别系统")
        self.setGeometry(100, 100, 1600, 1000)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)

        # ─── 模式切换 ───
        mode_group = QGroupBox()
        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(6, 4, 6, 4)
        mode_layout.setSpacing(8)

        self.btn_realtime = QPushButton("实时识别")
        self.btn_realtime.setMinimumSize(120, 36)
        self.btn_realtime.clicked.connect(lambda: self._switch_mode(self.MODE_REALTIME))

        self.btn_record_mode = QPushButton("录制识别")
        self.btn_record_mode.setMinimumSize(120, 36)
        self.btn_record_mode.clicked.connect(lambda: self._switch_mode(self.MODE_RECORD))

        self.btn_upload_mode = QPushButton("上传识别")
        self.btn_upload_mode.setMinimumSize(120, 36)
        self.btn_upload_mode.clicked.connect(lambda: self._switch_mode(self.MODE_UPLOAD))

        mode_layout.addWidget(self.btn_realtime)
        mode_layout.addWidget(self.btn_record_mode)
        mode_layout.addWidget(self.btn_upload_mode)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # ─── 控制面板 ───
        ctrl_group = QGroupBox("操作面板")
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        # 摄像头控制（实时/录制模式共用）
        self.btn_open = QPushButton("打开摄像头")
        self.btn_open.setStyleSheet(
            "QPushButton{background:#4CAF50;color:#fff;border:none;border-radius:4px;"
            "font-size:13px;font-weight:bold;padding:6px 14px}"
            "QPushButton:hover{background:#45a049}"
            "QPushButton:disabled{background:#ccc}")
        self.btn_open.clicked.connect(self.on_open_camera)

        self.btn_close = QPushButton("关闭摄像头")
        self.btn_close.setStyleSheet(
            "QPushButton{background:#e74c3c;color:#fff;border:none;border-radius:4px;"
            "font-size:13px;font-weight:bold;padding:6px 14px}"
            "QPushButton:hover{background:#c0392b}"
            "QPushButton:disabled{background:#ccc}")
        self.btn_close.clicked.connect(self.on_close_camera)
        self.btn_close.setEnabled(False)

        # 录制控制（仅录制模式）
        self.btn_record_start = QPushButton("开始录制")
        self.btn_record_start.setStyleSheet(
            "QPushButton{background:#e74c3c;color:#fff;border:none;border-radius:4px;"
            "font-size:13px;font-weight:bold;padding:6px 14px}"
            "QPushButton:hover{background:#c0392b}"
            "QPushButton:disabled{background:#ccc}")
        self.btn_record_start.clicked.connect(self.on_toggle_record)
        self.btn_record_start.setVisible(False)

        self.record_time_label = QLabel("")
        self.record_time_label.setStyleSheet("font-size:15px;font-weight:bold;color:#e74c3c;")
        self.record_time_label.setVisible(False)

        # 上传控制（仅上传模式）
        self.btn_upload_file = QPushButton("上传视频")
        self.btn_upload_file.setStyleSheet(
            "QPushButton{background:#FF9800;color:#fff;border:none;border-radius:4px;"
            "font-size:13px;font-weight:bold;padding:6px 14px}"
            "QPushButton:hover{background:#F57C00}"
            "QPushButton:disabled{background:#ccc}")
        self.btn_upload_file.clicked.connect(self.on_upload_video)
        self.btn_upload_file.setVisible(False)

        # 识别按钮（仅上传模式）
        self.btn_recognize = QPushButton("开始识别")
        self.btn_recognize.setStyleSheet(
            "QPushButton{background:#2196F3;color:#fff;border:none;border-radius:4px;"
            "font-size:13px;font-weight:bold;padding:6px 14px}"
            "QPushButton:hover{background:#1976D2}"
            "QPushButton:disabled{background:#ccc}")
        self.btn_recognize.clicked.connect(self.on_recognize)
        self.btn_recognize.setVisible(False)

        ctrl.addWidget(self.btn_open)
        ctrl.addWidget(self.btn_close)
        ctrl.addWidget(self.btn_record_start)
        ctrl.addWidget(self.record_time_label)
        ctrl.addWidget(self.btn_upload_file)
        ctrl.addWidget(self.btn_recognize)
        ctrl.addStretch()

        self.fps_label = QLabel("")
        self.fps_label.setStyleSheet("font-size:13px; color:#666;")
        ctrl.addWidget(self.fps_label)

        self.status_label = QLabel("请选择识别模式")
        ctrl.addWidget(self.status_label)

        ctrl_group.setLayout(ctrl)
        main_layout.addWidget(ctrl_group)

        # ─── 主内容 ───
        content = QHBoxLayout()
        content.setSpacing(8)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        preview_grp = QGroupBox("视频预览")
        preview_l = QVBoxLayout()
        self.video_label = QLabel("请先选择上方识别模式")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background:#1a1a1a;color:#888;border:2px dashed #444;border-radius:5px;")
        preview_l.addWidget(self.video_label)
        self.record_indicator = QLabel("")
        self.record_indicator.setAlignment(Qt.AlignCenter)
        self.record_indicator.setStyleSheet("font-size:14px;color:#e74c3c;font-weight:bold;")
        preview_l.addWidget(self.record_indicator)
        preview_grp.setLayout(preview_l)
        left_layout.addWidget(preview_grp, 5)

        fused_grp = QGroupBox("融合情感识别结果")
        fused_layout = QVBoxLayout()
        self.fused_result = EmotionResultWidget("融合情感识别结果", use_radar=True)
        fused_layout.addWidget(self.fused_result)
        fused_grp.setLayout(fused_layout)
        left_layout.addWidget(fused_grp, 2)
        content.addWidget(left, 2)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        visual_grp = QGroupBox("视觉情感 (YOLO)")
        visual_l = QVBoxLayout()
        self.visual_result = EmotionResultWidget("视觉情感", pie_size=90)
        visual_l.addWidget(self.visual_result)
        visual_grp.setLayout(visual_l)
        right_layout.addWidget(visual_grp, 1)

        speech_grp = QGroupBox("语音情感 (Wav2Vec2)")
        speech_l = QVBoxLayout()
        self.speech_result = EmotionResultWidget("语音情感", pie_size=90)
        speech_l.addWidget(self.speech_result)
        speech_grp.setLayout(speech_l)
        right_layout.addWidget(speech_grp, 1)

        text_grp = QGroupBox("文本情感 (ASR + TF-IDF+SVM)")
        text_l = QVBoxLayout()
        self.text_result = EmotionResultWidget("文本情感", pie_size=90)
        text_l.addWidget(self.text_result)
        text_grp.setLayout(text_l)
        right_layout.addWidget(text_grp, 1)

        trans_grp = QGroupBox("语音转录")
        trans_l = QVBoxLayout()
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("实时转录或上传文件识别结果...")
        self.text_input.setMaximumHeight(60)
        self.text_input.setReadOnly(True)
        trans_l.addWidget(self.text_input)
        trans_grp.setLayout(trans_l)
        right_layout.addWidget(trans_grp, 0)
        content.addWidget(right, 2)
        main_layout.addLayout(content)

        # ─── 底部 ───
        bottom = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(200)
        self.progress_bar.setVisible(False)
        bottom.addWidget(self.progress_bar)
        bottom.addStretch()
        main_layout.addLayout(bottom)

        # 定时器
        self._frame_timer = QTimer()
        self._frame_timer.timeout.connect(self._capture_frame)
        self._fps_count = 0
        self._fps_last = time.time()
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_fps)
        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self._update_record_time)
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self._blink)
        self._blink_on = False

    # ================================================================
    # 模式切换
    # ================================================================

    def _switch_mode(self, mode):
        if mode == self.current_mode:
            return
        old_mode = self.current_mode
        self.current_mode = mode

        # 离开录制或实时模式时关闭摄像头和录制
        if old_mode in (self.MODE_REALTIME, self.MODE_RECORD):
            self._close_camera_internal()

        # 离开上传模式: 清除上传状态（不删除文件，只清引用）
        if old_mode == self.MODE_UPLOAD:
            self.uploaded_video_path = None
            self.uploaded_audio_path = None

        # 切换模式时清空所有情感结果
        self.fused_result.clear()
        self.visual_result.clear()
        self.speech_result.clear()
        self.text_result.clear()
        self.text_input.clear()
        with self._result_lock:
            self._visual_probs[:] = 0
            self._visual_probs_ready = False
            self._speech_probs[:] = 0
            self._speech_probs_ready = False
            self._text_probs[:] = 0
            self._text_probs_ready = False
            self._fused_probs[:] = 0
            self._fused_probs_ready = False
            self._face_bbox = None

        # 更新模式按钮样式
        for m, btn in [(self.MODE_REALTIME, self.btn_realtime),
                       (self.MODE_RECORD, self.btn_record_mode),
                       (self.MODE_UPLOAD, self.btn_upload_mode)]:
            btn.setStyleSheet(
                "QPushButton{background:#1976D2;color:#fff;border:none;border-radius:4px;"
                "font-size:14px;font-weight:bold;padding:6px 16px}"
                if m == mode else
                "QPushButton{background:#e0e0e0;color:#666;border:none;border-radius:4px;"
                "font-size:14px;font-weight:bold;padding:6px 16px}"
                "QPushButton:hover{background:#d0d0d0}")

        # 显示/隐藏模式相关控件
        is_realtime = mode == self.MODE_REALTIME
        is_record = mode == self.MODE_RECORD
        is_upload = mode == self.MODE_UPLOAD

        self.btn_open.setVisible(is_realtime or is_record)
        self.btn_close.setVisible(is_realtime or is_record)

        self.btn_record_start.setVisible(is_record)
        self.record_time_label.setVisible(is_record)

        self.btn_upload_file.setVisible(is_upload)

        self.btn_recognize.setVisible(is_record or is_upload)  # 录制和上传模式需要手动识别

        # 重置控件状态
        self.btn_open.setEnabled(not is_upload)
        self.btn_close.setEnabled(False)
        self.btn_record_start.setEnabled(False)
        self.btn_record_start.setText("开始录制")
        self.btn_recognize.setEnabled(False)
        self.fps_label.setText("")

        # 重置录制状态
        self.is_recording = False
        self.recorded_frames = []
        self._recorded_audio_chunks = []
        self.recorded_video_path = None
        self.recorded_audio_path = None
        self.record_time_label.setText("")
        self.record_indicator.setText("")

        if is_realtime:
            self.status_label.setText("实时模式 — 打开摄像头即开始实时识别")
            self.video_label.setText("点击「打开摄像头」开始实时情感识别")
        elif is_record:
            self.status_label.setText("录制模式 — 打开摄像头后可以录制视频")
            self.video_label.setText("点击「打开摄像头」后使用录制功能")
        else:
            self.status_label.setText("上传模式 — 上传视频文件后点击「开始识别」")
            self.video_label.setText("点击「上传视频」选择文件")

        self.video_label.setStyleSheet(
            "background:#1a1a1a;color:#888;border:2px dashed #444;border-radius:5px;")

    def _close_camera_internal(self):
        """内部关闭摄像头（不触发录制停止的状态栏提示）"""
        self._stop_recording_internal()
        self.is_camera_running = False
        self._frame_timer.stop()
        if hasattr(self, 'audio_timer'):
            self.audio_timer.stop()
        self._fps_timer.stop()
        self.worker.stop_inference()
        self._stop_microphone()
        if self.capture:
            self.capture.release()
            self.capture = None

    def _stop_recording_internal(self):
        """内部停止录制（不保存文件、不弹提示）"""
        if self.is_recording:
            self.is_recording = False
            self.record_timer.stop()
            self.blink_timer.stop()
            self.record_indicator.setText("")
            self.record_time_label.setText("")
            self._stop_recording_mic()
            self.recorded_frames = []
            self._recorded_audio_chunks = []

    # ================================================================
    # 摄像头
    # ================================================================

    def on_open_camera(self):
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            QMessageBox.warning(self, "错误", "无法打开摄像头")
            return

        self.is_camera_running = True
        self.btn_open.setEnabled(False)
        self.btn_close.setEnabled(True)
        if self.current_mode == self.MODE_RECORD:
            self.btn_record_start.setEnabled(True)
        self.status_label.setText("摄像头 0 运行中")

        with self._result_lock:
            self._visual_probs[:] = 0
            self._visual_probs_ready = False
            self._speech_probs[:] = 0
            self._speech_probs_ready = False
            self._text_probs[:] = 0
            self._text_probs_ready = False
            self._fused_probs[:] = 0
            self._fused_probs_ready = False
            self._face_bbox = None

        self._audio_buffer = []
        self._audio_accumulated_samples = 0
        self._frame_timer.start(33)

        # 启动工作线程（实时模式全模态，录制模式仅视觉提供人脸框）
        self.worker.start_inference()
        if self.current_mode == self.MODE_REALTIME:
            self._start_microphone()
            self.audio_timer = QTimer()
            self.audio_timer.timeout.connect(self._flush_audio_buffer)
            self.audio_timer.start(int(self._audio_segment_sec * 1000))

        self._fps_timer.start(2000)

    def on_close_camera(self):
        self._stop_recording_internal()
        self.is_camera_running = False
        self._frame_timer.stop()
        if hasattr(self, 'audio_timer'):
            self.audio_timer.stop()
        self._fps_timer.stop()
        self.worker.stop_inference()
        self._stop_microphone()
        if self.capture:
            self.capture.release()
            self.capture = None

        self.btn_open.setEnabled(True)
        self.btn_close.setEnabled(False)
        self.btn_record_start.setEnabled(False)
        self.btn_record_start.setText("开始录制")
        self.status_label.setText("摄像头已关闭")
        self.fps_label.setText("")
        self.video_label.clear()
        self.video_label.setText("摄像头已关闭")
        self.video_label.setStyleSheet(
            "background:#1a1a1a;color:#888;border:2px dashed #444;border-radius:5px;")

    # ================================================================
    # 帧采集 + 显示
    # ================================================================

    def _capture_frame(self):
        if not self.is_camera_running:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        self._fps_count += 1

        if self.worker.running:
            try:
                self.worker.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

        if self.is_recording:
            self.recorded_frames.append(frame.copy())

        display = self._draw_overlay(frame.copy())
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._paint_text_overlay(pixmap, w, h)
        self.video_label.setPixmap(pixmap)
        self.video_label.setStyleSheet("border:none;")

    def _draw_overlay(self, frame):
        with self._result_lock:
            bbox = self._face_bbox
            fused_ready = self._fused_probs_ready

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (x1, y1), 3, (0, 255, 0), -1)

        if self.is_recording:
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)

        if fused_ready:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        return frame

    def _paint_text_overlay(self, pixmap, fw, fh):
        # 录制模式画面只画人脸框，不展示情感文字
        if self.current_mode == self.MODE_RECORD:
            return
        with self._result_lock:
            if not self._fused_probs_ready:
                return
            fused = self._fused_probs.copy()

        pw, ph = pixmap.width(), pixmap.height()
        s = min(pw / fw, ph / fh)
        ox = (pw - int(fw * s)) // 2
        oy = (ph - int(fh * s)) // 2

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        idx = int(np.argmax(fused))
        painter.setPen(EMOTION_COLORS_Q[idx])
        font = QFont("Microsoft YaHei", max(14, int(18 * s)))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(ox + int(12 * s), oy + int(32 * s),
                         f"融合: {Config.CLASS_NAMES[idx]}  {fused[idx]:.0%}")
        painter.end()

    # ================================================================
    # 录制
    # ================================================================

    def on_toggle_record(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self.is_camera_running:
            return
        self.is_recording = True
        self.recorded_frames = []
        self._recorded_audio_chunks = []
        self._record_start = time.time()
        self._start_recording_mic()

        self.btn_record_start.setText("停止录制")
        self.btn_record_start.setStyleSheet(
            "QPushButton{background:#333;color:#fff;border:2px solid #e74c3c;"
            "border-radius:4px;font-size:13px;font-weight:bold;padding:6px 14px}"
            "QPushButton:hover{background:#555}")
        self.record_timer.start(100)
        self.blink_timer.start(500)
        self.status_label.setText("正在录制...")

    def _stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.record_timer.stop()
        self.blink_timer.stop()
        self.record_indicator.setText("")
        self.record_time_label.setText("")

        self.btn_record_start.setText("开始录制")
        self.btn_record_start.setStyleSheet(
            "QPushButton{background:#e74c3c;color:#fff;border:none;border-radius:4px;"
            "font-size:13px;font-weight:bold;padding:6px 14px}"
            "QPushButton:hover{background:#c0392b}"
            "QPushButton:disabled{background:#ccc}")

        elapsed = time.time() - self._record_start if self._record_start else 0
        n = len(self.recorded_frames)
        self.status_label.setText(f"录制完成: {elapsed:.1f}s, {n} 帧")

        if n < 5:
            QMessageBox.warning(self, "提示", "录制时间太短，请至少录制 1-2 秒")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        v_only = os.path.join(self.tmp_dir, f"record_{ts}_v.mp4")
        a_path = os.path.join(self.tmp_dir, f"record_{ts}.wav")
        final = os.path.join(self.tmp_dir, f"record_{ts}.mp4")

        h, w = self.recorded_frames[0].shape[:2]
        fps = max(1.0, min(120.0, len(self.recorded_frames) / max(elapsed, 0.1)))
        try:
            out = cv2.VideoWriter(v_only, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for f in self.recorded_frames:
                out.write(f)
            out.release()
        except Exception as e:
            print(f"[录制] 视频保存失败: {e}")
            return

        has_audio = False
        if self._recorded_audio_chunks:
            try:
                import wave
                ad = np.concatenate(self._recorded_audio_chunks, axis=0)
                i16 = (ad * 32767).astype(np.int16)
                with wave.open(a_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self._audio_samplerate)
                    wf.writeframes(i16.tobytes())
                has_audio = True
            except Exception as e:
                print(f"[录制] 音频保存失败: {e}")

        if has_audio:
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', v_only, '-i', a_path,
                     '-c:v', 'copy', '-c:a', 'aac', '-shortest', final],
                    capture_output=True, text=True, timeout=30)
                if os.path.exists(final):
                    self.recorded_video_path = final
                    self.recorded_audio_path = a_path
                    try:
                        os.remove(v_only)
                    except Exception:
                        pass
                else:
                    self.recorded_video_path = v_only
                    self.recorded_audio_path = a_path
            except Exception as e:
                print(f"[录制] ffmpeg 合并失败: {e}")
                self.recorded_video_path = v_only
                self.recorded_audio_path = a_path
        else:
            self.recorded_video_path = v_only
            self.recorded_audio_path = None

        self._stop_recording_mic()
        self.recorded_frames = []

        # 停止录制后等待用户点击「开始识别」
        self.status_label.setText("录制已保存 — 点击「开始识别」")
        self.btn_recognize.setEnabled(True)
        self.btn_record_start.setEnabled(False)

    # ================================================================
    # 录制模式专用麦克风（仅录制时开启，与实时模式互斥）
    # ================================================================

    def _start_recording_mic(self):
        if not SOUNDDEVICE_AVAILABLE:
            return
        try:
            self._recorded_audio_chunks = []
            self._record_audio_stream = sd.InputStream(
                samplerate=self._audio_samplerate, channels=1,
                dtype='float32', callback=self._record_audio_callback)
            self._record_audio_stream.start()
        except Exception as e:
            print(f"[录制麦克风] 启动失败: {e}")

    def _stop_recording_mic(self):
        if self._record_audio_stream is not None:
            try:
                self._record_audio_stream.stop()
                self._record_audio_stream.close()
            except Exception:
                pass
            self._record_audio_stream = None

    def _record_audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[录制麦克风] {status}")
        self._recorded_audio_chunks.append(indata.copy())

    def _update_record_time(self):
        if self._record_start:
            self.record_time_label.setText(f"{time.time() - self._record_start:.1f}s")

    def _blink(self):
        self._blink_on = not self._blink_on
        if self._blink_on:
            self.record_indicator.setText("● REC")
            self.record_indicator.setStyleSheet("font-size:14px;color:#e74c3c;font-weight:bold;")
        else:
            self.record_indicator.setText("")

    # ================================================================
    # 上传视频
    # ================================================================

    def on_upload_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*)")
        if not path:
            return

        self.fused_result.clear()
        self.visual_result.clear()
        self.speech_result.clear()
        self.text_result.clear()
        self.text_input.clear()

        self.uploaded_video_path = path
        self.uploaded_audio_path = None

        a_path = os.path.join(self.tmp_dir, "upload_audio.wav")
        try:
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', path,
                 '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', a_path],
                capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and os.path.getsize(a_path) > 1000:
                self.uploaded_audio_path = a_path
        except Exception as e:
            print(f"[上传] 音频提取失败: {e}")

        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            self.video_label.setPixmap(
                QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))
                .scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.video_label.setStyleSheet("border:none;")

        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        dur = fc / fps if fps > 0 else 0
        cap.release()

        has_audio = self.uploaded_audio_path is not None
        fn = os.path.basename(path)
        self.status_label.setText(
            f"已加载: {fn} ({dur:.1f}s, {fc} 帧"
            f"{', 含音频' if has_audio else ', 无音频'}) → 点击「开始识别」")
        self.btn_recognize.setEnabled(True)

    # ================================================================
    # 一键识别
    # ================================================================

    def on_recognize(self):
        if self.current_mode == self.MODE_RECORD:
            if not (self.recorded_video_path and os.path.exists(self.recorded_video_path)):
                QMessageBox.warning(self, "错误", "请先录制视频")
                return
            video_path = self.recorded_video_path
            audio_path = (self.recorded_audio_path
                         if self.recorded_audio_path and os.path.exists(self.recorded_audio_path)
                         else None)
        elif self.current_mode == self.MODE_UPLOAD:
            if not (self.uploaded_video_path and os.path.exists(self.uploaded_video_path)):
                QMessageBox.warning(self, "错误", "请先上传视频文件")
                return
            video_path = self.uploaded_video_path
            audio_path = (self.uploaded_audio_path
                         if self.uploaded_audio_path and os.path.exists(self.uploaded_audio_path)
                         else None)
        else:
            QMessageBox.warning(self, "错误", "当前模式不支持识别操作")
            return

        # 录制模式识别前关闭摄像头，防止实时推理覆盖静态结果
        if self.current_mode == self.MODE_RECORD and self.is_camera_running:
            self.on_close_camera()

        self.btn_recognize.setEnabled(False)
        self.btn_record_start.setEnabled(False)
        self.btn_upload_file.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("正在识别...")
        QApplication.processEvents()

        try:
            self._do_recognize(video_path, audio_path)
        finally:
            self.progress_bar.setVisible(False)
            self.btn_recognize.setEnabled(True)
            self.btn_record_start.setEnabled(False)  # 摄像头已关闭，不可录制
            self.btn_upload_file.setEnabled(True)

    def _do_recognize(self, video_path, audio_path=None):
        t0 = time.time()
        audio_ok = audio_path and os.path.exists(audio_path)
        text_input = None

        if audio_ok and asr_model is not None:
            self.status_label.setText("正在进行语音识别 (ASR)...")
            QApplication.processEvents()
            try:
                r = asr_model.generate(input=audio_path)
                if r and len(r) > 0:
                    text_input = r[0].get('text', '').strip()
            except Exception as e:
                print(f"[ASR] {e}")

        self.text_input.setPlainText(text_input or "(未检测到语音)")

        self.status_label.setText("正在提取特征...")
        QApplication.processEvents()
        features = recognizer.feature_extractor.extract(
            video_path=video_path, audio_path=audio_path if audio_ok else None, text=text_input)

        ve = recognizer.feature_extractor.visual_extractor
        if ve and ve.model_loaded and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vf = []
            for idx in np.linspace(0, total - 1, min(5, total), dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    feat = ve.extract_from_frame(frame)
                    if feat.max() > (1.0 / 7.0 + 0.01):
                        vf.append(feat)
            cap.release()
            if vf:
                features['visual'] = np.mean(vf, axis=0)

        self.status_label.setText("正在融合预测...")
        QApplication.processEvents()
        vt = torch.tensor(features['visual'], dtype=torch.float32).unsqueeze(0).to(recognizer.device)
        st = torch.tensor(features['speech'], dtype=torch.float32).unsqueeze(0).to(recognizer.device)
        tt = torch.tensor(features['text'], dtype=torch.float32).unsqueeze(0).to(recognizer.device)
        with torch.no_grad():
            fused = torch.softmax(recognizer.model(vt, st, tt), dim=1).cpu().numpy()[0]

        elapsed = time.time() - t0

        fi = int(np.argmax(fused))
        self.fused_result.update_result(
            Config.CLASS_NAMES[fi], float(fused[fi]),
            {Config.CLASS_NAMES[i]: float(fused[i]) for i in range(7)})

        def safe_update(widget, arr, fail_label):
            if arr.max() > 0:
                idx = int(np.argmax(arr))
                widget.update_result(Config.CLASS_NAMES[idx], float(arr[idx]),
                                     {Config.CLASS_NAMES[i]: float(arr[i]) for i in range(7)})
            else:
                widget.update_result(fail_label, 0, {n: 0.0 for n in Config.CLASS_NAMES})

        safe_update(self.visual_result, features['visual'], "未检测到人脸")
        safe_update(self.speech_result, features['speech'],
                    "未检测到音频" if not audio_ok else "语音模型未加载")
        safe_update(self.text_result, features['text'],
                    "无文本输入" if not text_input else "文本模型未加载")

        self.status_label.setText(f"识别完成 ({elapsed:.2f}s)")

    # ================================================================
    # 麦克风
    # ================================================================

    def _start_microphone(self):
        if not SOUNDDEVICE_AVAILABLE:
            return
        try:
            self._audio_buffer = []
            self._audio_accumulated_samples = 0
            self._audio_stream = sd.InputStream(
                samplerate=self._audio_samplerate, channels=1,
                dtype='float32', callback=self._audio_callback)
            self._audio_stream.start()
        except Exception as e:
            print(f"[麦克风] 启动失败: {e}")
            self._audio_stream = None

    def _stop_microphone(self):
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None
        self._audio_buffer = []
        self._audio_accumulated_samples = 0
        self._recorded_audio_chunks = []

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[麦克风] {status}")
        self._audio_buffer.append(indata.copy())
        self._audio_accumulated_samples += frames
        if self.is_recording:
            self._recorded_audio_chunks.append(indata.copy())

    def _flush_audio_buffer(self):
        if not self.is_camera_running or not self._audio_buffer:
            return
        chunks = self._audio_buffer
        self._audio_buffer = []
        self._audio_accumulated_samples = 0
        audio_data = np.concatenate(chunks, axis=0).flatten()
        if self.worker.running:
            try:
                self.worker.audio_queue.put_nowait(audio_data)
            except queue.Full:
                pass

    # ================================================================
    # UI 更新
    # ================================================================

    def _update_visual_display(self, probs):
        i = int(np.argmax(probs))
        self.visual_result.update_result(
            Config.CLASS_NAMES[i], float(probs[i]),
            {Config.CLASS_NAMES[k]: float(probs[k]) for k in range(7)})

    def _update_speech_display(self, probs):
        i = int(np.argmax(probs))
        self.speech_result.update_result(
            Config.CLASS_NAMES[i], float(probs[i]),
            {Config.CLASS_NAMES[k]: float(probs[k]) for k in range(7)})

    def _update_text_display(self, probs):
        i = int(np.argmax(probs))
        self.text_result.update_result(
            Config.CLASS_NAMES[i], float(probs[i]),
            {Config.CLASS_NAMES[k]: float(probs[k]) for k in range(7)})

    def _update_fused_display(self, probs):
        i = int(np.argmax(probs))
        self.fused_result.update_result(
            Config.CLASS_NAMES[i], float(probs[i]),
            {Config.CLASS_NAMES[k]: float(probs[k]) for k in range(7)})

    def _update_fps(self):
        now = time.time()
        e = now - self._fps_last
        if e > 0:
            self.fps_label.setText(f"帧率: {self._fps_count / e:.0f} FPS")
        self._fps_count = 0
        self._fps_last = now

    # ================================================================
    # 自适应字号
    # ================================================================

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        title_fs = max(10, int(w * 0.020))
        conf_fs = max(9, int(w * 0.013))
        prob_fs = max(8, int(w * 0.011))

        for widget in (self.fused_result, self.visual_result,
                       self.speech_result, self.text_result):
            widget.emotion_label.setStyleSheet(
                f"color: #1976D2; padding: 2px 0; font-size: {title_fs}px; font-weight: bold;")
            widget.confidence_label.setStyleSheet(f"color: #666; font-size: {conf_fs}px;")
            for emotion in Config.CLASS_NAMES:
                if emotion in widget.prob_labels:
                    widget.prob_labels[emotion].setStyleSheet(f"font-size: {prob_fs}px;")

        grp_fs = max(9, int(w * 0.014))
        for grp in self.findChildren(QGroupBox):
            grp.setStyleSheet(
                f"QGroupBox {{ font-size: {grp_fs}px; font-weight: bold; "
                f"border:1px solid #ccc; border-radius:5px; "
                f"margin-top:10px; padding-top:10px; }}"
                f"QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 5px; }}")

        self.text_input.setStyleSheet(
            f"QTextEdit {{ font-size: {max(8, int(w * 0.010))}px; }}")

    # ================================================================
    # 清理
    # ================================================================

    def closeEvent(self, event):
        self._stop_recording_internal()
        self.is_camera_running = False
        self._frame_timer.stop()
        if hasattr(self, 'audio_timer'):
            self.audio_timer.stop()
        self._fps_timer.stop()
        self.worker.stop_inference()
        self.worker.wait(3000)
        self._stop_microphone()
        if self.capture:
            self.capture.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MultimodalEmotionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
