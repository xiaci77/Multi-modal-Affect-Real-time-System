"""
多模态特征提取器包装
调用现有单模态情感识别代码提取特征
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import cv2
import torch
import librosa
import numpy as np

# YOLO 模型导入
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[警告] Ultralytics 未安装，请运行: pip install ultralytics")

from transformers import Wav2Vec2Processor

from config import MultimodalConfig as Config

# 语音项目路径（从 config 读取）
SPEECH_PROJECT_DIR = Config.SPEECH_PROJECT_DIR


class VisualFeatureExtractor:
    """
    视觉特征提取器（基于YOLO双模型）
    - 人脸检测模型: yolov11n-人脸检测权重.pt
    - 表情分类模型: yolo11s-表情权重.pt

    输出7维情感概率向量，与 Config.CLASS_NAMES 对齐
    Config.CLASS_NAMES: ['中性', '快乐', '悲伤', '愤怒', '恐惧', '厌恶', '惊讶']

    YOLO模型类别顺序:
    - 0: angry   -> Config[3] = 愤怒
    - 1: disgust -> Config[5] = 厌恶
    - 2: fear    -> Config[4] = 恐惧
    - 3: happy   -> Config[1] = 快乐
    - 4: neutral -> Config[0] = 中性
    - 5: sad     -> Config[2] = 悲伤
    - 6: surprised -> Config[6] = 惊讶
    """

    # YOLO输出 -> Config.CLASS_NAMES 的映射索引
    YOLO_TO_CONFIG_MAPPING = [3, 5, 4, 1, 0, 2, 6]  # [愤怒, 厌恶, 恐惧, 快乐, 中性, 悲伤, 惊讶]

    def __init__(self, detect_model_path=None, cls_model_path=None):
        self.detect_model = None
        self.cls_model = None
        self.model_loaded = False

        # 模型路径
        if detect_model_path is None:
            detect_model_path = Config.YOLO_DETECT_MODEL
        if cls_model_path is None:
            cls_model_path = Config.YOLO_CLS_MODEL

        self.detect_model_path = detect_model_path
        self.cls_model_path = cls_model_path

        # 尝试加载模型
        self._load_models()

    def _load_models(self):
        """加载YOLO模型"""
        if not ULTRALYTICS_AVAILABLE:
            print("[警告] Ultralytics 未安装，无法使用YOLO模型")
            return

        try:
            # 加载人脸检测模型
            if os.path.exists(self.detect_model_path):
                print(f"[视觉] 正在加载 YOLO 人脸检测模型...")
                self.detect_model = YOLO(self.detect_model_path)
                print(f"[视觉] ✅ 人脸检测模型加载完成: {self.detect_model_path}")
            else:
                print(f"[视觉] ⚠️ 人脸检测模型不存在: {self.detect_model_path}")

            # 加载表情分类模型
            if os.path.exists(self.cls_model_path):
                print(f"[视觉] 正在加载 YOLO 表情分类模型...")
                self.cls_model = YOLO(self.cls_model_path)
                # 获取类别名称
                if hasattr(self.cls_model, 'names'):
                    print(f"[视觉] 表情类别: {self.cls_model.names}")
                print(f"[视觉] ✅ 表情分类模型加载完成: {self.cls_model_path}")
            else:
                print(f"[视觉] ⚠️ 表情分类模型不存在: {self.cls_model_path}")

            if self.detect_model is not None and self.cls_model is not None:
                self.model_loaded = True
                print("[视觉] ✅ 所有YOLO模型加载完成")

        except Exception as e:
            print(f"[视觉] 模型加载失败: {e}")
            self.model_loaded = False

    def _align_emotion_probs(self, raw_probs):
        """
        将YOLO模型输出的概率对齐到Config.CLASS_NAMES顺序

        Args:
            raw_probs: YOLO模型输出的7维概率 [angry, disgust, fear, happy, neutral, sad, surprised]

        Returns:
            aligned_probs: 对齐后的7维概率 ['中性', '快乐', '悲伤', '愤怒', '恐惧', '厌恶', '惊讶']
        """
        aligned = np.zeros(7, dtype=np.float32)
        for yolo_idx, config_idx in enumerate(self.YOLO_TO_CONFIG_MAPPING):
            aligned[config_idx] = raw_probs[yolo_idx]

        # 归一化
        total = aligned.sum()
        if total > 0:
            aligned = aligned / total
        else:
            # 如果全为0，返回均匀分布
            aligned = np.ones(7, dtype=np.float32) / 7

        return aligned

    def extract_from_frame(self, frame, conf_threshold=0.25):
        """
        从单帧图像提取表情特征

        Args:
            frame: BGR格式的图像 (numpy array)
            conf_threshold: 人脸检测置信度阈值

        Returns:
            7维情感概率向量，顺序与 Config.CLASS_NAMES 一致
        """
        if not self.model_loaded:
            print("[视觉] YOLO模型未加载，返回全零向量")
            return np.zeros(7, dtype=np.float32)

        try:
            # BGR转RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Step 1: 人脸检测
            detect_results = self.detect_model.predict(
                source=frame_rgb,
                conf=conf_threshold,
                verbose=False
            )

            if not detect_results or len(detect_results) == 0:
                return np.zeros(7, dtype=np.float32)

            res = detect_results[0]
            boxes = res.boxes

            if boxes is None or len(boxes) == 0:
                return np.zeros(7, dtype=np.float32)

            # 获取所有人脸框和置信度
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            # 选择置信度最高的人脸（暂不处理多面孔）
            best_idx = np.argmax(confs)
            x1, y1, x2, y2 = xyxy[best_idx].astype(int)

            # 边界检查
            h, w = frame_rgb.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # 裁剪人脸区域
            if x2 <= x1 or y2 <= y1:
                return np.zeros(7, dtype=np.float32)

            face_crop = frame_rgb[y1:y2, x1:x2]

            # Step 2: 表情分类
            cls_results = self.cls_model.predict(
                source=face_crop,
                verbose=False
            )

            if not cls_results or len(cls_results) == 0:
                return np.zeros(7, dtype=np.float32)

            cls_res = cls_results[0]

            # 获取分类概率
            if hasattr(cls_res, 'probs') and cls_res.probs is not None:
                probs_data = cls_res.probs.data.cpu().numpy()
                # 使用全概率分布
                raw_probs = probs_data
            else:
                # 如果没有概率分布，返回均匀分布（无法确定类别）
                print("[视觉] 分类结果无概率分布，返回均匀分布")
                raw_probs = np.ones(7, dtype=np.float32) / 7

            # 对齐到Config.CLASS_NAMES顺序
            aligned_probs = self._align_emotion_probs(raw_probs)
            return aligned_probs

        except Exception as e:
            print(f"[视觉] 特征提取失败: {e}")
            return np.zeros(7, dtype=np.float32)

    def extract_with_bbox(self, frame, conf_threshold=0.25):
        """
        从单帧提取表情特征，同时返回人脸边界框（给实时画面绘制用）。

        Returns:
            (7维情感概率向量, bbox_or_None)
            bbox: (x1, y1, x2, y2) in frame coordinates, or None
        """
        if not self.model_loaded:
            return np.zeros(7, dtype=np.float32), None

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detect_results = self.detect_model.predict(
                source=frame_rgb, conf=conf_threshold, verbose=False)
            if not detect_results or len(detect_results) == 0:
                return np.zeros(7, dtype=np.float32), None
            res = detect_results[0]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                return np.zeros(7, dtype=np.float32), None

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs)
            x1, y1, x2, y2 = xyxy[best_idx].astype(int)
            h, w = frame_rgb.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            bbox = (int(x1), int(y1), int(x2), int(y2))

            if x2 <= x1 or y2 <= y1:
                return np.zeros(7, dtype=np.float32), None

            face_crop = frame_rgb[y1:y2, x1:x2]
            cls_results = self.cls_model.predict(source=face_crop, verbose=False)
            if not cls_results or len(cls_results) == 0:
                return np.zeros(7, dtype=np.float32), bbox

            cls_res = cls_results[0]
            if hasattr(cls_res, 'probs') and cls_res.probs is not None:
                raw_probs = cls_res.probs.data.cpu().numpy()
            else:
                raw_probs = np.ones(7, dtype=np.float32) / 7

            aligned_probs = self._align_emotion_probs(raw_probs)
            return aligned_probs, bbox
        except Exception as e:
            print(f"[视觉] extract_with_bbox 失败: {e}")
            return np.zeros(7, dtype=np.float32), None

    def extract_from_video(self, video_path, sample_frames=10, conf_threshold=0.25):
        """
        从视频文件提取表情特征

        Args:
            video_path: 视频文件路径
            sample_frames: 采样帧数
            conf_threshold: 人脸检测置信度阈值

        Returns:
            7维情感概率向量，多帧取平均
        """
        if not os.path.exists(video_path):
            print(f"[视觉] 视频文件不存在: {video_path}")
            return np.zeros(7, dtype=np.float32)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return np.zeros(7, dtype=np.float32)

        # 均匀采样帧索引
        indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

        features = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                feat = self.extract_from_frame(frame, conf_threshold)
                # 过滤无人脸帧（全零）和分类失败帧（均匀分布），避免稀释有效帧
                if feat.max() > (1.0 / 7.0 + 0.01):
                    features.append(feat)

        cap.release()

        if len(features) == 0:
            return np.zeros(7, dtype=np.float32)

        # 对有效帧的概率取平均
        return np.mean(features, axis=0)


class SpeechFeatureExtractor:
    """语音特征提取器（基于Wav2Vec2）"""
    def __init__(self, model_path=Config.SPEECH_MODEL_PATH, processor_path=Config.SPEECH_PROCESSOR_PATH):
        self.device = Config.DEVICE
        import importlib.util

        original_config = sys.modules.get('config')

        try:
            speech_config_path = os.path.join(SPEECH_PROJECT_DIR, 'config.py')
            spec = importlib.util.spec_from_file_location('config', speech_config_path)
            config_module = importlib.util.module_from_spec(spec)
            sys.modules['config'] = config_module
            spec.loader.exec_module(config_module)

            speech_model_py_path = os.path.join(SPEECH_PROJECT_DIR, 'model.py')
            spec = importlib.util.spec_from_file_location('model', speech_model_py_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            self.model_class = model_module.Wav2Vec2EmotionModel
        finally:
            if original_config is not None:
                sys.modules['config'] = original_config
            else:
                if 'config' in sys.modules and sys.modules['config'] is config_module:
                    del sys.modules['config']
        self.model = None
        self.processor = None
        self.load_model(model_path, processor_path)

    def load_model(self, model_path, processor_path):
        try:
            print(f"[语音情感] 模型路径: {model_path}")
            print(f"[语音情感] 模型文件: {'✅' if os.path.exists(model_path) else '❌'}")
            if not os.path.exists(model_path):
                print(f"语音模型文件不存在: {model_path}")
                return
            self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            config = checkpoint['config']
            self.model = self.model_class(
                model_name=config['model_name'],
                num_classes=config['num_classes']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print("语音模型加载成功")
        except Exception as e:
            print(f"语音模型加载失败: {e}")

    def extract_from_audio(self, audio_path):
        if self.model is None or self.processor is None:
            print("语音模型未加载")
            return np.zeros(Config.SPEECH_FEAT_DIM, dtype=np.float32)

        try:
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000, duration=5.0)
            inputs = self.processor(
                speech_array, sampling_rate=sampling_rate,
                return_tensors="pt", padding=True,
                max_length=16000 * 5, truncation=True
            )
            input_values = inputs.input_values.to(self.device)
            if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
                attention_mask = inputs.attention_mask.to(self.device)
            else:
                attention_mask = torch.ones_like(input_values).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_values, attention_mask)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            # Wav2Vec2原始8类: [中性, 平静, 快乐, 悲伤, 愤怒, 恐惧, 厌恶, 惊讶]
            # 合并平静→中性，输出7类: [中性, 快乐, 悲伤, 愤怒, 恐惧, 厌恶, 惊讶]
            neutral_merged = probabilities[0] + probabilities[1]
            merged = np.array([
                neutral_merged,       # 中性（含平静）
                probabilities[2],     # 快乐
                probabilities[3],     # 悲伤
                probabilities[4],     # 愤怒
                probabilities[5],     # 恐惧
                probabilities[6],     # 厌恶
                probabilities[7],     # 惊讶
            ], dtype=np.float32)

            merged = merged / (merged.sum() + 1e-8)
            return merged
        except Exception as e:
            print(f"语音特征提取失败: {e}")
            return np.zeros(Config.SPEECH_FEAT_DIM, dtype=np.float32)


class TextFeatureExtractor:
    """
    文本特征提取器（基于 TF-IDF + SVM）
    输出7维情感概率向量，与视觉/语音模态对齐
    """
    def __init__(self, model_dir=None):
        import pickle
        import jieba

        if model_dir is None:
            model_dir = Config.TEXT_EMOTION_MODEL_DIR

        self.available = False
        self.tfidf = None
        self.svm = None
        self._jieba = jieba

        tfidf_path = os.path.join(model_dir, 'tfidf.pkl')
        svm_path = os.path.join(model_dir, 'svm.pkl')

        print(f"[文本情感] 模型目录: {model_dir}")
        print(f"[文本情感] tfidf.pkl: {'✅' if os.path.exists(tfidf_path) else '❌'} {tfidf_path}")
        print(f"[文本情感] svm.pkl:   {'✅' if os.path.exists(svm_path) else '❌'} {svm_path}")

        if not os.path.exists(tfidf_path) or not os.path.exists(svm_path):
            print(f"⚠️  文本情感模型未找到，请检查路径是否正确")
            return

        try:
            with open(tfidf_path, 'rb') as f:
                self.tfidf = pickle.load(f)
            with open(svm_path, 'rb') as f:
                self.svm = pickle.load(f)
            self.available = True

            # 构建类别对齐映射
            # svm.classes_ 的顺序可能与 Config.CLASS_NAMES 不同
            self._class_alignment = np.zeros(Config.NUM_CLASSES, dtype=int)
            svm_classes = list(self.svm.classes_)
            svm_classes_str = [str(c) for c in svm_classes]

            for i, target_name in enumerate(Config.CLASS_NAMES):
                if target_name in svm_classes:
                    self._class_alignment[i] = svm_classes.index(target_name)
                else:
                    # 尝试 int 标签：svm.classes_ 可能是 [0,1,2,...]
                    # 此时需要知道训练时 int→emotion 的映射
                    # 优先尝试用 Config.CLASS_NAMES 的索引直接匹配
                    try:
                        idx = int(target_name)
                        if idx in svm_classes:
                            self._class_alignment[i] = svm_classes.index(idx)
                    except (ValueError, TypeError):
                        # target_name 是中文标签但 svm_classes 是整数
                        # 假设 svm.classes_ 的顺序与 Config.CLASS_NAMES 一致
                        if i < len(svm_classes):
                            self._class_alignment[i] = i

            # 预热 jieba
            jieba.initialize()
            print(f"文本情感模型加载成功 (词汇表: {len(self.tfidf.vocabulary_)} 词)")

        except Exception as e:
            print(f"文本情感模型加载失败: {e}")
            self.available = False

    def extract_from_text(self, text):
        """
        从文本提取7维情感概率特征
        返回: (7,) numpy数组，顺序与 Config.CLASS_NAMES 一致
        """
        if not self.available:
            return np.zeros(Config.TEXT_FEAT_DIM, dtype=np.float32)

        if not text or not str(text).strip() or str(text).strip() == 'nan':
            return np.zeros(Config.TEXT_FEAT_DIM, dtype=np.float32)

        text = str(text).strip()

        try:
            # jieba 分词
            words = " ".join(self._jieba.cut(text))

            # TF-IDF 向量化
            X = self.tfidf.transform([words])

            # SVM 预测概率
            raw_probs = self.svm.predict_proba(X)[0]

            # 对齐到 Config.CLASS_NAMES 顺序
            aligned = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
            for i in range(Config.NUM_CLASSES):
                aligned[i] = raw_probs[self._class_alignment[i]]

            return aligned

        except Exception as e:
            print(f"文本特征提取失败: {e}")
            return np.zeros(Config.TEXT_FEAT_DIM, dtype=np.float32)


class _NoopCallback:
    """可 pickle 的空回调，替代 lambda"""
    def __call__(self, msg):
        pass


class MultimodalFeatureExtractor:
    """多模态特征提取器（聚合三类特征，懒加载）"""
    def __init__(self, on_progress=None):
        """
        on_progress: callable(step_name)  每完成一个子模块加载时回调
        """
        self._on_progress = on_progress or _NoopCallback()
        self.visual_extractor = None
        self.speech_extractor = None
        self.text_extractor = None
        self._visual_loaded = False
        self._speech_loaded = False
        self._text_loaded = False

    def _ensure_visual(self):
        if not self._visual_loaded:
            self._on_progress("正在加载视觉模型 (YOLO)...")
            self.visual_extractor = VisualFeatureExtractor()
            self._visual_loaded = True

    def _ensure_speech(self):
        if not self._speech_loaded:
            self._on_progress("正在加载语音模型 (Wav2Vec2)...")
            try:
                self.speech_extractor = SpeechFeatureExtractor()
            except Exception as e:
                print(f"[语音] 加载失败，跳过: {e}")
                self.speech_extractor = None
            self._speech_loaded = True

    def _ensure_text(self):
        if not self._text_loaded:
            self._on_progress("正在加载文本模型 (TF-IDF+SVM)...")
            try:
                self.text_extractor = TextFeatureExtractor()
            except Exception as e:
                print(f"[文本] 加载失败，跳过: {e}")
                self.text_extractor = None
            self._text_loaded = True

    def extract(self, video_path=None, audio_path=None, text=None, on_progress=None):
        """
        on_progress: callable(step_name)  特征提取阶段的进度回调
        """
        prog = on_progress or _NoopCallback()
        features = {}

        if video_path is not None and os.path.exists(video_path):
            self._ensure_visual()
            prog("正在提取视觉特征...")
            if self.visual_extractor and self.visual_extractor.model_loaded:
                features['visual'] = self.visual_extractor.extract_from_video(video_path)
            else:
                features['visual'] = np.zeros(Config.VISUAL_FEAT_DIM, dtype=np.float32)
        else:
            features['visual'] = np.zeros(Config.VISUAL_FEAT_DIM, dtype=np.float32)

        if audio_path is not None and os.path.exists(audio_path):
            self._ensure_speech()
            prog("正在提取语音特征...")
            if self.speech_extractor:
                features['speech'] = self.speech_extractor.extract_from_audio(audio_path)
            else:
                features['speech'] = np.zeros(Config.SPEECH_FEAT_DIM, dtype=np.float32)
        else:
            features['speech'] = np.zeros(Config.SPEECH_FEAT_DIM, dtype=np.float32)

        if text is not None and str(text).strip() and str(text).strip() != 'nan':
            self._ensure_text()
            prog("正在提取文本特征...")
            if self.text_extractor:
                features['text'] = self.text_extractor.extract_from_text(text)
            else:
                features['text'] = np.zeros(Config.TEXT_FEAT_DIM, dtype=np.float32)
        else:
            features['text'] = np.zeros(Config.TEXT_FEAT_DIM, dtype=np.float32)

        return features

    def extract_concat(self, **kwargs):
        features = self.extract(**kwargs)
        return np.concatenate([features['visual'], features['speech'], features['text']])


if __name__ == "__main__":
    extractor = MultimodalFeatureExtractor()
    print("特征提取器初始化完成")
    dummy_features = extractor.extract()
    for mod, feat in dummy_features.items():
        print(f"{mod}: shape={feat.shape}, sum={feat.sum():.2f}")