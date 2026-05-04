"""
多模态情感识别配置文件
"""
import os
import torch

class MultimodalConfig:
    # 路径配置
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "Multimodal", "mm-process")
    AUDIO_CSV = f"{DATASET_DIR}/audio.csv"
    VIDEO_CSV = f"{DATASET_DIR}/video.csv"
    TRANSCRIPTION_CSV = f"{DATASET_DIR}/transcription.csv"
    MM_CSV = f"{DATASET_DIR}/mm.csv"

    # 标签文件
    MM_LABEL_NPZ = f"{DATASET_DIR}/mm_label.npz"

    # YOLO 视觉模型路径
    YOLO_MODELS_DIR = os.path.join(os.path.dirname(__file__), "external", "face_emotion", "models")
    YOLO_DETECT_MODEL = os.path.join(YOLO_MODELS_DIR, "yolov11n-人脸检测权重.pt")
    YOLO_CLS_MODEL = os.path.join(YOLO_MODELS_DIR, "yolo11s-表情权重.pt")

    # 语音模型路径
    SPEECH_PROJECT_DIR = os.path.join(os.path.dirname(__file__), "external", "voice_emotion")
    SPEECH_MODEL_PATH = os.path.join(SPEECH_PROJECT_DIR, "wav2vec2_emotion_model.pth")
    SPEECH_PROCESSOR_PATH = os.path.join(SPEECH_PROJECT_DIR, "wav2vec2_processor")

    # 文本情感模型路径（TF-IDF + SVM）
    TEXT_EMOTION_MODEL_DIR = os.path.join(os.path.dirname(__file__), "external", "text_emotion", "models")

    # 特征维度
    VISUAL_FEAT_DIM = 7  # YOLO 表情分类模型输出7类情感概率
    SPEECH_FEAT_DIM = 7  # Wav2Vec2 原始8种，合并平静→中性后为7种
    TEXT_FEAT_DIM = 7    # TF-IDF + SVM 输出7类情感概率（与视觉/语音对齐）

    # 融合模型参数
    HIDDEN_DIM = 64      # 公共隐藏维度
    NUM_HEADS = 4        # Transformer头数
    NUM_LAYERS = 2       # Transformer层数
    DROPOUT = 0.1

    # 分类
    NUM_CLASSES = 7      # 情感类别数（与面部情感识别一致）
    CLASS_NAMES = ['中性', '快乐', '悲伤', '愤怒', '恐惧', '厌恶', '惊讶']

    # 训练参数
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 40
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 保存路径
    SAVE_DIR = "./checkpoints"
    MODEL_NAME = "multimodal_transformer.pth"