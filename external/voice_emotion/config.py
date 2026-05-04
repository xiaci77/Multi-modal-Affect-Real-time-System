# config.py
import torch


class Config:
    # 数据集配置
    DATA_DIR = "../../../语音识别/data/raw/Audio_Speech_Actors_01-24"
    SAMPLE_RATE = 16000
    MAX_DURATION = 5  # 秒

    # 情感标签映射
    EMOTION_MAP = {
        '01': 0, '02': 1, '03': 2, '04': 3,
        '05': 4, '06': 5, '07': 6, '08': 7
    }
    EMOTION_LABELS = [
        '中性', '平静', '快乐', '悲伤',
        '愤怒', '恐惧', '厌恶', '惊讶'
    ]

    # 模型配置
    MODEL_NAME = "facebook/wav2vec2-base"
    HIDDEN_SIZE = 768
    NUM_CLASSES = 8

    # 训练配置
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 3e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置
    MODEL_SAVE_PATH = "wav2vec2_emotion_model.pth"
    PROCESSOR_SAVE_PATH = "wav2vec2_processor"