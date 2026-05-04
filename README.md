# 多模态情感识别实时系统 (Multi-modal Affect Real-time System)

基于 **Transformer 跨模态对齐** 的实时多模态情感识别系统，集成 **面部表情**、**语音** 和 **文本** 三种模态，提供桌面级 GUI 交互体验。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         PyQt5 GUI 桌面应用                       │
├─────────────┬──────────────────┬────────────────┬───────────────┤
│  实时识别    │    录制识别      │   上传视频识别  │   结果展示     │
│  (摄像头+)  │  (录制后分析)     │  (文件分析)     │   (雷达图+     │
│   麦克风)   │                  │                │   概率条)      │
├──────┴──────┴──────────────────┴────────────────┴───────────────┤
│                     多模态 Transformer 融合                      │
│              (Cross-modal Transformer Fusion)                   │
├──────────────┬──────────────────┬───────────────────────────────┤
│  视觉模态    │   语音模态       │   文本模态                      │
│  YOLOv11n    │   Wav2Vec2      │   TF-IDF + SVM                 │
│  (人脸检测+  │   (情感识别)     │   (情感分类)                    │
│   表情分类)  │                  │                                │
├──────────────┴──────────────────┴───────────────────────────────┤
│                     FunASR Paraformer (语音转文本)               │
└─────────────────────────────────────────────────────────────────┘
```

## 功能特点

- **三种识别模式**：实时摄像头识别、录制后识别、上传视频文件识别
- **三模态融合**：面部表情 + 语音语调 + 文本内容
- **跨模态 Transformer**：基于 [CLS] 标记的跨模态注意力对齐
- **实时可视化**：情感概率雷达图、实时概率条形图、摄像头画面标注
- **7 类情感**：中性、快乐、悲伤、愤怒、恐惧、厌恶、惊讶

## 环境要求

- Python 3.9+
- CUDA (推荐) 或 CPU
- 摄像头 + 麦克风（实时识别模式）

## 安装

```bash
# 克隆仓库
git clone https://github.com/xiaci77/Multi-modal-Affect-Real-time-System.git
cd Multi-modal-Affect-Real-time-System

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 启动桌面应用

```bash
python app.py
```

### 命令行推理

```bash
python inference.py --video demo.mp4 --audio demo.wav --text "今天天气真好"
```

## 模型说明

| 模态 | 模型 | 功能 | 权重文件 |
|------|------|------|----------|
| 视觉 | YOLOv11n-face | 人脸检测 | `external/face_emotion/models/yolov11n-人脸检测权重.pt` |
| 视觉 | YOLO11s-cls | 表情分类 | `external/face_emotion/models/yolo11s-表情权重.pt` |
| 语音 | Wav2Vec2 | 语音情感识别 | `external/voice_emotion/wav2vec2_emotion_model.pth` |
| 文本 | TF-IDF + SVM | 文本情感分类 | `external/text_emotion/models/{svm.pkl, tfidf.pkl}` |
| 融合 | Cross-modal Transformer | 三模态融合 | `checkpoints/multimodal_transformer.pth` |

> 注意: `external/voice_emotion/wav2vec2_emotion_model.pth` (361MB) 超过 GitHub 100MB 限制，需通过 Git LFS 下载或自行训练。

## 项目结构

```
├── app.py                          # PyQt5 桌面 GUI 主程序
├── config.py                       # 全局配置文件
├── inference.py                    # 推理接口
├── feature_extractors.py           # 三模态特征提取器
├── transformer_fusion.py           # Transformer 融合模型
├── asr_transcribe_local.py         # 本地 ASR 语音转录
├── checkpoints/                    # 融合模型权重
│   └── multimodal_transformer.pth
├── external/                       # 外部单模态模型
│   ├── face_emotion/models/        # YOLO 视觉模型
│   ├── voice_emotion/              # Wav2Vec2 语音模型
│   └── text_emotion/models/        # TF-IDF + SVM 文本模型
└── requirements.txt                # Python 依赖
```

## 技术栈

- **深度学习框架**: PyTorch
- **桌面 GUI**: PyQt5
- **视觉**: Ultralytics YOLOv11, OpenCV
- **语音**: Wav2Vec2 (HuggingFace), Librosa
- **文本**: Scikit-learn, Jieba
- **语音转文本**: FunASR Paraformer
