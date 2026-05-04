"""
多模态情感识别推理脚本
"""
import os
import torch
import numpy as np
import argparse
import sys
sys.path.append('.')

from config import MultimodalConfig as Config
from feature_extractors import MultimodalFeatureExtractor
from transformer_fusion import MultimodalTransformer, SimpleFusionModel


class MultimodalEmotionRecognizer:
    """多模态情感识别器"""
    def __init__(self, model_path=None, model_type='transformer'):
        self.device = Config.DEVICE
        self.feature_extractor = MultimodalFeatureExtractor()
        self.model_type = model_type

        # 加载模型
        if model_type == 'transformer':
            self.model = MultimodalTransformer(Config).to(self.device)
        else:
            self.model = SimpleFusionModel(Config).to(self.device)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 尝试加载默认路径
            default_path = os.path.join(Config.SAVE_DIR, Config.MODEL_NAME)
            if os.path.exists(default_path):
                self.load_model(default_path)
            else:
                print("警告: 未找到训练好的模型，使用随机初始化")

        self.model.eval()

    def load_model(self, model_path):
        """加载预训练模型"""
        # 使用 weights_only=False 以支持包含非张量对象的检查点
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功: {model_path}")
        if 'val_acc' in checkpoint:
            print(f"验证准确率: {checkpoint['val_acc']:.4f}")

    def predict(self, video_path=None, audio_path=None, text=None):
        """
        预测多模态情感
        参数: 至少提供一个模态
        返回: 预测情绪标签、概率分布、各模态特征
        """
        # 提取特征
        features = self.feature_extractor.extract(
            video_path=video_path,
            audio_path=audio_path,
            text=text
        )

        # 转换为tensor
        visual_tensor = torch.tensor(features['visual'], dtype=torch.float32).unsqueeze(0).to(self.device)
        speech_tensor = torch.tensor(features['speech'], dtype=torch.float32).unsqueeze(0).to(self.device)
        text_tensor = torch.tensor(features['text'], dtype=torch.float32).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            logits = self.model(visual_tensor, speech_tensor, text_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probabilities)
            pred_label = Config.CLASS_NAMES[pred_idx]

        # 构建结果
        result = {
            'predicted_emotion': pred_label,
            'confidence': float(probabilities[pred_idx]),
            'probabilities': {Config.CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(Config.CLASS_NAMES))},
            'features': {
                'visual': features['visual'].tolist(),
                'speech': features['speech'].tolist(),
                'text': features['text'].tolist()
            }
        }
        return result

    def predict_from_files(self, video_file=None, audio_file=None, text_file=None):
        """从文件预测（支持文本文件）"""
        text = None
        if text_file and os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()

        return self.predict(video_path=video_file, audio_path=audio_file, text=text)

    def print_result(self, result):
        """打印结果"""
        print("\n" + "="*50)
        print("多模态情感识别结果")
        print("="*50)
        print(f"预测情绪: {result['predicted_emotion']}")
        print(f"置信度: {result['confidence']:.1%}")
        print("\n概率分布:")
        for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            print(f"  {emotion}: {bar} {prob:.1%}")

        print("\n各模态特征:")
        for mod, feat in result['features'].items():
            print(f"  {mod}: {np.array(feat).round(3)}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='多模态情感识别推理')
    parser.add_argument('--video', type=str, help='视频文件路径 (.mp4)')
    parser.add_argument('--audio', type=str, help='音频文件路径 (.wav)')
    parser.add_argument('--text', type=str, help='文本内容或文件路径')
    parser.add_argument('--text_file', type=str, help='文本文件路径')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'simple'])
    parser.add_argument('--model_path', type=str, help='模型路径，默认使用checkpoints/multimodal_transformer.pth')
    args = parser.parse_args()

    # 确定文本内容
    text_content = args.text
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()

    # 初始化识别器
    model_path = args.model_path or os.path.join(Config.SAVE_DIR, Config.MODEL_NAME)
    recognizer = MultimodalEmotionRecognizer(model_path=model_path, model_type=args.model)

    # 预测
    result = recognizer.predict(
        video_path=args.video,
        audio_path=args.audio,
        text=text_content
    )

    # 输出
    recognizer.print_result(result)


if __name__ == "__main__":
    main()