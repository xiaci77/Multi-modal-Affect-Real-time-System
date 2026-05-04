# model.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from config import Config


class Wav2Vec2EmotionModel(nn.Module):
    """基于Wav2Vec2的情感识别模型"""

    def __init__(self, model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES):
        super().__init__()

        # 加载预训练的Wav2Vec2模型
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        # 冻结Wav2Vec2的前几层（可选）
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(Config.HIDDEN_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_values, attention_mask=None):
        # 通过Wav2Vec2提取特征
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)

        # 取最后一个隐藏状态的平均值作为特征
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)

        # 分类
        logits = self.classifier(pooled_output)

        return logits

    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'model_name': Config.MODEL_NAME,
                'num_classes': Config.NUM_CLASSES
            }
        }, path)
        print(f"模型已保存到: {path}")

    @classmethod
    def load(cls, path, device=Config.DEVICE):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            model_name=checkpoint['config']['model_name'],
            num_classes=checkpoint['config']['num_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model