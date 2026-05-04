"""
跨模态Transformer融合模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import MultimodalConfig as Config


class MultimodalTransformer(nn.Module):
    """多模态Transformer融合模型"""
    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # 输入特征投影到公共隐藏维度
        self.visual_proj = nn.Linear(config.VISUAL_FEAT_DIM, config.HIDDEN_DIM)
        self.speech_proj = nn.Linear(config.SPEECH_FEAT_DIM, config.HIDDEN_DIM)
        self.text_proj = nn.Linear(config.TEXT_FEAT_DIM, config.HIDDEN_DIM)

        # 可学习的[CLS]标记
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.HIDDEN_DIM) * 0.02)

        # 模态嵌入 (3种模态 + CLS)
        self.modal_embedding = nn.Embedding(4, config.HIDDEN_DIM)  # 0: CLS, 1: visual, 2: speech, 3: text
        # 位置编码 (序列长度4)
        self.pos_embedding = nn.Parameter(torch.randn(1, 4, config.HIDDEN_DIM) * 0.02)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.HIDDEN_DIM,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.HIDDEN_DIM * 4,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for module in [self.visual_proj, self.speech_proj, self.text_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.modal_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, visual_feat, speech_feat, text_feat):
        """
        前向传播
        visual_feat: (B, VISUAL_FEAT_DIM)
        speech_feat: (B, SPEECH_FEAT_DIM)
        text_feat: (B, TEXT_FEAT_DIM)
        返回: (B, NUM_CLASSES) 分类logits
        """
        batch_size = visual_feat.size(0)

        # 投影到隐藏维度
        visual_hidden = self.visual_proj(visual_feat).unsqueeze(1)  # (B, 1, H)
        speech_hidden = self.speech_proj(speech_feat).unsqueeze(1)
        text_hidden = self.text_proj(text_feat).unsqueeze(1)

        # 添加CLS标记
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, H)

        # 拼接序列: [CLS, visual, speech, text]
        sequence = torch.cat([cls_tokens, visual_hidden, speech_hidden, text_hidden], dim=1)  # (B, 4, H)

        # 添加模态嵌入
        modal_ids = torch.arange(4, device=visual_feat.device).unsqueeze(0)  # (1, 4)
        modal_emb = self.modal_embedding(modal_ids)  # (1, 4, H)
        sequence = sequence + modal_emb

        # 添加位置编码
        sequence = sequence + self.pos_embedding

        # Transformer编码
        encoded = self.transformer(sequence)  # (B, 4, H)

        # 取CLS标记输出 (第一个位置)
        cls_output = encoded[:, 0, :]  # (B, H)

        # 分类
        logits = self.classifier(cls_output)  # (B, NUM_CLASSES)
        return logits

    def predict_emotion(self, visual_feat, speech_feat, text_feat):
        """预测情绪类别和概率"""
        with torch.no_grad():
            logits = self.forward(visual_feat, speech_feat, text_feat)
            probabilities = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probabilities, dim=1)
        return pred_class, probabilities


class SimpleFusionModel(nn.Module):
    """简单拼接+全连接融合模型（基线）"""
    def __init__(self, config=Config):
        super().__init__()
        total_dim = config.VISUAL_FEAT_DIM + config.SPEECH_FEAT_DIM + config.TEXT_FEAT_DIM
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        )

    def forward(self, visual_feat, speech_feat, text_feat):
        combined = torch.cat([visual_feat, speech_feat, text_feat], dim=1)
        return self.classifier(combined)


if __name__ == "__main__":
    # 测试模型
    config = Config
    model = MultimodalTransformer(config)
    print(model)

    # 随机输入
    batch = 2
    visual = torch.randn(batch, config.VISUAL_FEAT_DIM)
    speech = torch.randn(batch, config.SPEECH_FEAT_DIM)
    text = torch.randn(batch, config.TEXT_FEAT_DIM)

    logits = model(visual, speech, text)
    print(f"输入形状: visual {visual.shape}, speech {speech.shape}, text {text.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"输出示例: {logits[0]}")