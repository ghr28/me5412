"""
步态分析AI模型训练模块
包含时序卷积网络(TCN)、Transformer等架构
用于步态模式识别和疾病分类
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaitDataset(Dataset):
    """步态数据集类"""
    
    def __init__(self, 
                 data: np.ndarray, 
                 labels: np.ndarray,
                 metadata: Optional[Dict] = None):
        """
        初始化数据集
        
        Args:
            data: 步态数据 (N, sequence_length, feature_dim)
            labels: 标签 (N,)
            metadata: 元数据
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.metadata = metadata or {}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TemporalConvBlock(nn.Module):
    """时序卷积块"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 dropout: float = 0.2):
        """
        初始化时序卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dilation: 膨胀率
            dropout: Dropout率
        """
        super(TemporalConvBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        original_length = x.size(2)
        
        # 第一个卷积层
        out = self.conv1(x)
        # Crop to maintain original length
        if out.size(2) > original_length:
            out = out[:, :, :original_length]
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # 第二个卷积层
        out = self.conv2(out)
        # Crop to maintain original length
        if out.size(2) > original_length:
            out = out[:, :, :original_length]
        out = self.norm2(out)
        
        # 残差连接
        if self.residual is not None:
            x = self.residual(x)
        
        # Ensure same size for residual connection
        if x.size(2) != out.size(2):
            min_len = min(x.size(2), out.size(2))
            x = x[:, :, :min_len]
            out = out[:, :, :min_len]
        
        out += x
        out = F.relu(out)
        
        return out


class TCNClassifier(nn.Module):
    """基于时序卷积网络的步态分类器"""
    
    def __init__(self,
                 input_dim: int = 6,
                 num_classes: int = 4,
                 num_channels: List[int] = [64, 64, 128, 128],
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        """
        初始化TCN分类器
        
        Args:
            input_dim: 输入特征维度
            num_classes: 分类数量
            num_channels: 每层通道数
            kernel_size: 卷积核大小
            dropout: Dropout率
        """
        super(TCNClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # TCN层
        layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TemporalConvBlock(
                in_channels, out_channels, kernel_size, dilation, dropout
            ))
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*layers)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        # 转换为 (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # TCN特征提取
        features = self.tcn(x)
        
        # 全局池化
        pooled = self.global_pool(features).squeeze(-1)
        
        # 分类
        output = self.classifier(pooled)
        
        return output


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头注意力
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerClassifier(nn.Module):
    """基于Transformer的步态分类器"""
    
    def __init__(self,
                 input_dim: int = 6,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 num_classes: int = 4,
                 max_seq_length: int = 100,
                 dropout: float = 0.1):
        """
        初始化Transformer分类器
        
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            num_classes: 分类数量
            max_seq_length: 最大序列长度
            dropout: Dropout率
        """
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        if seq_len <= self.max_seq_length:
            pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_enc
        
        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)
        
        # 分类
        output = self.classifier(x)
        
        return output


class HybridGaitClassifier(nn.Module):
    """混合模型：结合TCN和Transformer的优势"""
    
    def __init__(self,
                 input_dim: int = 6,
                 num_classes: int = 4,
                 tcn_channels: List[int] = [64, 128],
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_transformer_layers: int = 2,
                 dropout: float = 0.2):
        """
        初始化混合分类器
        
        Args:
            input_dim: 输入特征维度
            num_classes: 分类数量
            tcn_channels: TCN通道数
            d_model: Transformer模型维度
            num_heads: 注意力头数
            num_transformer_layers: Transformer层数
            dropout: Dropout率
        """
        super(HybridGaitClassifier, self).__init__()
        
        # TCN特征提取器
        tcn_layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(tcn_channels):
            dilation = 2 ** i
            tcn_layers.append(TemporalConvBlock(
                in_channels, out_channels, 3, dilation, dropout
            ))
            in_channels = out_channels
        
        self.tcn_feature_extractor = nn.Sequential(*tcn_layers)
        
        # 投影到Transformer维度
        self.tcn_to_transformer = nn.Linear(tcn_channels[-1], d_model)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        # TCN特征提取
        x_tcn = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)
        tcn_features = self.tcn_feature_extractor(x_tcn)
        tcn_features = tcn_features.transpose(1, 2)  # (batch_size, sequence_length, channels)
        
        # 投影到Transformer维度
        transformer_input = self.tcn_to_transformer(tcn_features)
        
        # Transformer处理
        for layer in self.transformer_layers:
            transformer_input = layer(transformer_input)
        
        # 全局平均池化
        pooled_features = torch.mean(transformer_input, dim=1)
        
        # 分类
        output = self.classifier(pooled_features)
        
        return output


class GaitModelTrainer:
    """步态模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        logger.info(f"Initialized trainer on {device}")
        
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              lr: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 10) -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            
        Returns:
            训练历史
        """
        # 优化器和损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            # 记录历史
            self.training_history['loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['accuracy'].append(train_accuracy)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # 输出训练信息
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f'Epoch [{epoch+1}/{epochs}] - '
                    f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - '
                    f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}'
                )
            
            # 早停
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return self.training_history
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            评估指标
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")
        
        return metrics
    
    def save_model(self, save_path: str, include_history: bool = True):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.__dict__ if hasattr(self.model, '__dict__') else {},
            'training_history': self.training_history if include_history else None
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")


def create_sample_training_pipeline():
    """创建示例训练流水线"""
    # 创建示例数据
    num_samples = 1000
    sequence_length = 100
    feature_dim = 6
    num_classes = 4
    
    # 生成模拟数据
    data = np.random.randn(num_samples, sequence_length, feature_dim)
    labels = np.random.randint(0, num_classes, num_samples)
    
    # 划分数据集
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    
    train_data, train_labels = data[:train_size], labels[:train_size]
    val_data, val_labels = data[train_size:train_size+val_size], labels[train_size:train_size+val_size]
    test_data, test_labels = data[train_size+val_size:], labels[train_size+val_size:]
    
    # 创建数据集和数据加载器
    train_dataset = GaitDataset(train_data, train_labels)
    val_dataset = GaitDataset(val_data, val_labels)
    test_dataset = GaitDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 创建示例训练流水线
    train_loader, val_loader, test_loader = create_sample_training_pipeline()
    
    # 测试不同模型
    models = {
        'TCN': TCNClassifier(input_dim=6, num_classes=4),
        'Transformer': TransformerClassifier(input_dim=6, num_classes=4),
        'Hybrid': HybridGaitClassifier(input_dim=6, num_classes=4)
    }
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name} model...")
        
        # 初始化训练器
        trainer = GaitModelTrainer(model)
        
        # 训练模型
        history = trainer.train(train_loader, val_loader, epochs=50)
        
        # 评估模型
        metrics = trainer.evaluate(test_loader)
        
        # 保存模型
        os.makedirs('/home/ghr/5412/models', exist_ok=True)
        trainer.save_model(f'/home/ghr/5412/models/{model_name.lower()}_model.pth')
        
        results[model_name] = {
            'history': history,
            'metrics': metrics
        }
    
    # 保存结果
    with open('/home/ghr/5412/models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training completed for all models!")
    
    # 输出结果比较
    print("\nModel Comparison:")
    print("-" * 50)
    for model_name, result in results.items():
        accuracy = result['metrics']['accuracy']
        f1_score = result['metrics']['f1_score']
        print(f"{model_name:12}: Accuracy: {accuracy:.4f}, F1-Score: {f1_score:.4f}")
