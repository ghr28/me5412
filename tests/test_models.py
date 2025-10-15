"""
测试模型训练模块
"""

import unittest
import numpy as np
import sys
import os

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    from models.train import (
        GaitDataset,
        TCNClassifier,
        TransformerClassifier,
        HybridGaitClassifier,
        GaitModelTrainer
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestModelTraining(unittest.TestCase):
    """测试模型训练功能"""
    
    def setUp(self):
        """测试前设置"""
        # 创建测试数据
        self.num_samples = 100
        self.sequence_length = 100
        self.feature_dim = 6
        self.num_classes = 4
        
        # 生成模拟数据
        self.data = np.random.randn(self.num_samples, self.sequence_length, self.feature_dim).astype(np.float32)
        self.labels = np.random.randint(0, self.num_classes, self.num_samples)
    
    def test_gait_dataset(self):
        """测试步态数据集"""
        dataset = GaitDataset(self.data, self.labels)
        
        self.assertEqual(len(dataset), self.num_samples)
        
        # 测试数据获取
        data_sample, label_sample = dataset[0]
        self.assertEqual(data_sample.shape, (self.sequence_length, self.feature_dim))
        self.assertIsInstance(label_sample.item(), int)
    
    def test_tcn_classifier(self):
        """测试TCN分类器"""
        model = TCNClassifier(
            input_dim=self.feature_dim,
            num_classes=self.num_classes,
            num_channels=[32, 64],
            dropout=0.1
        )
        
        # 测试前向传播
        batch_data = torch.FloatTensor(self.data[:10])
        output = model(batch_data)
        
        self.assertEqual(output.shape, (10, self.num_classes))
    
    def test_transformer_classifier(self):
        """测试Transformer分类器"""
        model = TransformerClassifier(
            input_dim=self.feature_dim,
            d_model=64,
            num_heads=4,
            num_layers=2,
            num_classes=self.num_classes,
            max_seq_length=self.sequence_length
        )
        
        # 测试前向传播
        batch_data = torch.FloatTensor(self.data[:10])
        output = model(batch_data)
        
        self.assertEqual(output.shape, (10, self.num_classes))
    
    def test_hybrid_classifier(self):
        """测试混合分类器"""
        model = HybridGaitClassifier(
            input_dim=self.feature_dim,
            num_classes=self.num_classes,
            tcn_channels=[32, 64],
            d_model=64,
            num_heads=4,
            num_transformer_layers=2
        )
        
        # 测试前向传播
        batch_data = torch.FloatTensor(self.data[:10])
        output = model(batch_data)
        
        self.assertEqual(output.shape, (10, self.num_classes))
    
    def test_model_trainer_initialization(self):
        """测试模型训练器初始化"""
        model = TCNClassifier(input_dim=self.feature_dim, num_classes=self.num_classes)
        trainer = GaitModelTrainer(model, device='cpu')
        
        self.assertEqual(trainer.device, 'cpu')
        self.assertIsNotNone(trainer.model)
        self.assertIsInstance(trainer.training_history, dict)


class TestWithoutTorch(unittest.TestCase):
    """不需要PyTorch的测试"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        # 创建测试数据
        data = np.random.randn(50, 100, 6)
        labels = np.random.randint(0, 4, 50)
        
        # 基本的数据验证
        self.assertEqual(data.shape, (50, 100, 6))
        self.assertEqual(labels.shape, (50,))
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels <= 3))


if __name__ == '__main__':
    unittest.main()
