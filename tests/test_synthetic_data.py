"""
测试合成数据生成模块
"""

import unittest
import numpy as np
import torch
import sys
import os

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from synthetic.generate_data import (
    SyntheticDataGenerator,
    GaitGAN,
    GaitVAE,
    ConditionalGaitGAN,
    create_sample_training_data
)


class TestSyntheticDataGeneration(unittest.TestCase):
    """测试合成数据生成功能"""
    
    def setUp(self):
        """测试前设置"""
        self.device = 'cpu'  # 使用CPU进行测试
        self.generator = SyntheticDataGenerator(device=self.device)
        
        # 创建小的测试数据
        self.test_data, self.test_labels = create_sample_training_data()
        self.test_data = self.test_data[:50]  # 减少数据量以加快测试
        self.test_labels = self.test_labels[:50]
    
    def test_create_sample_training_data(self):
        """测试示例训练数据创建"""
        data, labels = create_sample_training_data()
        
        # 检查数据形状
        self.assertEqual(data.shape, (400, 100, 6))  # 4类 x 100样本 x 序列长度100 x 特征维度6
        self.assertEqual(labels.shape, (400,))
        
        # 检查标签范围
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels <= 3))
        
        # 检查每个类别的样本数量
        unique, counts = np.unique(labels, return_counts=True)
        self.assertEqual(len(unique), 4)
        self.assertTrue(np.all(counts == 100))
    
    def test_gait_gan_initialization(self):
        """测试GAN模型初始化"""
        model = GaitGAN(
            latent_dim=50,
            sequence_length=100,
            feature_dim=6,
            hidden_dim=64
        )
        
        # 测试生成器
        noise = torch.randn(10, 50)
        fake_data = model.forward_generator(noise)
        self.assertEqual(fake_data.shape, (10, 100, 6))
        
        # 测试判别器
        real_data = torch.randn(10, 100, 6)
        pred = model.forward_discriminator(real_data)
        self.assertEqual(pred.shape, (10, 1))
    
    def test_conditional_gan_initialization(self):
        """测试条件GAN模型初始化"""
        model = ConditionalGaitGAN(
            latent_dim=50,
            sequence_length=100,
            feature_dim=6,
            num_conditions=4,
            hidden_dim=64
        )
        
        # 测试条件生成器
        noise = torch.randn(10, 50)
        conditions = torch.randint(0, 4, (10,))
        fake_data = model.forward_generator(noise, conditions)
        self.assertEqual(fake_data.shape, (10, 100, 6))
        
        # 测试条件判别器
        real_data = torch.randn(10, 100, 6)
        pred = model.forward_discriminator(real_data, conditions)
        self.assertEqual(pred.shape, (10, 1))
    
    def test_vae_initialization(self):
        """测试VAE模型初始化"""
        model = GaitVAE(
            sequence_length=100,
            feature_dim=6,
            latent_dim=32,
            hidden_dim=64
        )
        
        # 测试VAE前向传播
        real_data = torch.randn(10, 100, 6)
        recon_data, mu, logvar = model(real_data)
        
        self.assertEqual(recon_data.shape, (10, 100, 6))
        self.assertEqual(mu.shape, (10, 32))
        self.assertEqual(logvar.shape, (10, 32))
    
    def test_synthetic_data_generator_training(self):
        """测试合成数据生成器训练（快速版本）"""
        # 使用小数据集和少数轮次进行快速测试
        small_data = self.test_data[:20]
        
        # 测试条件GAN训练
        history = self.generator.train_gan(
            small_data, 
            model_type='conditional', 
            epochs=2,  # 很少的轮次
            batch_size=10
        )
        
        self.assertIn('conditional', self.generator.models)
        self.assertIn('g_loss', history)
        self.assertIn('d_loss', history)
        self.assertEqual(len(history['g_loss']), 2)
        
        # 测试VAE训练
        vae_history = self.generator.train_vae(
            small_data,
            epochs=2,
            batch_size=10
        )
        
        self.assertIn('vae', self.generator.models)
        self.assertIn('loss', vae_history)
    
    def test_synthetic_data_generation(self):
        """测试合成数据生成"""
        # 先训练一个小模型
        small_data = self.test_data[:20]
        self.generator.train_gan(
            small_data,
            model_type='conditional',
            epochs=2,
            batch_size=10
        )
        
        # 生成合成数据
        synthetic_data = self.generator.generate_synthetic_data(
            num_samples=15,
            condition='healthy',
            model_type='conditional'
        )
        
        self.assertEqual(synthetic_data.shape, (15, 100, 6))
        self.assertTrue(np.isfinite(synthetic_data).all())
    
    def test_condition_mapping(self):
        """测试疾病条件映射"""
        expected_mapping = {
            'healthy': 0,
            'parkinsons': 1,
            'cerebral_palsy': 2,
            'stroke': 3
        }
        
        self.assertEqual(self.generator.condition_map, expected_mapping)


if __name__ == '__main__':
    unittest.main()
