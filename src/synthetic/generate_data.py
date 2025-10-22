"""
合成步态数据生成模块
使用生成对抗网络(GAN)和变分自编码器(VAE)生成多样化的步态数据
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
import os
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaitGAN(nn.Module):
    """用于生成步态数据的生成对抗网络"""
    
    def __init__(self, 
                 latent_dim: int = 100,
                 sequence_length: int = 100,
                 feature_dim: int = 6,
                 hidden_dim: int = 128):
        """
        初始化GAN
        
        Args:
            latent_dim: 潜在空间维度
            sequence_length: 序列长度
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
        """
        super(GaitGAN, self).__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 生成器
        self.generator = self._build_generator()
        
        # 判别器
        self.discriminator = self._build_discriminator()
    
    def _build_generator(self) -> nn.Module:
        """构建生成器网络"""
        generator = nn.Sequential(
            # 输入层
            nn.Linear(self.latent_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            # 隐藏层1
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            
            # 隐藏层2
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Linear(self.hidden_dim * 2, self.sequence_length * self.feature_dim),
            nn.Tanh()
        )
        
        return generator
    
    def _build_discriminator(self) -> nn.Module:
        """构建判别器网络"""
        discriminator = nn.Sequential(
            # 输入层
            nn.Linear(self.sequence_length * self.feature_dim, self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # 隐藏层1
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # 隐藏层2
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # 输出层
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        return discriminator
    
    def forward_generator(self, noise: torch.Tensor) -> torch.Tensor:
        """生成器前向传播"""
        fake_data = self.generator(noise)
        return fake_data.view(-1, self.sequence_length, self.feature_dim)
    
    def forward_discriminator(self, data: torch.Tensor) -> torch.Tensor:
        """判别器前向传播"""
        data_flat = data.view(data.size(0), -1)
        return self.discriminator(data_flat)


class GaitVAE(nn.Module):
    """用于生成步态数据的变分自编码器"""
    
    def __init__(self,
                 sequence_length: int = 100,
                 feature_dim: int = 6,
                 latent_dim: int = 32,
                 hidden_dim: int = 128):
        """
        初始化VAE
        
        Args:
            sequence_length: 序列长度
            feature_dim: 特征维度
            latent_dim: 潜在空间维度
            hidden_dim: 隐藏层维度
        """
        super(GaitVAE, self).__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        input_dim = sequence_length * feature_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # 潜在空间参数
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Tanh()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码器"""
        x_flat = x.view(x.size(0), -1)
        h = self.encoder(x_flat)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码器"""
        x_recon = self.decoder(z)
        return x_recon.view(-1, self.sequence_length, self.feature_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class ConditionalGaitGAN(nn.Module):
    """条件生成对抗网络，可以根据疾病类型生成特定的步态模式"""
    
    def __init__(self,
                 latent_dim: int = 100,
                 sequence_length: int = 100,
                 feature_dim: int = 6,
                 num_conditions: int = 4,  # 健康、帕金森、脑瘫、脑卒中
                 hidden_dim: int = 128):
        """
        初始化条件GAN
        
        Args:
            latent_dim: 潜在空间维度
            sequence_length: 序列长度
            feature_dim: 特征维度
            num_conditions: 条件数量（疾病类型）
            hidden_dim: 隐藏层维度
        """
        super(ConditionalGaitGAN, self).__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_conditions = num_conditions
        self.hidden_dim = hidden_dim
        
        # 条件嵌入层
        self.condition_embedding = nn.Embedding(num_conditions, latent_dim)
        
        # 生成器
        self.generator = self._build_conditional_generator()
        
        # 判别器
        self.discriminator = self._build_conditional_discriminator()
    
    def _build_conditional_generator(self) -> nn.Module:
        """构建条件生成器"""
        generator = nn.Sequential(
            # 输入: 噪声 + 条件嵌入
            nn.Linear(self.latent_dim * 2, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            nn.Linear(self.hidden_dim * 2, self.sequence_length * self.feature_dim),
            nn.Tanh()
        )
        
        return generator
    
    def _build_conditional_discriminator(self) -> nn.Module:
        """构建条件判别器"""
        discriminator = nn.Sequential(
            # 输入: 数据 + 条件嵌入
            nn.Linear(self.sequence_length * self.feature_dim + self.latent_dim, 
                     self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        return discriminator
    
    def forward_generator(self, noise: torch.Tensor, 
                         conditions: torch.Tensor) -> torch.Tensor:
        """条件生成器前向传播"""
        # 获取条件嵌入
        condition_embed = self.condition_embedding(conditions)
        
        # 拼接噪声和条件
        gen_input = torch.cat([noise, condition_embed], dim=1)
        
        # 生成数据
        fake_data = self.generator(gen_input)
        return fake_data.view(-1, self.sequence_length, self.feature_dim)
    
    def forward_discriminator(self, data: torch.Tensor, 
                            conditions: torch.Tensor) -> torch.Tensor:
        """条件判别器前向传播"""
        # 获取条件嵌入
        condition_embed = self.condition_embedding(conditions)
        
        # 拼接数据和条件
        data_flat = data.view(data.size(0), -1)
        disc_input = torch.cat([data_flat, condition_embed], dim=1)
        
        return self.discriminator(disc_input)


class SyntheticDataGenerator:
    """合成数据生成器管理类"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化合成数据生成器
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.models = {}
        
        # 疾病类型映射
        self.condition_map = {
            'healthy': 0,
            'parkinsons': 1,
            'cerebral_palsy': 2,
            'stroke': 3
        }
        
        logger.info(f"Initialized SyntheticDataGenerator on {device}")
    
    def train_gan(self, 
                  real_data: np.ndarray,
                  model_type: str = 'conditional',
                  epochs: int = 1000,
                  batch_size: int = 32,
                  lr: float = 0.0002) -> Dict:
        """
        训练GAN模型
        
        Args:
            real_data: 真实数据
            model_type: 模型类型 ('basic' 或 'conditional')
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            
        Returns:
            训练历史
        """
        logger.info(f"Training {model_type} GAN...")
        
        # 初始化模型
        if model_type == 'conditional':
            model = ConditionalGaitGAN().to(self.device)
        else:
            model = GaitGAN().to(self.device)
        
        # 优化器
        g_optimizer = optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # 损失函数
        criterion = nn.BCELoss()
        
        # 训练数据
        real_tensor = torch.FloatTensor(real_data).to(self.device)
        
        # 训练历史
        history = {'g_loss': [], 'd_loss': []}
        
        for epoch in range(epochs):
            # 判别器训练
            d_optimizer.zero_grad()
            
            # 真实数据
            batch_size_current = min(batch_size, len(real_data))
            indices = np.random.choice(len(real_data), batch_size_current, replace=False)
            real_batch = real_tensor[indices]
            
            real_labels = torch.ones(batch_size_current, 1).to(self.device)
            fake_labels = torch.zeros(batch_size_current, 1).to(self.device)
            
            # 生成假数据
            noise = torch.randn(batch_size_current, model.latent_dim).to(self.device)
            
            if model_type == 'conditional':
                # 随机选择条件
                conditions = torch.randint(0, len(self.condition_map), 
                                         (batch_size_current,)).to(self.device)
                fake_batch = model.forward_generator(noise, conditions)
                
                # 判别器损失
                real_pred = model.forward_discriminator(real_batch, conditions)
                fake_pred = model.forward_discriminator(fake_batch.detach(), conditions)
            else:
                fake_batch = model.forward_generator(noise)
                real_pred = model.forward_discriminator(real_batch)
                fake_pred = model.forward_discriminator(fake_batch.detach())
            
            d_real_loss = criterion(real_pred, real_labels)
            d_fake_loss = criterion(fake_pred, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            
            d_loss.backward()
            d_optimizer.step()
            
            # 生成器训练
            g_optimizer.zero_grad()
            
            noise = torch.randn(batch_size_current, model.latent_dim).to(self.device)
            
            if model_type == 'conditional':
                conditions = torch.randint(0, len(self.condition_map), 
                                         (batch_size_current,)).to(self.device)
                fake_batch = model.forward_generator(noise, conditions)
                fake_pred = model.forward_discriminator(fake_batch, conditions)
            else:
                fake_batch = model.forward_generator(noise)
                fake_pred = model.forward_discriminator(fake_batch)
            
            g_loss = criterion(fake_pred, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
            # 记录损失
            history['g_loss'].append(g_loss.item())
            history['d_loss'].append(d_loss.item())
            
            if epoch % 100 == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] - "
                          f"G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
        
        # 保存模型
        self.models[model_type] = model
        
        return history
    
    def train_vae(self,
                  real_data: np.ndarray,
                  epochs: int = 1000,
                  batch_size: int = 32,
                  lr: float = 0.001) -> Dict:
        """
        训练VAE模型
        
        Args:
            real_data: 真实数据
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            
        Returns:
            训练历史
        """
        logger.info("Training VAE...")
        
        # 初始化模型
        model = GaitVAE().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 训练数据
        real_tensor = torch.FloatTensor(real_data).to(self.device)
        
        # 训练历史
        history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 随机批次
            batch_size_current = min(batch_size, len(real_data))
            indices = np.random.choice(len(real_data), batch_size_current, replace=False)
            real_batch = real_tensor[indices]
            
            # 前向传播
            recon_batch, mu, logvar = model(real_batch)
            
            # 计算损失
            recon_loss = nn.MSELoss()(recon_batch, real_batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + 0.001 * kl_loss  # KL权重
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 记录损失
            history['loss'].append(total_loss.item())
            history['recon_loss'].append(recon_loss.item())
            history['kl_loss'].append(kl_loss.item())
            
            if epoch % 100 == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] - "
                          f"Total Loss: {total_loss.item():.4f}, "
                          f"Recon Loss: {recon_loss.item():.4f}, "
                          f"KL Loss: {kl_loss.item():.4f}")
        
        # 保存模型
        self.models['vae'] = model
        
        return history
    
    def generate_synthetic_data(self,
                               num_samples: int = 1000,
                               condition: str = 'healthy',
                               model_type: str = 'conditional') -> np.ndarray:
        """
        生成合成数据
        
        Args:
            num_samples: 生成样本数量
            condition: 疾病条件
            model_type: 使用的模型类型
            
        Returns:
            合成数据
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")
        
        model = self.models[model_type]
        model.eval()
        
        synthetic_data = []
        
        with torch.no_grad():
            batch_size = 100
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                current_batch_size = min(batch_size, 
                                       num_samples - batch_idx * batch_size)
                
                if model_type == 'conditional':
                    noise = torch.randn(current_batch_size, model.latent_dim).to(self.device)
                    conditions = torch.full((current_batch_size,), 
                                          self.condition_map[condition]).to(self.device)
                    fake_data = model.forward_generator(noise, conditions)
                elif model_type == 'vae':
                    z = torch.randn(current_batch_size, model.latent_dim).to(self.device)
                    fake_data = model.decode(z)
                else:  # basic GAN
                    noise = torch.randn(current_batch_size, model.latent_dim).to(self.device)
                    fake_data = model.forward_generator(noise)
                
                synthetic_data.append(fake_data.cpu().numpy())
        
        return np.concatenate(synthetic_data, axis=0)
    
    def save_models(self, save_dir: str):
        """保存训练好的模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
    
    def load_models(self, save_dir: str, model_configs: Dict):
        """加载预训练模型"""
        for model_name, config in model_configs.items():
            model_path = os.path.join(save_dir, f"{model_name}_model.pth")
            
            if os.path.exists(model_path):
                if model_name == 'conditional':
                    model = ConditionalGaitGAN(**config).to(self.device)
                elif model_name == 'vae':
                    model = GaitVAE(**config).to(self.device)
                else:
                    model = GaitGAN(**config).to(self.device)
                
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.models[model_name] = model
                logger.info(f"Loaded {model_name} model from {model_path}")


def create_sample_training_data():
    """
    创建示例训练数据 - 统一相对运动幅度模型
    
    已改进为统一的6维相对运动幅度模型：
    - 维度1-2: 左腿前后/左右运动
    - 维度3: 左腿垂直运动  
    - 维度4-5: 右腿前后/左右运动
    - 维度6: 右腿垂直运动
    
    所有维度均为标准化的相对运动幅度[-1,1]，消除了原始混合物理量的问题
    """
    # 生成多种疾病类型的模拟数据
    conditions = ['healthy', 'parkinsons', 'cerebral_palsy', 'stroke']
    all_data = []
    all_labels = []
    
    for i, condition in enumerate(conditions):
        for _ in range(100):  # 每种条件100个样本
            # 基础步态信号
            t = np.linspace(0, 1, 100)
            
            if condition == 'healthy':
                # 健康步态：规律、对称相对运动幅度
                signal = np.column_stack([
                    np.sin(2 * np.pi * t) + 0.05 * np.random.randn(100),           # 左腿前后
                    np.sin(2 * np.pi * t + np.pi) + 0.05 * np.random.randn(100),   # 左腿左右
                    0.5 * np.sin(4 * np.pi * t) + 0.03 * np.random.randn(100),     # 左腿垂直
                    np.sin(2 * np.pi * t + np.pi) + 0.05 * np.random.randn(100),   # 右腿前后
                    np.sin(2 * np.pi * t) + 0.05 * np.random.randn(100),           # 右腿左右
                    0.5 * np.sin(4 * np.pi * t + np.pi) + 0.03 * np.random.randn(100)  # 右腿垂直
                ])
            elif condition == 'parkinsons':
                # 帕金森：步幅减小、震颤、相对运动幅度
                tremor = 0.15 * np.sin(20 * np.pi * t)  # 震颤成分
                signal = np.column_stack([
                    0.7 * np.sin(2 * np.pi * t) + tremor + 0.1 * np.random.randn(100),      # 左腿前后
                    0.7 * np.sin(2 * np.pi * t + np.pi) + tremor + 0.1 * np.random.randn(100),  # 左腿左右
                    0.35 * np.sin(4 * np.pi * t) + 0.5 * tremor + 0.08 * np.random.randn(100),  # 左腿垂直
                    0.7 * np.sin(2 * np.pi * t + np.pi) + tremor + 0.1 * np.random.randn(100),  # 右腿前后
                    0.7 * np.sin(2 * np.pi * t) + tremor + 0.1 * np.random.randn(100),      # 右腿左右
                    0.35 * np.sin(4 * np.pi * t + np.pi) + 0.5 * tremor + 0.08 * np.random.randn(100)  # 右腿垂直
                ])
            elif condition == 'cerebral_palsy':
                # 脑瘫：不对称、痉挛性相对运动幅度
                spasticity = 0.2 * np.sin(6 * np.pi * t)  # 痉挛成分
                signal = np.column_stack([
                    np.sin(2 * np.pi * t) + spasticity + 0.15 * np.random.randn(100),       # 左腿前后
                    0.6 * np.sin(2 * np.pi * t + np.pi) + 0.1 * np.random.randn(100),      # 左腿左右(不对称)
                    0.4 * np.sin(4 * np.pi * t) + 0.15 * np.sin(8 * np.pi * t) + 0.1 * np.random.randn(100),  # 左腿垂直
                    0.8 * np.sin(2 * np.pi * t + np.pi) + 0.7 * spasticity + 0.15 * np.random.randn(100),  # 右腿前后
                    0.4 * np.sin(2 * np.pi * t) + 0.1 * np.random.randn(100),              # 右腿左右(严重不对称)
                    0.3 * np.sin(4 * np.pi * t + np.pi) + 0.1 * np.random.randn(100)       # 右腿垂直
                ])
            else:  # stroke
                # 脑卒中：偏瘫相对运动幅度
                signal = np.column_stack([
                    0.8 * np.sin(2 * np.pi * t) + 0.08 * np.random.randn(100),              # 左腿前后(健侧)
                    0.8 * np.sin(2 * np.pi * t + np.pi) + 0.08 * np.random.randn(100),      # 左腿左右(健侧)
                    0.4 * np.sin(4 * np.pi * t) + 0.06 * np.random.randn(100),              # 左腿垂直(健侧)
                    0.3 * np.sin(2 * np.pi * t + np.pi) + 0.2 * np.random.randn(100),       # 右腿前后(患侧)
                    0.2 * np.sin(2 * np.pi * t) + 0.25 * np.random.randn(100),              # 右腿左右(患侧)
                    0.15 * np.sin(4 * np.pi * t + np.pi) + 0.15 * np.random.randn(100)      # 右腿垂直(患侧)
                ])
            
            all_data.append(signal)
            all_labels.append(i)
    
    return np.array(all_data), np.array(all_labels)


if __name__ == "__main__":
    # 创建示例数据
    logger.info("Creating sample training data...")
    data, labels = create_sample_training_data()
    
    # 初始化生成器
    generator = SyntheticDataGenerator()
    
    # 训练模型
    logger.info("Training conditional GAN...")
    gan_history = generator.train_gan(data, model_type='conditional', epochs=500)
    
    logger.info("Training VAE...")
    vae_history = generator.train_vae(data, epochs=500)
    
    # 生成合成数据
    logger.info("Generating synthetic data...")
    synthetic_healthy = generator.generate_synthetic_data(
        num_samples=200, condition='healthy', model_type='conditional')
    synthetic_parkinsons = generator.generate_synthetic_data(
        num_samples=200, condition='parkinsons', model_type='conditional')
    
    # 保存结果
    os.makedirs('/home/ghr/5412/data/synthetic', exist_ok=True)
    np.save('/home/ghr/5412/data/synthetic/synthetic_healthy.npy', synthetic_healthy)
    np.save('/home/ghr/5412/data/synthetic/synthetic_parkinsons.npy', synthetic_parkinsons)
    
    # 保存模型
    generator.save_models('/home/ghr/5412/models')
    
    logger.info("Synthetic data generation completed!")
    logger.info(f"Generated {len(synthetic_healthy)} healthy samples")
    logger.info(f"Generated {len(synthetic_parkinsons)} Parkinson's samples")
