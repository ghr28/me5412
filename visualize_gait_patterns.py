#!/usr/bin/env python
"""
步态模式可视化分析
展示不同疾病的数学模拟步态特征差异
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from pathlib import Path

def generate_single_sample(condition, duration=1.0, sample_rate=100):
    """
    生成单个样本的步态数据
    
    Args:
        condition: 疾病类型
        duration: 时间长度(秒)
        sample_rate: 采样频率(Hz)
    
    Returns:
        时间序列数据 (time_points, 6_features)
    """
    n_points = int(duration * sample_rate)
    t = np.linspace(0, duration, n_points)
    
    # 为了一致性，使用固定随机种子
    np.random.seed(42)
    
    if condition == 'healthy':
        # 健康步态：规律、对称
        signal = np.column_stack([
            np.sin(2 * np.pi * t) + 0.1 * np.random.randn(n_points),
            np.sin(2 * np.pi * t + np.pi) + 0.1 * np.random.randn(n_points),
            0.5 * np.sin(4 * np.pi * t) + 0.05 * np.random.randn(n_points),
            0.5 * np.sin(4 * np.pi * t + np.pi) + 0.05 * np.random.randn(n_points),
            0.3 * np.cos(2 * np.pi * t) + 0.05 * np.random.randn(n_points),
            9.8 + 0.2 * np.sin(2 * np.pi * t) + 0.1 * np.random.randn(n_points)
        ])
    elif condition == 'parkinsons':
        # 帕金森：步幅减小、不规律
        signal = np.column_stack([
            0.7 * np.sin(2 * np.pi * t) + 0.2 * np.random.randn(n_points),
            0.7 * np.sin(2 * np.pi * t + np.pi) + 0.2 * np.random.randn(n_points),
            0.3 * np.sin(4 * np.pi * t) + 0.1 * np.random.randn(n_points),
            0.3 * np.sin(4 * np.pi * t + np.pi) + 0.1 * np.random.randn(n_points),
            0.2 * np.cos(2 * np.pi * t) + 0.1 * np.random.randn(n_points),
            9.8 + 0.1 * np.sin(2 * np.pi * t) + 0.15 * np.random.randn(n_points)
        ])
    elif condition == 'cerebral_palsy':
        # 脑瘫：不对称、痉挛性
        signal = np.column_stack([
            np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t) + 0.2 * np.random.randn(n_points),
            0.6 * np.sin(2 * np.pi * t + np.pi) + 0.2 * np.random.randn(n_points),
            0.4 * np.sin(4 * np.pi * t) + 0.2 * np.sin(8 * np.pi * t) + 0.1 * np.random.randn(n_points),
            0.3 * np.sin(4 * np.pi * t + np.pi) + 0.1 * np.random.randn(n_points),
            0.4 * np.cos(2 * np.pi * t) + 0.1 * np.cos(6 * np.pi * t) + 0.1 * np.random.randn(n_points),
            9.8 + 0.3 * np.sin(2 * np.pi * t) + 0.2 * np.random.randn(n_points)
        ])
    else:  # stroke
        # 脑卒中：一侧偏瘫
        signal = np.column_stack([
            0.8 * np.sin(2 * np.pi * t) + 0.15 * np.random.randn(n_points),
            0.4 * np.sin(2 * np.pi * t + np.pi) + 0.3 * np.random.randn(n_points),
            0.4 * np.sin(4 * np.pi * t) + 0.1 * np.random.randn(n_points),
            0.2 * np.sin(4 * np.pi * t + np.pi) + 0.2 * np.random.randn(n_points),
            0.3 * np.cos(2 * np.pi * t) + 0.1 * np.random.randn(n_points),
            9.8 + 0.25 * np.sin(2 * np.pi * t) + 0.15 * np.random.randn(n_points)
        ])
    
    return t, signal

def create_gait_comparison_plot():
    """创建步态模式对比图"""
    
    # 设置图形参数
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('神经康复疾病步态模式数学模拟对比', fontsize=16, fontweight='bold')
    
    # 疾病类型和颜色
    conditions = ['healthy', 'parkinsons', 'cerebral_palsy', 'stroke']
    condition_names = ['健康对照', '帕金森病', '脑瘫', '脑卒中']
    colors = ['green', 'red', 'blue', 'orange']
    
    # 特征名称
    feature_names = [
        '左腿水平运动', '右腿水平运动', 
        '左腿垂直运动', '右腿垂直运动', 
        '躯干运动', '垂直加速度'
    ]
    
    # 为每种疾病生成数据并绘图
    for i, (condition, name, color) in enumerate(zip(conditions, condition_names, colors)):
        t, signal = generate_single_sample(condition, duration=2.0)  # 2秒数据更清晰
        
        # 左列：时域信号
        ax_time = axes[i, 0]
        
        # 绘制主要特征（左右腿水平运动）
        ax_time.plot(t, signal[:, 0], label='左腿', linewidth=2, alpha=0.8)
        ax_time.plot(t, signal[:, 1], label='右腿', linewidth=2, alpha=0.8, linestyle='--')
        
        ax_time.set_title(f'{name} - 时域步态信号', fontweight='bold')
        ax_time.set_xlabel('时间 (秒)')
        ax_time.set_ylabel('运动幅度')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        
        # 右列：功率谱分析
        ax_freq = axes[i, 1]
        
        # 计算功率谱
        from scipy.signal import welch
        
        # 对左腿信号进行频谱分析
        freqs, psd = welch(signal[:, 0], fs=100, nperseg=128)
        ax_freq.semilogy(freqs, psd, color=color, linewidth=2, label='左腿')
        
        # 对右腿信号进行频谱分析
        freqs, psd = welch(signal[:, 1], fs=100, nperseg=128)
        ax_freq.semilogy(freqs, psd, color=color, linewidth=2, linestyle='--', alpha=0.7, label='右腿')
        
        ax_freq.set_title(f'{name} - 频域分析', fontweight='bold')
        ax_freq.set_xlabel('频率 (Hz)')
        ax_freq.set_ylabel('功率谱密度')
        ax_freq.legend()
        ax_freq.grid(True, alpha=0.3)
        ax_freq.set_xlim(0, 10)  # 只显示0-10Hz
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path('/home/ghr/me5412/docs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'gait_patterns_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 步态模式对比图已保存到: docs/gait_patterns_comparison.png")

def create_feature_statistics_plot():
    """创建特征统计对比图"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('不同疾病步态特征统计对比', fontsize=16, fontweight='bold')
    
    conditions = ['healthy', 'parkinsons', 'cerebral_palsy', 'stroke']
    condition_names = ['健康', '帕金森', '脑瘫', '脑卒中']
    colors = ['green', 'red', 'blue', 'orange']
    
    feature_names = [
        '左腿水平', '右腿水平', 
        '左腿垂直', '右腿垂直', 
        '躯干运动', '垂直加速度'
    ]
    
    # 生成多个样本计算统计信息
    n_samples = 50
    all_stats = {condition: {'mean': [], 'std': [], 'asymmetry': []} for condition in conditions}
    
    for condition in conditions:
        for _ in range(n_samples):
            t, signal = generate_single_sample(condition)
            
            # 计算统计特征
            feature_means = np.mean(signal, axis=0)
            feature_stds = np.std(signal, axis=0)
            
            # 计算左右不对称性（左腿vs右腿）
            left_right_asymmetry = abs(np.mean(signal[:, 0]) - np.mean(signal[:, 1]))
            
            all_stats[condition]['mean'].append(feature_means)
            all_stats[condition]['std'].append(feature_stds)
            all_stats[condition]['asymmetry'].append(left_right_asymmetry)
    
    # 转换为numpy数组
    for condition in conditions:
        all_stats[condition]['mean'] = np.array(all_stats[condition]['mean'])
        all_stats[condition]['std'] = np.array(all_stats[condition]['std'])
        all_stats[condition]['asymmetry'] = np.array(all_stats[condition]['asymmetry'])
    
    # 绘制每个特征的对比图
    axes_flat = axes.flatten()
    
    for feature_idx in range(6):
        ax = axes_flat[feature_idx]
        
        # 为每种疾病绘制箱线图数据
        feature_data = []
        labels = []
        
        for condition, name, color in zip(conditions, condition_names, colors):
            feature_values = all_stats[condition]['mean'][:, feature_idx]
            feature_data.append(feature_values)
            labels.append(name)
        
        # 绘制箱线图
        box_plot = ax.boxplot(feature_data, labels=labels, patch_artist=True)
        
        # 设置颜色
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{feature_names[feature_idx]}特征分布', fontweight='bold')
        ax.set_ylabel('特征值')
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path('/home/ghr/me5412/docs')
    plt.savefig(output_dir / 'gait_features_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 特征统计对比图已保存到: docs/gait_features_statistics.png")

def create_asymmetry_analysis():
    """创建步态不对称性分析图"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('步态不对称性分析 - 疾病特征差异', fontsize=16, fontweight='bold')
    
    conditions = ['healthy', 'parkinsons', 'cerebral_palsy', 'stroke']
    condition_names = ['健康', '帕金森', '脑瘫', '脑卒中']
    colors = ['green', 'red', 'blue', 'orange']
    
    # 计算不对称性指标
    asymmetry_data = []
    variability_data = []
    
    for condition in conditions:
        condition_asymmetry = []
        condition_variability = []
        
        for _ in range(100):  # 100个样本
            t, signal = generate_single_sample(condition)
            
            # 左右腿不对称性
            left_leg = signal[:, 0]
            right_leg = signal[:, 1]
            asymmetry = np.mean(abs(left_leg - right_leg))
            condition_asymmetry.append(asymmetry)
            
            # 步态变异性
            variability = np.std(left_leg) + np.std(right_leg)
            condition_variability.append(variability)
        
        asymmetry_data.append(condition_asymmetry)
        variability_data.append(condition_variability)
    
    # 绘制不对称性箱线图
    box1 = ax1.boxplot(asymmetry_data, labels=condition_names, patch_artist=True)
    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('步态不对称性对比', fontweight='bold')
    ax1.set_ylabel('不对称性指数')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 绘制变异性箱线图
    box2 = ax2.boxplot(variability_data, labels=condition_names, patch_artist=True)
    for patch, color in zip(box2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('步态变异性对比', fontweight='bold')
    ax2.set_ylabel('变异性指数')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path('/home/ghr/me5412/docs')
    plt.savefig(output_dir / 'gait_asymmetry_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 不对称性分析图已保存到: docs/gait_asymmetry_analysis.png")

def main():
    """主函数"""
    print("🎨 开始生成步态模式可视化分析...")
    
    # 确保scipy可用（用于频谱分析）
    try:
        import scipy.signal
    except ImportError:
        print("⚠️  警告: scipy未安装，将跳过频谱分析")
        return
    
    # 生成各种分析图
    create_gait_comparison_plot()
    create_feature_statistics_plot()
    create_asymmetry_analysis()
    
    print("\n📊 所有可视化图表已生成完成！")
    print("📁 查看文件:")
    print("  - docs/gait_patterns_comparison.png")
    print("  - docs/gait_features_statistics.png") 
    print("  - docs/gait_asymmetry_analysis.png")
    
    print("\n💡 这些图表展示了:")
    print("  1. 不同疾病的时域和频域步态特征差异")
    print("  2. 各种疾病步态参数的统计分布")
    print("  3. 步态不对称性和变异性的疾病特异性")

if __name__ == "__main__":
    main()
