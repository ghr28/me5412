#!/usr/bin/env python
"""
项目测试脚本 - 快速验证项目功能
无需真实数据集，使用内置的合成数据生成功能
"""

import os
import sys
import numpy as np
import torch
print("🔍 测试项目环境...")

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_synthetic_data_generation():
    """测试合成数据生成功能"""
    print("\n📊 测试1: 合成数据生成")
    
    try:
        from src.synthetic.generate_data import create_sample_training_data, SyntheticDataGenerator
        
        # 生成测试用的病人数据集
        print("  - 生成测试用病人数据集...")
        data, labels = create_sample_training_data()
        
        print(f"  ✅ 成功生成数据: {data.shape}")
        print(f"  ✅ 标签分布: {np.bincount(labels)}")
        print(f"  ✅ 疾病类型: 健康({np.sum(labels==0)}) | 帕金森({np.sum(labels==1)}) | 脑瘫({np.sum(labels==2)}) | 脑卒中({np.sum(labels==3)})")
        
        # 测试合成数据生成器
        print("  - 测试合成数据生成器...")
        generator = SyntheticDataGenerator(device='cpu')
        
        # 快速训练（少数轮次用于测试）
        print("  - 快速训练条件GAN（测试模式：5轮）...")
        history = generator.train_gan(data[:50], model_type='conditional', epochs=5, batch_size=16)
        
        print(f"  ✅ GAN训练完成，损失历史长度: {len(history['g_loss'])}")
        
        # 生成少量合成数据进行测试
        print("  - 生成测试用合成数据...")
        synthetic_data = generator.generate_synthetic_data(
            num_samples=20, condition='healthy', model_type='conditional'
        )
        
        print(f"  ✅ 合成数据生成成功: {synthetic_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 合成数据生成测试失败: {e}")
        return False

def test_model_training():
    """测试模型训练功能"""
    print("\n🤖 测试2: 模型训练")
    
    try:
        from src.models.train import HybridGaitClassifier, GaitModelTrainer, GaitDataset
        from torch.utils.data import DataLoader
        
        # 创建小规模测试数据
        print("  - 创建测试数据...")
        data = np.random.randn(100, 100, 6).astype(np.float32)  # 100个样本
        labels = np.random.randint(0, 4, 100)
        
        # 划分训练/测试数据
        train_data, train_labels = data[:80], labels[:80]
        test_data, test_labels = data[80:], labels[80:]
        
        # 创建数据加载器
        train_dataset = GaitDataset(train_data, train_labels)
        test_dataset = GaitDataset(test_data, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        print(f"  ✅ 训练数据: {len(train_dataset)} 样本")
        print(f"  ✅ 测试数据: {len(test_dataset)} 样本")
        
        # 创建混合模型
        print("  - 创建混合分类模型...")
        model = HybridGaitClassifier(input_dim=6, num_classes=4)
        trainer = GaitModelTrainer(model, device='cpu')
        
        # 快速训练（少数轮次用于测试）
        print("  - 快速训练模型（测试模式：5轮）...")
        history = trainer.train(
            train_loader, test_loader, 
            epochs=5, lr=1e-3, patience=3
        )
        
        print(f"  ✅ 模型训练完成，训练轮数: {len(history['loss'])}")
        
        # 评估模型
        print("  - 评估模型性能...")
        metrics = trainer.evaluate(test_loader)
        
        print(f"  ✅ 测试准确率: {metrics['accuracy']:.3f}")
        print(f"  ✅ F1分数: {metrics['f1_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型训练测试失败: {e}")
        return False

def test_pipeline():
    """测试完整流水线"""
    print("\n🔄 测试3: 完整流水线")
    
    try:
        # 模拟运行主流水线（简化版）
        print("  - 运行简化版主流水线...")
        
        # 1. 生成基础数据
        from src.synthetic.generate_data import create_sample_training_data
        data, labels = create_sample_training_data()
        print(f"  ✅ 步骤1: 生成基础数据 {data.shape}")
        
        # 2. 数据预处理
        data_scaled = (data - data.mean()) / (data.std() + 1e-8)
        print(f"  ✅ 步骤2: 数据标准化完成")
        
        # 3. 划分数据集
        n_train = int(0.7 * len(data_scaled))
        n_val = int(0.15 * len(data_scaled))
        
        train_data = data_scaled[:n_train]
        val_data = data_scaled[n_train:n_train+n_val]
        test_data = data_scaled[n_train+n_val:]
        
        train_labels = labels[:n_train]
        val_labels = labels[n_train:n_train+n_val]
        test_labels = labels[n_train+n_val:]
        
        print(f"  ✅ 步骤3: 数据划分 - 训练:{len(train_data)} | 验证:{len(val_data)} | 测试:{len(test_data)}")
        
        # 4. 创建数据加载器
        from src.models.train import GaitDataset
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(GaitDataset(train_data, train_labels), batch_size=32, shuffle=True)
        val_loader = DataLoader(GaitDataset(val_data, val_labels), batch_size=32, shuffle=False)
        test_loader = DataLoader(GaitDataset(test_data, test_labels), batch_size=32, shuffle=False)
        
        print(f"  ✅ 步骤4: 数据加载器创建完成")
        
        # 5. 训练模型
        from src.models.train import HybridGaitClassifier, GaitModelTrainer
        
        model = HybridGaitClassifier(input_dim=6, num_classes=4)
        trainer = GaitModelTrainer(model, device='cpu')
        
        # 快速训练
        history = trainer.train(train_loader, val_loader, epochs=3, lr=1e-3)
        print(f"  ✅ 步骤5: 模型训练完成")
        
        # 6. 评估
        metrics = trainer.evaluate(test_loader)
        print(f"  ✅ 步骤6: 最终测试准确率: {metrics['accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 完整流水线测试失败: {e}")
        return False

def test_existing_demo_data():
    """测试现有的演示数据"""
    print("\n📂 测试4: 现有演示数据")
    
    try:
        demo_data_path = os.path.join(project_root, "data", "synthetic", "demo_synth_data.npy")
        demo_labels_path = os.path.join(project_root, "data", "synthetic", "demo_synth_labels.npy")
        
        if os.path.exists(demo_data_path) and os.path.exists(demo_labels_path):
            demo_data = np.load(demo_data_path)
            demo_labels = np.load(demo_labels_path)
            
            print(f"  ✅ 加载演示数据: {demo_data.shape}")
            print(f"  ✅ 加载演示标签: {demo_labels.shape}")
            print(f"  ✅ 标签分布: {np.bincount(demo_labels)}")
            
            return True
        else:
            print("  ⚠️  演示数据文件不存在，这是正常的")
            return True
            
    except Exception as e:
        print(f"  ❌ 演示数据测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🧪 开始项目功能测试...")
    print("=" * 50)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(test_synthetic_data_generation())
    test_results.append(test_model_training())
    test_results.append(test_existing_demo_data())
    test_results.append(test_pipeline())
    
    # 总结测试结果
    print("\n" + "=" * 50)
    print("📋 测试结果总结:")
    
    test_names = [
        "合成数据生成",
        "模型训练", 
        "现有演示数据",
        "完整流水线"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {i+1}. {name}: {status}")
    
    print(f"\n🏆 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目功能正常。")
        print("\n💡 接下来您可以:")
        print("  - 运行完整训练: python run_pipeline.py")
        print("  - 添加真实数据到 data/raw/ 目录")
        print("  - 调整模型参数在 configs/train_config.yaml")
    else:
        print("⚠️  部分测试失败，请检查环境配置。")

if __name__ == "__main__":
    main()
