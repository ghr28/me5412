# 🧪 项目测试数据位置和运行指南

## 📍 **您要找的测试数据和合成数据位置**

### 1. **测试用病人数据集生成代码**
```python
# 位置: /home/ghr/me5412/src/synthetic/generate_data.py
# 函数: create_sample_training_data() (第560行开始)

def create_sample_training_data():
    """
    🏥 这里生成测试用的病人数据集
    - 4种疾病类型：健康、帕金森、脑瘫、脑卒中  
    - 每种100个样本，总共400个测试样本
    - 数据格式：(400, 100, 6) 
      * 400个样本
      * 每个样本100个时间步（1秒数据，100Hz采样）
      * 6个特征维度（IMU传感器 + 生物力学特征）
    """
```

### 2. **现有的合成数据文件**
```bash
/home/ghr/me5412/data/synthetic/
├── demo_synth_data.npy     # 已生成的200个合成样本
└── demo_synth_labels.npy   # 对应的疾病标签
```

### 3. **主要运行入口**
```python
# 完整流水线: /home/ghr/me5412/run_pipeline.py
# 测试脚本: /home/ghr/me5412/test_project.py
```

## 🚀 **快速运行项目测试（3种方式）**

### 方式1: 运行完整演示流水线
```bash
cd /home/ghr/me5412
conda activate 5412
python run_pipeline.py
```

### 方式2: 运行项目测试脚本
```bash
cd /home/ghr/me5412
conda activate 5412  
python test_project.py
```

### 方式3: 单独测试数据生成
```bash
cd /home/ghr/me5412
conda activate 5412
python -c "
from src.synthetic.generate_data import create_sample_training_data
data, labels = create_sample_training_data()
print(f'生成测试数据: {data.shape}')
print(f'疾病标签分布: {labels}')
"
```

## 📊 **测试数据的疾病特征说明**

| 疾病类型 | 标签 | 数据特征模拟 | 样本数量 |
|---------|-----|-------------|---------|
| 健康对照 | 0 | 规律、对称的正弦波步态 | 100 |
| 帕金森病 | 1 | 步幅减小 + 5Hz震颤 | 100 |
| 脑瘫 | 2 | 不对称 + 痉挛性不规律 | 100 |
| 脑卒中 | 3 | 一侧偏瘫 + 步态不对称 | 100 |

## 🔧 **如果遇到问题**

### 问题1: 模块导入错误
```bash
# 确保在项目根目录
cd /home/ghr/me5412
export PYTHONPATH=/home/ghr/me5412:$PYTHONPATH
```

### 问题2: 缺少依赖包
```bash
conda activate 5412
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

### 问题3: 快速验证环境
```bash
cd /home/ghr/me5412
python -c "
import sys
print('Python路径:', sys.path)
try:
    from src.synthetic.generate_data import create_sample_training_data
    print('✅ 数据生成模块正常')
except Exception as e:
    print('❌ 模块导入失败:', e)
"
```

## 📈 **预期测试结果**

运行成功后您应该看到：
- ✅ 生成400个测试样本（4种疾病各100个）
- ✅ 训练GAN生成更多合成数据
- ✅ 训练混合分类模型
- ✅ 测试准确率 > 70%（演示数据）
- ✅ 保存模型和结果到 `models/` 目录

## 💡 **项目核心价值**

这个项目解决了医疗AI的核心痛点：
1. **数据稀缺** → 用GAN生成无限合成数据
2. **隐私保护** → 合成数据不包含真实患者信息  
3. **泛化能力** → 零样本学习适应新患者
4. **临床实用** → 非侵入式快速步态评估
