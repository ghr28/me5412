# 真实病人数据放置指南

## 📂 数据目录结构（如果有真实数据的话）

```
data/raw/
├── normal/                 # 健康对照组步态数据
│   ├── patient001_normal_20241012_trial1.csv
│   ├── patient002_normal_20241012_trial1.csv
│   └── ...
├── parkinson/             # 帕金森病患者数据  
│   ├── patient101_parkinson_20241012_trial1.csv
│   ├── patient102_parkinson_20241012_trial1.csv
│   └── ...
├── cerebral_palsy/        # 脑瘫患者数据
│   ├── patient201_cerebral_palsy_20241012_trial1.csv
│   └── ...
├── stroke/                # 脑卒中患者数据
│   ├── patient301_stroke_20241012_trial1.csv
│   └── ...
└── metadata/              # 患者元数据信息
    └── patients_info.json
```

## 📊 真实数据格式要求

### CSV文件格式示例：
```csv
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,joint_angle1,joint_angle2,...
0.01,0.2,9.8,0.1,0.05,0.02,0.03,45.2,120.5,...
0.02,0.3,9.7,0.2,0.06,0.01,0.04,46.1,121.2,...
...
```

### 数据特征说明：
- **时间戳**: 采样时间点
- **加速度**: IMU传感器xyz轴加速度 (m/s²)
- **角速度**: IMU传感器xyz轴角速度 (rad/s)  
- **关节角度**: 髋关节、膝关节、踝关节角度 (度)
- **压力数据**: 足底压力分布 (可选)
- **肌电信号**: EMG信号 (可选)

## 🏥 数据收集要求

### 设备要求：
- IMU传感器 (加速度计+陀螺仪)
- 步态分析仪或压力传感器
- 采样频率: 100-1000Hz
- 步行时长: 30-60秒

### 伦理要求：
- 患者知情同意
- 数据匿名化处理
- 医院伦理委员会批准
- 遵循HIPAA等隐私法规

## ⚠️ 当前状态
**目前项目没有真实病人数据，使用数学模拟数据进行算法验证。**
