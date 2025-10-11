# 神经康复步态分析AI系统

## 版本信息
__version__ = "1.0.0"
__author__ = "Gait Analysis Team"
__email__ = "team@gaitanalysis.com"

## 模块导入
from .data import processor
from .models import train
from .synthetic import generate_data
from .evaluation import evaluate
# from .medical import app  # Import only when needed to avoid Flask dependency

## 公共配置
SUPPORTED_DISEASES = ['healthy', 'parkinsons', 'cerebral_palsy', 'stroke']
DISEASE_NAMES_ZH = ['健康', '帕金森病', '脑瘫', '脑卒中']

## 默认参数
DEFAULT_SEQUENCE_LENGTH = 100
DEFAULT_FEATURE_DIM = 6
DEFAULT_SAMPLING_RATE = 100

print(f"步态分析AI系统 v{__version__} 已加载")
print("支持的疾病类型:", DISEASE_NAMES_ZH)
