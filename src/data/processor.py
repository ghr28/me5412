"""
数据预处理模块
处理原始步态数据，包括清洗、标准化、特征提取等
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaitDataProcessor:
    """步态数据预处理器"""
    
    def __init__(self, 
                 sampling_rate: int = 100,
                 filter_freq: Tuple[float, float] = (0.5, 10.0),
                 normalize_method: str = 'standard'):
        """
        初始化数据处理器
        
        Args:
            sampling_rate: 采样率 (Hz)
            filter_freq: 带通滤波频率范围 (Hz)
            normalize_method: 标准化方法 ('standard' 或 'minmax')
        """
        self.sampling_rate = sampling_rate
        self.filter_freq = filter_freq
        self.normalize_method = normalize_method
        self.scaler = StandardScaler() if normalize_method == 'standard' else MinMaxScaler()
        
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        加载原始步态数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            pandas DataFrame containing gait data
        """
        try:
            # 根据文件格式选择加载方式
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
            logger.info(f"Loaded data with shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗：处理缺失值、异常值等
        
        Args:
            data: 原始数据
            
        Returns:
            清洗后的数据
        """
        # 处理缺失值
        data = data.dropna()
        
        # 移除异常值 (使用3-sigma准则)
        for column in data.select_dtypes(include=[np.number]).columns:
            mean = data[column].mean()
            std = data[column].std()
            data = data[(data[column] >= mean - 3*std) & (data[column] <= mean + 3*std)]
        
        logger.info(f"Data shape after cleaning: {data.shape}")
        return data
    
    def apply_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """
        应用带通滤波器去除噪声
        
        Args:
            signal_data: 信号数据
            
        Returns:
            滤波后的信号
        """
        nyquist = self.sampling_rate / 2
        low = self.filter_freq[0] / nyquist
        high = self.filter_freq[1] / nyquist
        
        # 设计Butterworth带通滤波器
        b, a = signal.butter(4, [low, high], btype='band')
        
        # 应用零相位滤波
        filtered_data = signal.filtfilt(b, a, signal_data, axis=0)
        
        return filtered_data
    
    def extract_gait_cycles(self, data: np.ndarray, 
                           heel_strike_column: int = 0) -> List[np.ndarray]:
        """
        提取步态周期
        
        Args:
            data: 步态数据
            heel_strike_column: 足跟着地信号列索引
            
        Returns:
            步态周期列表
        """
        heel_strikes = self._detect_heel_strikes(data[:, heel_strike_column])
        
        gait_cycles = []
        for i in range(len(heel_strikes) - 1):
            start_idx = heel_strikes[i]
            end_idx = heel_strikes[i + 1]
            cycle = data[start_idx:end_idx]
            
            # 标准化步态周期长度为100个点
            cycle_normalized = self._normalize_cycle_length(cycle, target_length=100)
            gait_cycles.append(cycle_normalized)
        
        return gait_cycles
    
    def _detect_heel_strikes(self, heel_signal: np.ndarray, 
                            threshold: float = None) -> List[int]:
        """
        检测足跟着地事件
        
        Args:
            heel_signal: 足跟压力/加速度信号
            threshold: 检测阈值
            
        Returns:
            足跟着地时间点索引列表
        """
        if threshold is None:
            threshold = np.mean(heel_signal) + 2 * np.std(heel_signal)
        
        # 寻找峰值
        peaks, _ = signal.find_peaks(heel_signal, 
                                   height=threshold, 
                                   distance=int(0.5 * self.sampling_rate))
        
        return peaks.tolist()
    
    def _normalize_cycle_length(self, cycle: np.ndarray, 
                               target_length: int = 100) -> np.ndarray:
        """
        将步态周期标准化为固定长度
        
        Args:
            cycle: 原始步态周期数据
            target_length: 目标长度
            
        Returns:
            标准化长度的步态周期
        """
        from scipy.interpolate import interp1d
        
        original_length = len(cycle)
        if original_length == target_length:
            return cycle
        
        # 使用线性插值调整长度
        original_indices = np.linspace(0, 1, original_length)
        target_indices = np.linspace(0, 1, target_length)
        
        normalized_cycle = np.zeros((target_length, cycle.shape[1]))
        
        for col in range(cycle.shape[1]):
            f = interp1d(original_indices, cycle[:, col], kind='linear')
            normalized_cycle[:, col] = f(target_indices)
        
        return normalized_cycle
    
    def extract_features(self, gait_cycles: List[np.ndarray]) -> np.ndarray:
        """
        提取步态特征
        
        Args:
            gait_cycles: 步态周期列表
            
        Returns:
            特征矩阵
        """
        features = []
        
        for cycle in gait_cycles:
            cycle_features = []
            
            # 时域特征
            cycle_features.extend(self._extract_temporal_features(cycle))
            
            # 频域特征
            cycle_features.extend(self._extract_frequency_features(cycle))
            
            # 对称性特征
            cycle_features.extend(self._extract_symmetry_features(cycle))
            
            features.append(cycle_features)
        
        return np.array(features)
    
    def _extract_temporal_features(self, cycle: np.ndarray) -> List[float]:
        """提取时域特征"""
        features = []
        
        for col in range(cycle.shape[1]):
            signal_col = cycle[:, col]
            
            # 基本统计特征
            features.extend([
                np.mean(signal_col),
                np.std(signal_col),
                np.min(signal_col),
                np.max(signal_col),
                np.median(signal_col)
            ])
            
            # 步态相位特征
            stance_phase = np.where(signal_col > np.mean(signal_col))[0]
            if len(stance_phase) > 0:
                stance_duration = len(stance_phase) / len(signal_col)
                features.append(stance_duration)
            else:
                features.append(0.0)
        
        return features
    
    def _extract_frequency_features(self, cycle: np.ndarray) -> List[float]:
        """提取频域特征"""
        features = []
        
        for col in range(cycle.shape[1]):
            signal_col = cycle[:, col]
            
            # FFT分析
            fft = np.fft.fft(signal_col)
            power_spectrum = np.abs(fft) ** 2
            frequencies = np.fft.fftfreq(len(signal_col), 1/self.sampling_rate)
            
            # 主频率
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_freq = frequencies[dominant_freq_idx]
            features.append(dominant_freq)
            
            # 频率中心
            freq_centroid = np.sum(frequencies[:len(frequencies)//2] * 
                                 power_spectrum[:len(power_spectrum)//2]) / \
                           np.sum(power_spectrum[:len(power_spectrum)//2])
            features.append(freq_centroid)
        
        return features
    
    def _extract_symmetry_features(self, cycle: np.ndarray) -> List[float]:
        """提取对称性特征"""
        features = []
        
        # 左右对称性分析
        mid_point = len(cycle) // 2
        left_half = cycle[:mid_point]
        right_half = cycle[mid_point:]
        
        for col in range(cycle.shape[1]):
            if len(left_half) == len(right_half):
                correlation = np.corrcoef(left_half[:, col], 
                                        right_half[:, col][::-1])[0, 1]
                if not np.isnan(correlation):
                    features.append(correlation)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        return features
    
    def normalize_features(self, features: np.ndarray, 
                          fit: bool = True) -> np.ndarray:
        """
        标准化特征
        
        Args:
            features: 特征矩阵
            fit: 是否拟合缩放器
            
        Returns:
            标准化后的特征
        """
        if fit:
            normalized_features = self.scaler.fit_transform(features)
        else:
            normalized_features = self.scaler.transform(features)
        
        return normalized_features
    
    def process_pipeline(self, file_path: str, 
                        patient_info: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        完整的数据处理流水线
        
        Args:
            file_path: 数据文件路径
            patient_info: 患者信息
            
        Returns:
            处理后的特征和元数据
        """
        logger.info(f"Processing data from: {file_path}")
        
        # 1. 加载原始数据
        raw_data = self.load_raw_data(file_path)
        
        # 2. 数据清洗
        clean_data = self.clean_data(raw_data)
        
        # 3. 应用滤波器
        numeric_data = clean_data.select_dtypes(include=[np.number]).values
        filtered_data = self.apply_filter(numeric_data)
        
        # 4. 提取步态周期
        gait_cycles = self.extract_gait_cycles(filtered_data)
        
        # 5. 特征提取
        features = self.extract_features(gait_cycles)
        
        # 6. 特征标准化
        normalized_features = self.normalize_features(features)
        
        # 7. 生成元数据
        metadata = {
            'original_shape': raw_data.shape,
            'processed_shape': normalized_features.shape,
            'num_gait_cycles': len(gait_cycles),
            'sampling_rate': self.sampling_rate,
            'filter_freq': self.filter_freq,
            'patient_info': patient_info or {}
        }
        
        logger.info(f"Processing completed. Features shape: {normalized_features.shape}")
        
        return normalized_features, metadata


def create_sample_data():
    """创建示例步态数据用于测试"""
    import os
    
    # 创建示例数据
    time_points = 1000  # 10秒数据，100Hz采样率
    time = np.linspace(0, 10, time_points)
    
    # 模拟步态信号
    # 主步态频率约为1Hz
    gait_freq = 1.0
    
    # 左右脚压力信号
    left_foot = np.sin(2 * np.pi * gait_freq * time) + \
                0.3 * np.sin(2 * np.pi * 2 * gait_freq * time) + \
                0.1 * np.random.normal(0, 1, time_points)
    
    right_foot = np.sin(2 * np.pi * gait_freq * time + np.pi) + \
                 0.3 * np.sin(2 * np.pi * 2 * gait_freq * time + np.pi) + \
                 0.1 * np.random.normal(0, 1, time_points)
    
    # 躯干加速度信号
    trunk_acc_x = 0.5 * np.sin(2 * np.pi * gait_freq * time) + \
                  0.1 * np.random.normal(0, 1, time_points)
    trunk_acc_y = 0.3 * np.cos(2 * np.pi * gait_freq * time) + \
                  0.1 * np.random.normal(0, 1, time_points)
    trunk_acc_z = 9.8 + 0.2 * np.sin(2 * np.pi * gait_freq * time) + \
                  0.1 * np.random.normal(0, 1, time_points)
    
    # 创建DataFrame
    sample_data = pd.DataFrame({
        'time': time,
        'left_foot_pressure': left_foot,
        'right_foot_pressure': right_foot,
        'trunk_acc_x': trunk_acc_x,
        'trunk_acc_y': trunk_acc_y,
        'trunk_acc_z': trunk_acc_z
    })
    
    # 保存到文件
    os.makedirs('/home/ghr/5412/data/raw', exist_ok=True)
    sample_file = '/home/ghr/5412/data/raw/sample_gait_data.csv'
    sample_data.to_csv(sample_file, index=False)
    
    return sample_file


if __name__ == "__main__":
    # 创建示例数据
    sample_file = create_sample_data()
    
    # 初始化处理器
    processor = GaitDataProcessor(
        sampling_rate=100,
        filter_freq=(0.5, 10.0),
        normalize_method='standard'
    )
    
    # 处理数据
    features, metadata = processor.process_pipeline(
        sample_file,
        patient_info={
            'age': 65,
            'gender': 'male',
            'diagnosis': 'parkinsons'
        }
    )
    
    print("Data processing completed!")
    print(f"Features shape: {features.shape}")
    print(f"Metadata: {metadata}")
