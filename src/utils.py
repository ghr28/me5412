"""
项目配置管理模块
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        project_root = Path(__file__).parent.parent
        return str(project_root / "configs" / "train_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif self.config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {self.config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持嵌套访问（如 'model.tcn.dropout'）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, save_path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径，如果不指定则覆盖原文件
        """
        path = save_path or self.config_path
        
        with open(path, 'w', encoding='utf-8') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.safe_dump(self.config, f, ensure_ascii=False, indent=2)
            elif path.endswith('.json'):
                json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def update(self, updates: Dict[str, Any]):
        """
        批量更新配置
        
        Args:
            updates: 更新的配置字典
        """
        def deep_update(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        deep_update(self.config, updates)
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.get('data', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get('model', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.get('training', {})
    
    @property
    def synthetic_config(self) -> Dict[str, Any]:
        """获取合成数据配置"""
        return self.get('synthetic', {})


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取全局配置实例"""
    return config


def setup_logging(config: Config):
    """
    设置日志配置
    
    Args:
        config: 配置对象
    """
    import logging
    
    log_config = config.get('logging', {})
    
    # 创建日志目录
    log_file = log_config.get('file', 'logs/app.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def get_model_save_path(model_name: str) -> str:
    """
    获取模型保存路径
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型保存路径
    """
    models_dir = config.get('deployment.model_save_path', 'models')
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, f"{model_name}.pth")


def get_data_path(data_type: str) -> str:
    """
    获取数据路径
    
    Args:
        data_type: 数据类型 ('raw', 'processed', 'synthetic')
        
    Returns:
        数据路径
    """
    data_config = config.data_config
    path_key = f"{data_type}_data_path"
    
    if path_key in data_config:
        path = data_config[path_key]
        os.makedirs(path, exist_ok=True)
        return path
    else:
        raise ValueError(f"未找到数据类型 '{data_type}' 的配置")


if __name__ == "__main__":
    # 配置测试
    config = Config()
    print("数据配置:", config.data_config)
    print("模型配置:", config.model_config)
    print("训练配置:", config.training_config)
    
    # 测试路径获取
    print("原始数据路径:", get_data_path('raw'))
    print("模型保存路径:", get_model_save_path('test_model'))
