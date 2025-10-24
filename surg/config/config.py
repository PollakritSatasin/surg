"""
Configuration loading and validation.
"""

import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save YAML file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for training.
    
    Returns:
        dict: Default configuration
    """
    return {
        'model': {
            'num_classes': 2,  # Including background
            'pretrained': True,
            'backbone': 'resnet50'
        },
        'data': {
            'train_root': './data/train',
            'train_annotations': './data/train/annotations.json',
            'val_root': './data/val',
            'val_annotations': './data/val/annotations.json',
            'augmentation': True
        },
        'training': {
            'batch_size': 2,
            'num_workers': 4,
            'learning_rate': 0.005,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'num_epochs': 10,
            'device': 'auto',  # 'auto', 'cuda', or 'cpu'
            'save_dir': './checkpoints',
            'log_dir': './logs',
            'save_interval': 1,
            'print_freq': 10
        },
        'inference': {
            'confidence_threshold': 0.5,
            'device': 'auto'
        }
    }
