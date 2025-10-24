#!/usr/bin/env python3
"""
Example training script for instance segmentation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from surg.models import SegmentationModel
from surg.data import SegmentationDataset
from surg.training import Trainer
from surg.utils import get_transforms
from surg.config import load_config


def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        print("Config file not found. Using default configuration.")
        from surg.config.config import get_default_config
        config = get_default_config()
    
    # Initialize model
    print("Initializing model...")
    model = SegmentationModel(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        backbone=config['model']['backbone']
    )
    
    # Load datasets
    print("Loading datasets...")
    try:
        train_dataset = SegmentationDataset(
            root=config['data']['train_root'],
            annotations_file=config['data']['train_annotations'],
            transforms=get_transforms(train=True, augment=config['data']['augmentation'])
        )
        
        val_dataset = None
        if os.path.exists(config['data']['val_annotations']):
            val_dataset = SegmentationDataset(
                root=config['data']['val_root'],
                annotations_file=config['data']['val_annotations'],
                transforms=get_transforms(train=False, augment=False)
            )
        
        print(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure your data is in COCO format and paths are correct.")
        return
    
    # Initialize trainer
    print("Initializing trainer...")
    
    # Handle 'auto' device selection
    training_config = config['training'].copy()
    if training_config['device'] == 'auto':
        training_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config
    )
    
    # Start training
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(config['training']['save_dir'], 'final_model.pth')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")


if __name__ == '__main__':
    main()
