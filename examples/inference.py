#!/usr/bin/env python3
"""
Example inference script for instance segmentation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from PIL import Image
from surg.models import SegmentationModel
from surg.utils import visualize_predictions
from surg.config import load_config
import torchvision.transforms as T


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save output')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print("Config file not found. Using defaults.")
        from surg.config.config import get_default_config
        config = get_default_config()
    
    # Initialize model
    print("Loading model...")
    device = config['inference']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = SegmentationModel(
        num_classes=config['model']['num_classes'],
        pretrained=False,  # We're loading trained weights
        backbone=config['model']['backbone']
    )
    
    # Load trained weights
    model.load(args.model, device=device)
    print(f"Model loaded from: {args.model}")
    
    # Load and preprocess image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert('RGB')
    
    # Transform to tensor
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)
    
    # Run inference
    print("Running inference...")
    predictions = model.predict(
        [image_tensor],
        device=device,
        confidence_threshold=args.threshold
    )[0]
    
    print(f"Detected {len(predictions['labels'])} objects")
    
    # Visualize results
    print(f"Saving visualization to: {args.output}")
    visualize_predictions(
        image_tensor,
        predictions,
        save_path=args.output,
        show=False,
        threshold=args.threshold
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
