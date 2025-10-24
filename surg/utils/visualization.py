"""
Visualization utilities for predictions and annotations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import List, Dict, Optional


def visualize_predictions(
    image: torch.Tensor,
    predictions: Dict,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    threshold: float = 0.5
):
    """
    Visualize instance segmentation predictions on an image.
    
    Args:
        image (Tensor): Image tensor [C, H, W]
        predictions (dict): Predictions containing boxes, labels, scores, masks
        class_names (list): List of class names
        save_path (str): Path to save the visualization
        show (bool): Whether to show the plot
        threshold (float): Confidence threshold for displaying predictions
    """
    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()
    
    # Denormalize if needed (assuming ImageNet normalization)
    if image.max() <= 1.0:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Filter by threshold
    if 'scores' in predictions:
        mask = predictions['scores'] > threshold
        boxes = predictions['boxes'][mask]
        labels = predictions['labels'][mask]
        scores = predictions['scores'][mask]
        masks = predictions['masks'][mask] if 'masks' in predictions else None
    else:
        boxes = predictions['boxes']
        labels = predictions['labels']
        scores = None
        masks = predictions.get('masks', None)
    
    # Draw masks
    if masks is not None:
        for i, mask in enumerate(masks):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Get the mask as binary
            if mask.ndim == 3:
                mask = mask[0]  # Remove channel dimension
            mask = mask > 0.5
            
            # Create colored overlay
            color = plt.cm.Set3(i % 12)[:3]
            colored_mask = np.zeros((*mask.shape, 3))
            for c in range(3):
                colored_mask[:, :, c] = color[c]
            
            # Overlay mask with transparency
            ax.imshow(colored_mask, alpha=0.5 * mask)
    
    # Draw boxes and labels
    for i, box in enumerate(boxes):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Random color for each instance
        color = plt.cm.Set3(i % 12)
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Draw label
        label_idx = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
        label_text = class_names[label_idx] if class_names else f"Class {label_idx}"
        
        if scores is not None:
            score = scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i]
            label_text += f" {score:.2f}"
        
        ax.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(facecolor=color, alpha=0.5),
            fontsize=10,
            color='white'
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_dataset_sample(dataset, idx=0, save_path=None, show=True):
    """
    Visualize a sample from the dataset with ground truth annotations.
    
    Args:
        dataset: Instance segmentation dataset
        idx (int): Index of sample to visualize
        save_path (str): Path to save the visualization
        show (bool): Whether to show the plot
    """
    image, target = dataset[idx]
    
    # Use visualize_predictions with ground truth
    visualize_predictions(
        image,
        target,
        save_path=save_path,
        show=show,
        threshold=0.0  # Show all ground truth
    )
