"""
Data transformation utilities for augmentation and preprocessing.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import numpy as np


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image to Tensor."""
    
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    """Randomly flip image and targets horizontally."""
    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = F.hflip(image)
            
            if "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
                target["boxes"] = boxes
            
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        
        return image, target


class Normalize:
    """Normalize image with mean and std."""
    
    def __init__(self, mean=None, std=None):
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
    
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    """Resize image and adjust targets accordingly."""
    
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, target):
        original_size = image.shape[-2:]
        image = F.resize(image, self.size)
        new_size = image.shape[-2:]
        
        if "boxes" in target:
            boxes = target["boxes"]
            # Scale boxes
            scale_y = new_size[0] / original_size[0]
            scale_x = new_size[1] / original_size[1]
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target["boxes"] = boxes
        
        if "masks" in target:
            masks = target["masks"]
            masks = F.resize(masks.unsqueeze(1), self.size).squeeze(1)
            target["masks"] = masks
        
        return image, target


def get_transforms(train=True, augment=True):
    """
    Get default transforms for training or validation.
    
    Args:
        train (bool): Whether this is for training
        augment (bool): Whether to apply data augmentation
        
    Returns:
        Compose: Composed transforms
    """
    transforms = [ToTensor()]
    
    if train and augment:
        transforms.append(RandomHorizontalFlip(prob=0.5))
    
    # Normalize with ImageNet stats
    transforms.append(Normalize())
    
    return Compose(transforms)
