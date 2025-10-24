"""
Tests for data transformation utilities.
"""

import torch
import pytest
from PIL import Image
import numpy as np
from surg.utils.transforms import (
    ToTensor,
    RandomHorizontalFlip,
    Normalize,
    get_transforms
)


def test_to_tensor():
    """Test ToTensor transform."""
    transform = ToTensor()
    
    # Create dummy PIL image
    image = Image.new('RGB', (100, 100), color='red')
    target = {'boxes': torch.tensor([[10, 10, 50, 50]])}
    
    image_t, target_t = transform(image, target)
    
    assert isinstance(image_t, torch.Tensor)
    assert image_t.shape == (3, 100, 100)
    assert torch.all(target_t['boxes'] == target['boxes'])


def test_random_horizontal_flip():
    """Test RandomHorizontalFlip transform."""
    transform = RandomHorizontalFlip(prob=1.0)  # Always flip
    
    # Create dummy tensor image
    image = torch.rand(3, 100, 100)
    target = {'boxes': torch.tensor([[10.0, 10.0, 50.0, 50.0]])}
    
    image_t, target_t = transform(image, target)
    
    assert image_t.shape == image.shape
    assert isinstance(target_t['boxes'], torch.Tensor)


def test_normalize():
    """Test Normalize transform."""
    transform = Normalize()
    
    # Create dummy tensor image
    image = torch.rand(3, 100, 100)
    target = {'boxes': torch.tensor([[10, 10, 50, 50]])}
    
    image_t, target_t = transform(image, target)
    
    assert image_t.shape == image.shape
    # Check that values are normalized (mean should be close to 0)
    assert abs(image_t.mean().item()) < 1.0


def test_get_transforms():
    """Test getting default transforms."""
    train_transforms = get_transforms(train=True, augment=True)
    val_transforms = get_transforms(train=False, augment=False)
    
    assert train_transforms is not None
    assert val_transforms is not None


if __name__ == '__main__':
    pytest.main([__file__])
