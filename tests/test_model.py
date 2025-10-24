"""
Basic tests for the segmentation model.
"""

import torch
import pytest
from surg.models import SegmentationModel


def test_model_initialization():
    """Test that the model can be initialized."""
    model = SegmentationModel(num_classes=2, pretrained=False, backbone='resnet50')
    assert model is not None
    assert model.num_classes == 2


def test_model_forward():
    """Test forward pass with dummy data."""
    model = SegmentationModel(num_classes=2, pretrained=False, backbone='resnet50')
    model.model.eval()
    
    # Create dummy input
    images = [torch.rand(3, 224, 224)]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
    
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert 'boxes' in outputs[0]
    assert 'labels' in outputs[0]
    assert 'scores' in outputs[0]
    assert 'masks' in outputs[0]


def test_model_predict():
    """Test prediction method."""
    model = SegmentationModel(num_classes=2, pretrained=False, backbone='resnet50')
    
    # Create dummy input
    images = [torch.rand(3, 224, 224)]
    
    # Predict
    predictions = model.predict(images, device='cpu', confidence_threshold=0.5)
    
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert 'boxes' in predictions[0]
    assert 'labels' in predictions[0]
    assert 'scores' in predictions[0]
    assert 'masks' in predictions[0]


if __name__ == '__main__':
    pytest.main([__file__])
