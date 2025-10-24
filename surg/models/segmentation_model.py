"""
Instance segmentation model using Mask R-CNN architecture.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class SegmentationModel(nn.Module):
    """
    Wrapper for instance segmentation model (Mask R-CNN).
    
    Args:
        num_classes (int): Number of classes including background
        pretrained (bool): Whether to use pretrained weights
        backbone (str): Backbone architecture (default: 'resnet50')
    """
    
    def __init__(self, num_classes=2, pretrained=True, backbone='resnet50'):
        super(SegmentationModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained Mask R-CNN model
        if backbone == 'resnet50':
            self.model = maskrcnn_resnet50_fpn(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )
        
        # Get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        
        # Replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
    
    def forward(self, images, targets=None):
        """
        Forward pass of the model.
        
        Args:
            images (list[Tensor]): Images to be processed
            targets (list[Dict[Tensor]]): Ground-truth boxes, labels and masks
            
        Returns:
            During training: dict with losses
            During inference: list of predictions
        """
        return self.model(images, targets)
    
    def predict(self, images, device='cpu', confidence_threshold=0.5):
        """
        Make predictions on a batch of images.
        
        Args:
            images (list[Tensor]): Images to predict on
            device (str): Device to run inference on
            confidence_threshold (float): Minimum confidence for predictions
            
        Returns:
            list[Dict]: Predictions for each image
        """
        self.model.eval()
        self.model.to(device)
        
        with torch.no_grad():
            predictions = self.model(images)
        
        # Filter predictions by confidence threshold
        filtered_predictions = []
        for pred in predictions:
            mask = pred['scores'] > confidence_threshold
            filtered_pred = {
                'boxes': pred['boxes'][mask],
                'labels': pred['labels'][mask],
                'scores': pred['scores'][mask],
                'masks': pred['masks'][mask]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def save(self, path):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path, device='cpu'):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=device))
