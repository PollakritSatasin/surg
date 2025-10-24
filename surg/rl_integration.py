"""
Reinforcement Learning integration module.
Provides utilities to use the segmentation model in RL environments.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import gym
from gym import spaces


class SegmentationObservationWrapper(gym.ObservationWrapper):
    """
    Gym wrapper that adds segmentation predictions to observations.
    
    This wrapper augments the environment observation with instance segmentation
    masks and bounding boxes, useful for vision-based RL tasks.
    
    Args:
        env (gym.Env): Base environment
        segmentation_model: Trained segmentation model
        device (str): Device to run inference on
        confidence_threshold (float): Confidence threshold for predictions
        include_masks (bool): Whether to include segmentation masks in observation
        include_boxes (bool): Whether to include bounding boxes in observation
    """
    
    def __init__(
        self,
        env: gym.Env,
        segmentation_model,
        device: str = 'cpu',
        confidence_threshold: float = 0.5,
        include_masks: bool = True,
        include_boxes: bool = True
    ):
        super().__init__(env)
        self.model = segmentation_model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.include_masks = include_masks
        self.include_boxes = include_boxes
        
        self.model.model.eval()
        self.model.model.to(device)
    
    def observation(self, obs):
        """
        Augment observation with segmentation predictions.
        
        Args:
            obs: Original observation (assumed to be image)
            
        Returns:
            dict: Augmented observation with segmentation info
        """
        # Convert observation to tensor
        if isinstance(obs, np.ndarray):
            image = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
        else:
            image = obs
        
        # Run segmentation
        with torch.no_grad():
            predictions = self.model.predict(
                [image.to(self.device)],
                device=self.device,
                confidence_threshold=self.confidence_threshold
            )[0]
        
        # Build augmented observation
        augmented_obs = {
            'image': obs,
            'num_objects': len(predictions['labels'])
        }
        
        if self.include_boxes and len(predictions['boxes']) > 0:
            augmented_obs['boxes'] = predictions['boxes'].cpu().numpy()
            augmented_obs['labels'] = predictions['labels'].cpu().numpy()
            augmented_obs['scores'] = predictions['scores'].cpu().numpy()
        
        if self.include_masks and len(predictions['masks']) > 0:
            masks = predictions['masks'].squeeze(1).cpu().numpy()
            augmented_obs['masks'] = masks
        
        return augmented_obs


class SegmentationFeatureExtractor:
    """
    Extract features from segmentation predictions for RL policies.
    
    This class processes segmentation outputs into feature vectors that can be
    used as input to RL policies.
    
    Args:
        feature_type (str): Type of features to extract
            - 'boxes': Bounding box coordinates and sizes
            - 'spatial': Spatial statistics of masks
            - 'combined': Combination of box and spatial features
    """
    
    def __init__(self, feature_type: str = 'combined'):
        self.feature_type = feature_type
    
    def extract_box_features(self, boxes: np.ndarray) -> np.ndarray:
        """
        Extract features from bounding boxes.
        
        Args:
            boxes (np.ndarray): Array of boxes [N, 4] in [x1, y1, x2, y2] format
            
        Returns:
            np.ndarray: Box features [N, 6] (x, y, w, h, area, aspect_ratio)
        """
        if len(boxes) == 0:
            return np.zeros((0, 6))
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Center coordinates
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Width and height
        w = x2 - x1
        h = y2 - y1
        
        # Area
        area = w * h
        
        # Aspect ratio
        aspect_ratio = w / (h + 1e-6)
        
        features = np.stack([cx, cy, w, h, area, aspect_ratio], axis=1)
        return features
    
    def extract_mask_features(self, masks: np.ndarray) -> np.ndarray:
        """
        Extract spatial features from segmentation masks.
        
        Args:
            masks (np.ndarray): Array of masks [N, H, W]
            
        Returns:
            np.ndarray: Mask features [N, 4] (centroid_x, centroid_y, area, compactness)
        """
        if len(masks) == 0:
            return np.zeros((0, 4))
        
        features = []
        for mask in masks:
            mask_binary = mask > 0.5
            
            # Centroid
            y_coords, x_coords = np.where(mask_binary)
            if len(x_coords) > 0:
                cx = x_coords.mean()
                cy = y_coords.mean()
                area = mask_binary.sum()
                
                # Compactness (ratio of area to bounding box area)
                bbox_area = (x_coords.max() - x_coords.min() + 1) * (y_coords.max() - y_coords.min() + 1)
                compactness = area / (bbox_area + 1e-6)
            else:
                cx, cy, area, compactness = 0, 0, 0, 0
            
            features.append([cx, cy, area, compactness])
        
        return np.array(features)
    
    def extract(self, predictions: Dict) -> np.ndarray:
        """
        Extract features from segmentation predictions.
        
        Args:
            predictions (dict): Segmentation predictions
            
        Returns:
            np.ndarray: Feature vector
        """
        features = []
        
        if self.feature_type in ['boxes', 'combined']:
            if 'boxes' in predictions and len(predictions['boxes']) > 0:
                box_features = self.extract_box_features(predictions['boxes'])
                features.append(box_features)
        
        if self.feature_type in ['spatial', 'combined']:
            if 'masks' in predictions and len(predictions['masks']) > 0:
                mask_features = self.extract_mask_features(predictions['masks'])
                features.append(mask_features)
        
        if features:
            # Concatenate features
            combined = np.concatenate(features, axis=1)
            # Flatten to single vector (could also use per-object features)
            return combined.flatten()
        else:
            return np.array([])


def create_rl_env_with_segmentation(
    base_env_name: str,
    segmentation_model,
    device: str = 'cpu',
    **kwargs
) -> gym.Env:
    """
    Create a Gym environment augmented with segmentation observations.
    
    Args:
        base_env_name (str): Name of base Gym environment
        segmentation_model: Trained segmentation model
        device (str): Device for inference
        **kwargs: Additional arguments for wrapper
        
    Returns:
        gym.Env: Wrapped environment
    """
    base_env = gym.make(base_env_name)
    wrapped_env = SegmentationObservationWrapper(
        base_env,
        segmentation_model,
        device=device,
        **kwargs
    )
    return wrapped_env
