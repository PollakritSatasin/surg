"""
SURG - Segmentation and Understanding for Reinforcement Learning with Graphics
A framework for training and using instance segmentation models for RL applications.
"""

__version__ = "0.1.0"

from surg.models import SegmentationModel
from surg.data import SegmentationDataset
from surg.training import Trainer

__all__ = ["SegmentationModel", "SegmentationDataset", "Trainer"]
