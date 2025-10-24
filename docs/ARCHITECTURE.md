# SURG Framework Architecture

This document describes the architecture and design decisions of the SURG framework.

## Overview

SURG (Segmentation and Understanding for Reinforcement Learning with Graphics) is a modular framework designed to bridge computer vision and reinforcement learning by providing instance segmentation capabilities for RL environments.

## Design Principles

1. **Modularity**: Each component is independent and can be used separately
2. **Flexibility**: Support for different backbones, datasets, and configurations
3. **RL-First**: Designed with reinforcement learning integration as a primary goal
4. **Ease of Use**: Simple APIs with sensible defaults
5. **Extensibility**: Easy to extend with custom models, transforms, and features

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (Training Scripts, Inference Scripts, RL Integration)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       API Layer                              │
│  (High-level interfaces: Trainer, SegmentationModel, etc.)  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                              │
│    (Models, Data Loaders, Training Logic, Transforms)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Foundation Layer                           │
│         (PyTorch, torchvision, OpenCV, Gym)                  │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### 1. Models (`surg/models/`)

**Purpose**: Define neural network architectures for instance segmentation.

**Components**:
- `SegmentationModel`: Wrapper around Mask R-CNN
  - Uses torchvision's pre-trained models
  - Customizable number of classes
  - Simple predict() interface for inference
  - Save/load functionality

**Design Decisions**:
- Built on Mask R-CNN for its proven performance
- ResNet50 backbone as default (balance of speed and accuracy)
- Pretrained on COCO for transfer learning
- Easy to extend with other architectures

### 2. Data (`surg/data/`)

**Purpose**: Handle dataset loading and preprocessing.

**Components**:
- `SegmentationDataset`: COCO-format dataset loader
  - Loads images, boxes, labels, and masks
  - Supports both polygon and RLE mask formats
  - Custom collate function for variable-size batches
  - Integration with transforms

**Design Decisions**:
- COCO format for wide compatibility
- PyTorch Dataset interface for seamless integration
- Lazy loading for memory efficiency
- Flexible transform pipeline

### 3. Training (`surg/training/`)

**Purpose**: Manage the training process.

**Components**:
- `Trainer`: Complete training pipeline
  - SGD optimizer with momentum
  - Learning rate scheduling
  - TensorBoard logging
  - Checkpoint saving/loading
  - Validation support

**Design Decisions**:
- Configuration-driven for flexibility
- Automatic device selection (CUDA/CPU)
- Progress tracking with tqdm
- Modular design for easy customization

### 4. Utils (`surg/utils/`)

**Purpose**: Provide utility functions for data processing and visualization.

**Components**:
- `transforms.py`: Data augmentation
  - ToTensor, Normalize, RandomHorizontalFlip, Resize
  - Composable transform pipeline
  - Automatic target adjustment
  
- `visualization.py`: Prediction visualization
  - Box and mask overlay
  - Color-coded instances
  - Confidence score display

**Design Decisions**:
- Compatible with torchvision transforms
- Proper handling of both images and targets
- ImageNet normalization by default

### 5. RL Integration (`surg/rl_integration.py`)

**Purpose**: Bridge segmentation and reinforcement learning.

**Components**:
- `SegmentationObservationWrapper`: Gym environment wrapper
  - Adds segmentation to observations
  - Configurable output (boxes, masks, or both)
  - Confidence filtering
  
- `SegmentationFeatureExtractor`: Feature extraction
  - Converts predictions to feature vectors
  - Multiple feature types (boxes, spatial, combined)
  - Ready for RL policy input

**Design Decisions**:
- Standard Gym interface for compatibility
- Flexible feature extraction for different RL algorithms
- Efficient inference with caching
- Modular design for custom features

### 6. Config (`surg/config/`)

**Purpose**: Manage configuration and hyperparameters.

**Components**:
- YAML-based configuration
- Default configuration templates
- Load/save utilities

**Design Decisions**:
- Human-readable YAML format
- Hierarchical structure (model, data, training, inference)
- Easy to version control

## Data Flow

### Training Flow

```
Data (COCO format)
    ↓
SegmentationDataset
    ↓
Transforms (augmentation, normalization)
    ↓
DataLoader (batching)
    ↓
SegmentationModel (forward)
    ↓
Loss Computation
    ↓
Optimizer (backward)
    ↓
Checkpoint Saving
```

### Inference Flow

```
Input Image
    ↓
Preprocessing (to_tensor)
    ↓
SegmentationModel.predict()
    ↓
Confidence Filtering
    ↓
Predictions (boxes, labels, scores, masks)
    ↓
Visualization / RL Integration
```

### RL Integration Flow

```
RL Environment Reset/Step
    ↓
Image Observation
    ↓
SegmentationObservationWrapper
    ↓
Segmentation Model Inference
    ↓
Feature Extraction
    ↓
Augmented Observation (image + segmentation)
    ↓
RL Agent Policy
```

## Key Design Patterns

### 1. Wrapper Pattern

Used extensively for extending functionality:
- `SegmentationObservationWrapper` wraps Gym environments
- `SegmentationModel` wraps torchvision's Mask R-CNN
- Transform classes wrap image operations

### 2. Factory Pattern

Used for creating configured instances:
- `get_transforms()` creates transform pipelines
- `get_default_config()` creates default configurations
- `create_rl_env_with_segmentation()` creates wrapped environments

### 3. Strategy Pattern

Used for flexible feature extraction:
- `SegmentationFeatureExtractor` with different feature types
- Configurable transform pipelines
- Pluggable model architectures

## Extension Points

### Adding New Models

```python
# In surg/models/your_model.py
class YourModel(nn.Module):
    def forward(self, images, targets=None):
        # Your implementation
        pass
    
    def predict(self, images, device='cpu', confidence_threshold=0.5):
        # Your inference logic
        pass
```

### Adding New Transforms

```python
# In surg/utils/transforms.py
class YourTransform:
    def __call__(self, image, target):
        # Transform image and adjust target
        return image, target
```

### Adding New Features

```python
# In surg/rl_integration.py
class YourFeatureExtractor:
    def extract(self, predictions):
        # Extract custom features
        return features
```

## Performance Considerations

### Training

- **Batch Size**: Balance GPU memory and convergence speed
- **Num Workers**: Use multiple workers for data loading
- **Mixed Precision**: Can be added for faster training
- **Gradient Accumulation**: For larger effective batch sizes

### Inference

- **Batch Processing**: Process multiple images together
- **Confidence Threshold**: Higher threshold = faster (fewer detections)
- **Image Size**: Smaller images = faster inference
- **Model Caching**: Keep model in memory for RL applications

### RL Integration

- **Feature Caching**: Cache features when observation doesn't change
- **Lazy Evaluation**: Only run segmentation when needed
- **Asynchronous Inference**: Run segmentation in parallel thread
- **Feature Compression**: Use compact feature representations

## Testing Strategy

### Unit Tests

- Model initialization and forward pass
- Transform correctness
- Dataset loading
- Feature extraction

### Integration Tests

- End-to-end training pipeline
- Inference on sample images
- RL environment wrapping

### Validation

- Visual inspection of predictions
- TensorBoard metrics
- Performance benchmarks

## Future Enhancements

1. **More Architectures**: YOLOv8, Detectron2 integration
2. **Advanced Features**: Attention maps, object relationships
3. **Optimization**: Model quantization, TensorRT
4. **RL Algorithms**: Integration with stable-baselines3
5. **Multi-GPU**: Distributed training support
6. **Export**: ONNX export for deployment
7. **Tracking**: Multi-object tracking for video
8. **3D Support**: Point cloud segmentation

## References

- Mask R-CNN: He et al., "Mask R-CNN", ICCV 2017
- COCO Dataset: Lin et al., "Microsoft COCO", ECCV 2014
- PyTorch: Paszke et al., "PyTorch: An Imperative Style Deep Learning Library"
- OpenAI Gym: Brockman et al., "OpenAI Gym"
