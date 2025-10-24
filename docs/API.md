# API Documentation

## Models

### SegmentationModel

Instance segmentation model based on Mask R-CNN.

```python
from surg.models import SegmentationModel

model = SegmentationModel(num_classes=2, pretrained=True, backbone='resnet50')
```

#### Parameters

- `num_classes` (int): Number of classes including background
- `pretrained` (bool): Whether to use pretrained weights on COCO
- `backbone` (str): Backbone architecture ('resnet50')

#### Methods

##### forward(images, targets=None)

Forward pass through the model.

**Parameters:**
- `images` (list[Tensor]): List of images
- `targets` (list[dict], optional): Ground truth annotations

**Returns:**
- During training: dict with losses
- During inference: list of predictions

##### predict(images, device='cpu', confidence_threshold=0.5)

Make predictions on images.

**Parameters:**
- `images` (list[Tensor]): Images to predict on
- `device` (str): Device for inference
- `confidence_threshold` (float): Minimum confidence score

**Returns:**
- list[dict]: Predictions with keys 'boxes', 'labels', 'scores', 'masks'

##### save(path)

Save model weights to file.

##### load(path, device='cpu')

Load model weights from file.

## Data

### SegmentationDataset

Dataset loader for instance segmentation in COCO format.

```python
from surg.data import SegmentationDataset

dataset = SegmentationDataset(
    root='./data/train/images',
    annotations_file='./data/train/annotations.json',
    transforms=transforms
)
```

#### Parameters

- `root` (str): Directory containing images
- `annotations_file` (str): Path to COCO format JSON
- `transforms` (callable, optional): Transformations to apply

#### Methods

##### \_\_getitem\_\_(idx)

Get an image and its annotations.

**Returns:**
- tuple: (image, target) where target is a dict with:
  - `boxes` (Tensor[N, 4]): Bounding boxes
  - `labels` (Tensor[N]): Class labels
  - `masks` (Tensor[N, H, W]): Segmentation masks
  - `image_id` (Tensor): Image ID
  - `area` (Tensor[N]): Box areas
  - `iscrowd` (Tensor[N]): Crowd flags

##### collate_fn(batch)

Custom collate function for DataLoader.

## Training

### Trainer

Training manager for segmentation models.

```python
from surg.training import Trainer

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config
)
```

#### Parameters

- `model`: SegmentationModel instance
- `train_dataset`: Training dataset
- `val_dataset` (optional): Validation dataset
- `config` (dict): Training configuration

#### Configuration Options

```python
config = {
    'batch_size': 2,
    'num_workers': 4,
    'learning_rate': 0.005,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'num_epochs': 10,
    'device': 'cuda',
    'save_dir': './checkpoints',
    'log_dir': './logs',
    'save_interval': 1,
    'print_freq': 10
}
```

#### Methods

##### train()

Run the complete training loop.

##### train_epoch(epoch)

Train for one epoch.

**Returns:**
- float: Average training loss

##### validate()

Validate the model.

**Returns:**
- dict: Validation metrics

##### save_checkpoint(epoch)

Save a training checkpoint.

##### load_checkpoint(checkpoint_path)

Load a training checkpoint.

## Utilities

### Transforms

Data transformation utilities.

```python
from surg.utils.transforms import (
    Compose, ToTensor, RandomHorizontalFlip, Normalize, Resize, get_transforms
)
```

#### get_transforms(train=True, augment=True)

Get default transform pipeline.

**Parameters:**
- `train` (bool): Whether for training
- `augment` (bool): Whether to apply augmentation

**Returns:**
- Compose: Composed transforms

#### Custom Transforms

##### ToTensor()

Convert PIL Image to Tensor.

##### RandomHorizontalFlip(prob=0.5)

Randomly flip images horizontally.

##### Normalize(mean=None, std=None)

Normalize with ImageNet statistics.

##### Resize(size)

Resize images and adjust targets.

### Visualization

```python
from surg.utils.visualization import visualize_predictions

visualize_predictions(
    image=image_tensor,
    predictions=predictions,
    class_names=['bg', 'class1', 'class2'],
    save_path='output.png',
    show=True,
    threshold=0.5
)
```

#### visualize_predictions

Visualize segmentation predictions.

**Parameters:**
- `image` (Tensor): Image tensor
- `predictions` (dict): Prediction dictionary
- `class_names` (list, optional): Class names for labels
- `save_path` (str, optional): Path to save figure
- `show` (bool): Whether to display
- `threshold` (float): Confidence threshold

## RL Integration

### SegmentationObservationWrapper

Gym wrapper that adds segmentation to observations.

```python
from surg.rl_integration import SegmentationObservationWrapper

wrapped_env = SegmentationObservationWrapper(
    env=base_env,
    segmentation_model=model,
    device='cuda',
    confidence_threshold=0.5,
    include_masks=True,
    include_boxes=True
)
```

#### Parameters

- `env` (gym.Env): Base environment
- `segmentation_model`: Trained model
- `device` (str): Inference device
- `confidence_threshold` (float): Confidence threshold
- `include_masks` (bool): Include masks in observation
- `include_boxes` (bool): Include boxes in observation

### SegmentationFeatureExtractor

Extract features from predictions for RL policies.

```python
from surg.rl_integration import SegmentationFeatureExtractor

extractor = SegmentationFeatureExtractor(feature_type='combined')
features = extractor.extract(predictions)
```

#### Parameters

- `feature_type` (str): Type of features
  - 'boxes': Box coordinates and sizes
  - 'spatial': Spatial mask statistics
  - 'combined': Both box and spatial features

#### Methods

##### extract(predictions)

Extract feature vector from predictions.

**Returns:**
- ndarray: Feature vector

##### extract_box_features(boxes)

Extract features from bounding boxes.

##### extract_mask_features(masks)

Extract spatial features from masks.

### Helper Functions

#### create_rl_env_with_segmentation

Create a Gym environment with segmentation.

```python
from surg.rl_integration import create_rl_env_with_segmentation

env = create_rl_env_with_segmentation(
    base_env_name='YourEnv-v0',
    segmentation_model=model,
    device='cuda',
    confidence_threshold=0.5
)
```

## Configuration

### load_config / save_config

Load and save YAML configurations.

```python
from surg.config import load_config, save_config

config = load_config('config.yaml')
save_config(config, 'new_config.yaml')
```

### get_default_config

Get default configuration dictionary.

```python
from surg.config.config import get_default_config

config = get_default_config()
```

## Examples

See the `examples/` directory for complete usage examples:

- `train.py`: Complete training pipeline
- `inference.py`: Running inference on images
- `rl_example.py`: RL integration example
- `config.yaml`: Configuration template
