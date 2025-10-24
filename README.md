# SURG - Segmentation and Understanding for Reinforcement Learning with Graphics

A comprehensive framework for training and deploying instance segmentation models for reinforcement learning applications.

## Features

- **Instance Segmentation**: Built on Mask R-CNN with ResNet50 backbone
- **Easy Training**: Simple API for training on custom datasets
- **RL Integration**: Seamless integration with OpenAI Gym environments
- **COCO Format Support**: Compatible with COCO-style annotations
- **Data Augmentation**: Built-in augmentation pipeline
- **Visualization Tools**: Utilities for visualizing predictions and annotations
- **TensorBoard Support**: Track training progress with TensorBoard

## Installation

```bash
# Clone the repository
git clone https://github.com/PollakritSatasin/surg.git
cd surg

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### 1. Prepare Your Dataset

Organize your data in COCO format:

```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

The `annotations.json` should follow the COCO format with images, annotations, and categories.

### 2. Configure Training

Edit `examples/config.yaml` to set your dataset paths and training parameters:

```yaml
model:
  num_classes: 2  # Including background
  pretrained: true
  backbone: resnet50

data:
  train_root: ./data/train/images
  train_annotations: ./data/train/annotations.json
  val_root: ./data/val/images
  val_annotations: ./data/val/annotations.json

training:
  batch_size: 2
  num_epochs: 10
  learning_rate: 0.005
  device: auto  # Automatically selects CUDA if available
```

### 3. Train the Model

```bash
cd examples
python train.py
```

Training logs will be saved to `./logs` and checkpoints to `./checkpoints`.

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs
```

### 4. Run Inference

```bash
python inference.py --image path/to/image.jpg --model checkpoints/final_model.pth --output result.png
```

### 5. Use in Reinforcement Learning

```python
from surg.models import SegmentationModel
from surg.rl_integration import create_rl_env_with_segmentation

# Load trained model
model = SegmentationModel(num_classes=2, pretrained=False)
model.load('checkpoints/final_model.pth')

# Create RL environment with segmentation
env = create_rl_env_with_segmentation(
    base_env_name='YourEnv-v0',
    segmentation_model=model,
    device='cuda'
)

# Use in your RL training loop
obs = env.reset()
# obs now contains segmentation information: masks, boxes, labels
```

## API Reference

### Models

```python
from surg.models import SegmentationModel

# Initialize model
model = SegmentationModel(num_classes=2, pretrained=True, backbone='resnet50')

# Make predictions
predictions = model.predict(images, device='cuda', confidence_threshold=0.5)

# Save/Load model
model.save('model.pth')
model.load('model.pth', device='cuda')
```

### Dataset

```python
from surg.data import SegmentationDataset
from surg.utils import get_transforms

# Create dataset
dataset = SegmentationDataset(
    root='./data/train/images',
    annotations_file='./data/train/annotations.json',
    transforms=get_transforms(train=True, augment=True)
)
```

### Training

```python
from surg.training import Trainer

# Configure and train
config = {
    'batch_size': 2,
    'num_epochs': 10,
    'learning_rate': 0.005,
    'device': 'cuda'
}

trainer = Trainer(model, train_dataset, val_dataset, config)
trainer.train()
```

### RL Integration

```python
from surg.rl_integration import (
    SegmentationObservationWrapper,
    SegmentationFeatureExtractor
)

# Wrap Gym environment
wrapped_env = SegmentationObservationWrapper(
    env, 
    segmentation_model,
    device='cuda',
    confidence_threshold=0.5
)

# Extract features for RL policy
feature_extractor = SegmentationFeatureExtractor(feature_type='combined')
features = feature_extractor.extract(predictions)
```

## Project Structure

```
surg/
├── surg/
│   ├── models/           # Model architectures
│   ├── data/             # Dataset loaders
│   ├── training/         # Training utilities
│   ├── utils/            # Helper functions
│   ├── config/           # Configuration management
│   └── rl_integration.py # RL integration utilities
├── examples/             # Example scripts
│   ├── train.py
│   ├── inference.py
│   ├── rl_example.py
│   └── config.yaml
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
└── README.md
```

## Advanced Usage

### Custom Transforms

```python
from surg.utils.transforms import Compose, ToTensor, RandomHorizontalFlip, Normalize

transforms = Compose([
    ToTensor(),
    RandomHorizontalFlip(prob=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SegmentationDataset(..., transforms=transforms)
```

### Visualization

```python
from surg.utils import visualize_predictions

# Visualize predictions
visualize_predictions(
    image,
    predictions,
    class_names=['background', 'object'],
    save_path='output.png',
    threshold=0.5
)
```

### Resume Training

```python
# Load checkpoint
trainer.load_checkpoint('checkpoints/checkpoint_epoch_5.pth')

# Continue training
trainer.train()
```

## Examples

Check the `examples/` directory for complete examples:

- **train.py**: Full training pipeline
- **inference.py**: Run inference on images
- **rl_example.py**: Integration with RL environments
- **config.yaml**: Configuration template

## Requirements

- Python >= 3.7
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- OpenCV >= 4.8.0
- Other dependencies in `requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{surg2024,
  title={SURG: Segmentation and Understanding for Reinforcement Learning with Graphics},
  author={PollakritSatasin},
  year={2024},
  url={https://github.com/PollakritSatasin/surg}
}
```

## Acknowledgments

- Built on top of torchvision's Mask R-CNN implementation
- Supports COCO format annotations
- Inspired by vision-based reinforcement learning research