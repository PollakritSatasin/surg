# Quick Start Guide

This guide will help you get started with SURG quickly.

## Installation

```bash
git clone https://github.com/PollakritSatasin/surg.git
cd surg
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

### COCO Format

Your dataset should be in COCO format with this structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234,
      "segmentation": [...],  # Polygon or RLE format
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "object_class"
    }
  ]
}
```

### Directory Structure

```
your_project/
├── data/
│   ├── train/
│   │   ├── images/
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   └── annotations.json
│   └── val/
│       ├── images/
│       └── annotations.json
├── examples/
│   └── config.yaml
```

## Training

### 1. Configure

Edit `examples/config.yaml`:

```yaml
model:
  num_classes: 3  # background + 2 object classes

data:
  train_root: /path/to/data/train/images
  train_annotations: /path/to/data/train/annotations.json
  val_root: /path/to/data/val/images
  val_annotations: /path/to/data/val/annotations.json

training:
  batch_size: 2
  num_epochs: 10
  device: auto
```

### 2. Train

```bash
cd examples
python train.py
```

### 3. Monitor

```bash
tensorboard --logdir logs
```

## Inference

```bash
python inference.py \
  --image /path/to/test.jpg \
  --model checkpoints/final_model.pth \
  --output result.png \
  --threshold 0.5
```

## Using in Python

### Basic Usage

```python
from surg.models import SegmentationModel
import torch
from PIL import Image
import torchvision.transforms as T

# Load model
model = SegmentationModel(num_classes=3, pretrained=False)
model.load('checkpoints/final_model.pth', device='cuda')

# Load image
image = Image.open('test.jpg').convert('RGB')
transform = T.ToTensor()
image_tensor = transform(image)

# Predict
predictions = model.predict(
    [image_tensor],
    device='cuda',
    confidence_threshold=0.5
)[0]

print(f"Detected {len(predictions['labels'])} objects")
```

### RL Integration

```python
from surg.rl_integration import create_rl_env_with_segmentation

# Create environment with segmentation
env = create_rl_env_with_segmentation(
    base_env_name='YourEnv-v0',
    segmentation_model=model,
    device='cuda',
    confidence_threshold=0.5
)

# Use in RL loop
obs = env.reset()
# obs contains: image, boxes, labels, scores, masks, num_objects

done = False
while not done:
    action = agent.select_action(obs)
    obs, reward, done, info = env.step(action)
```

## Advanced Features

### Custom Transforms

```python
from surg.utils.transforms import Compose, ToTensor, RandomHorizontalFlip, Normalize

transforms = Compose([
    ToTensor(),
    RandomHorizontalFlip(0.5),
    Normalize()
])

dataset = SegmentationDataset(..., transforms=transforms)
```

### Feature Extraction for RL

```python
from surg.rl_integration import SegmentationFeatureExtractor

feature_extractor = SegmentationFeatureExtractor(feature_type='combined')
features = feature_extractor.extract(predictions)
# Use features as input to RL policy
```

### Resume Training

```python
trainer = Trainer(model, train_dataset, val_dataset, config)
trainer.load_checkpoint('checkpoints/checkpoint_epoch_5.pth')
trainer.train()
```

## Common Issues

### Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 1
```

### Slow Training

Use GPU and reduce image size:
```yaml
training:
  device: cuda
  num_workers: 4
```

### Low Accuracy

- Train for more epochs
- Use data augmentation
- Check annotation quality
- Increase model capacity

## Next Steps

- Check out the full [README](../README.md)
- Read [CONTRIBUTING](../CONTRIBUTING.md) to contribute
- See example scripts in `examples/`
- Run tests: `pytest tests/`

## Support

For issues and questions, please open an issue on GitHub.
