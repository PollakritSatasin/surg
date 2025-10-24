#!/usr/bin/env python3
"""
Create a minimal sample dataset in COCO format for testing.
This creates synthetic data for demonstration purposes.
"""

import json
import os
import numpy as np
from PIL import Image, ImageDraw


def create_synthetic_image(output_path, image_id, size=(640, 480)):
    """Create a synthetic image with simple shapes."""
    # Create a random background
    img = Image.new('RGB', size, color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    
    # Draw some random rectangles and circles
    annotations = []
    num_objects = np.random.randint(1, 4)
    
    for i in range(num_objects):
        # Random position and size
        x = np.random.randint(50, size[0] - 150)
        y = np.random.randint(50, size[1] - 150)
        w = np.random.randint(50, 100)
        h = np.random.randint(50, 100)
        
        # Random color
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw shape
        if np.random.random() > 0.5:
            # Rectangle
            draw.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0), width=2)
            shape_type = 'rectangle'
        else:
            # Ellipse
            draw.ellipse([x, y, x + w, y + h], fill=color, outline=(0, 0, 0), width=2)
            shape_type = 'ellipse'
        
        # Create annotation
        ann = {
            'bbox': [x, y, w, h],
            'category_id': 1 if shape_type == 'rectangle' else 2,
            'area': w * h,
            # Simple polygon for segmentation
            'segmentation': [[
                float(x), float(y),
                float(x + w), float(y),
                float(x + w), float(y + h),
                float(x), float(y + h)
            ]]
        }
        annotations.append(ann)
    
    # Save image
    img.save(output_path)
    return annotations


def create_sample_dataset(output_dir, num_images=10):
    """Create a sample COCO-format dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # COCO format structure
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 1, 'name': 'rectangle'},
            {'id': 2, 'name': 'ellipse'}
        ]
    }
    
    annotation_id = 1
    
    print(f"Creating {num_images} sample images...")
    for image_id in range(1, num_images + 1):
        filename = f'sample_{image_id:04d}.jpg'
        filepath = os.path.join(output_dir, filename)
        
        # Create synthetic image
        annotations = create_synthetic_image(filepath, image_id)
        
        # Add image info
        coco_data['images'].append({
            'id': image_id,
            'file_name': filename,
            'height': 480,
            'width': 640
        })
        
        # Add annotations
        for ann in annotations:
            ann['id'] = annotation_id
            ann['image_id'] = image_id
            ann['iscrowd'] = 0
            coco_data['annotations'].append(ann)
            annotation_id += 1
        
        if image_id % 5 == 0:
            print(f"  Created {image_id}/{num_images} images")
    
    # Save annotations
    annotations_path = os.path.join(output_dir, 'annotations.json')
    with open(annotations_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nDataset created successfully!")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Location: {output_dir}")
    print(f"  Annotations file: {annotations_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample dataset for testing')
    parser.add_argument('--output', type=str, default='./sample_data',
                        help='Output directory for dataset')
    parser.add_argument('--num-train', type=int, default=20,
                        help='Number of training images')
    parser.add_argument('--num-val', type=int, default=5,
                        help='Number of validation images')
    args = parser.parse_args()
    
    # Create train set
    train_dir = os.path.join(args.output, 'train')
    print("Creating training set...")
    create_sample_dataset(train_dir, args.num_train)
    
    # Create validation set
    val_dir = os.path.join(args.output, 'val')
    print("\nCreating validation set...")
    create_sample_dataset(val_dir, args.num_val)
    
    print("\n" + "="*60)
    print("Sample dataset created successfully!")
    print("="*60)
    print("\nTo use this dataset, update your config.yaml:")
    print(f"  train_root: {train_dir}")
    print(f"  train_annotations: {os.path.join(train_dir, 'annotations.json')}")
    print(f"  val_root: {val_dir}")
    print(f"  val_annotations: {os.path.join(val_dir, 'annotations.json')}")
    print("\nThen run: python train.py")


if __name__ == '__main__':
    main()
