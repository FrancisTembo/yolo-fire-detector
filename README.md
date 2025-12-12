# YOLO Fire Detector

A real-time fire detection system built on the YOLOv11 (Ultralytics) object detection framework.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Configuration](#configuration)
- [License](#license)

## Overview

This repository contains a fire detection system that leverages YOLOv11 for accurate and efficient detection of fire in images and video streams. The system is designed for real-time applications including surveillance systems, industrial safety monitoring, and early fire warning systems.

## Features

- **Training Pipeline**: Train custom fire detection models using YOLOv11 architecture with configurable hyperparameters
- **Model Evaluation**: Comprehensive evaluation scripts with metrics visualisation and JSON export
- **Real-time Inference**: Support for multiple input sources including:
  - Static images
  - Image folders (batch processing)
  - Video files
  - USB webcams (live detection)
- **Video Recording**: Record annotated detection results to video files
- **Prediction Logging**: Export detection results to CSV for further analysis
- **Modular Architecture**: Clean, well-documented codebase with separated concerns

## Requirements

- Python 3.12 or higher
- CUDA-capable GPU (recommended for training and real-time inference)

### Development Environment

This project was developed and tested on the following system configuration:

- **Framework**: Ultralytics 8.3.236
- **Python**: 3.12.3
- **PyTorch**: 2.9.1+cu128
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (3768MiB VRAM)
- **CUDA**: Version 12.8
- **FFMPEG**: Version 6.1.1-3ubuntu5

## Installation

1. Clone the repository:

```bash
git clone https://github.com/FrancisTembo/yolo-fire-detector.git
cd yolo-fire-detector
```

2. Install dependencies using uv (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install ultralytics ipykernel
```

## Project Structure

```
yolo-fire-detector/
├── config.yaml          # Dataset configuration
├── hyp.yaml             # Training hyperparameters
├── eval.yaml            # Evaluation configuration
├── train.py             # Training script
├── train.ipynb          # Training notebook (interactive)
├── eval.py              # Evaluation script
├── inference.py         # Inference script
├── inference/           # Inference module
│   ├── config.py        # Detection configuration
│   ├── logger.py        # FPS and prediction logging
│   ├── source_manager.py# Input source handling
│   ├── utils.py         # Utility functions
│   └── visualizer.py    # Detection visualisation
├── dataset/             # Dataset directory
│   ├── train/           # Training images and labels
│   └── val/             # Validation images and labels
├── runs/                # Training outputs and weights
└── yolo11*.pt           # Pretrained model weights
```

## Usage

### Training

Train a fire detection model using the command line:

```bash
uv run python train.py --model yolo11s --data config.yaml --hyp hyp.yaml
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | YOLO model variant (yolo11n, yolo11s, etc.) | yolo11s |
| `--data` | Path to dataset configuration file | config.yaml |
| `--hyp` | Path to hyperparameters configuration file | hyp.yaml |

Training outputs (weights, metrics, plots) are saved to `runs/detect/yolo-training/`.

### Evaluation

Evaluate a trained model on the validation dataset:

```bash
uv run python eval.py --model runs/detect/yolo-training/weights/best.pt --data config.yaml --cfg eval.yaml
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to trained model weights | Required |
| `--data` | Path to dataset configuration file | config.yaml |
| `--cfg` | Path to evaluation configuration file | eval.yaml |

### Inference

Run inference on various input sources [2]:

**Single image:**

```bash
uv run python inference.py --model path/to/model/weights --source path/to/image.jpg
```

**Video file:**

```bash
uv run python inference.py --model path/to/model/weights --source path/to/video.mp4
```

**USB webcam:**

```bash
uv run python inference.py --model path/to/model/weights --source usb0
```

**With video recording:**

```bash
uv run python inference.py --model path/to/model/weights --source usb0 --resolution 640x480 --record --output output.avi
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to YOLO model weights | Required |
| `--source` | Input source (image, folder, video, or USB camera) | Required |
| `--thresh` | Confidence threshold for detections | 0.5 |
| `--resolution` | Display resolution (WxH format) | Source resolution |
| `--record` | Enable video recording | False |
| `--output` | Output filename for recorded video | demo1.avi |
| `--save-predictions` | Path to save prediction CSV | None |

## Configuration

### Dataset Configuration (config.yaml).

```yaml
path: dataset
train: train/images
val: val/images

nc: 1
names: ['fire']
```

### Training Hyperparameters (hyp.yaml)

Configure training behavior by modifying the following parameters. Full parameter documentation is available at https://docs.ultralytics.com/modes/train/

```yaml
name: 'yolo-training'  # Experiment name for organising outputs
epochs: 10             # Number of training epochs
batch: 12              # Batch size (adjust based on GPU memory)
imgsz: 640             # Input image size (pixels)
device: 0              # GPU device ID (0 for first GPU, 'cpu' for CPU)
patience: 10           # Early stopping patience (epochs without improvement)
save_period: 1         # Save checkpoint every N epochs
plots: True            # Generate training plots and visualisations
optimizer: 'auto'      # Optimizer selection (auto, SGD, Adam, AdamW)
cos_lr: True           # Use cosine learning rate scheduler
```

**Key Parameters:**

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `epochs` | Training duration | 50-300 for production models |
| `batch` | Samples per iteration | 8-32 (depends on GPU memory) |
| `imgsz` | Image resolution | 480, 640 |
| `patience` | Early stopping threshold | 10-50 epochs |

### Evaluation Configuration (eval.yaml)

Configure validation and testing parameters. Full parameter documentation is available at https://docs.ultralytics.com/modes/val/

```yaml
name: 'yolo-validation'  # Evaluation experiment name
batch: 12                # Batch size for validation
imgsz: 640               # Input image size (pixels)
device: 0                # GPU device ID
conf: 0.25               # Confidence threshold for predictions
plots: True              # Generate evaluation plots
iou: 0.7                 # IoU threshold for NMS
save_json: True          # Save results in COCO JSON format
```

**Key Parameters:**

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `conf` | Minimum confidence score | 0.1-0.5 for evaluation |
| `iou` | IoU threshold for NMS | 0.45-0.7 |
| `save_json` | Export COCO format results | True for detailed analysis |

## References

1. https://docs.ultralytics.com/
2. https://www.ejtech.io/learn/train-yolo-models
