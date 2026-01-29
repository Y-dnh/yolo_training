# YOLO IR Detection Training Pipeline

A complete pipeline for training YOLO models. This project includes tools for COCO to YOLO format conversion, dataset preparation, model training/validation, and export to various formats.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Convert COCO to YOLO Format](#1-convert-coco-to-yolo-format)
  - [2. Prepare Dataset](#2-prepare-dataset)
  - [3. Train Model](#3-train-model)
  - [4. Validate Model](#4-validate-model)
  - [5. Export Model](#5-export-model)
- [Configuration](#configuration)
- [Dataset Structure](#dataset-structure)
- [Training Parameters](#training-parameters)
- [Thermal Image Considerations](#thermal-image-considerations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## ✨ Features

- **COCO to YOLO Conversion**: Convert COCO format annotations to YOLO format with support for:
  - Bounding boxes
  - Segmentation masks (optional)
  - Keypoints (optional)
  - Class mapping (91 to 80 classes)

- **Dataset Preparation**: Automated splitting into train/val/test sets with:
  - Configurable split ratios
  - Automatic YAML configuration generation
  - Label verification

- **Validation Tools**: Model validation and performance evaluation

- **Model Export**: Export trained models to various formats:
  - ONNX (universal, CPU/GPU)
  - TensorRT (NVIDIA GPU optimization)
  - TFLite (mobile/edge devices)
  - OpenVINO (Intel CPU/GPU)
  - CoreML (Apple devices)
  - And more (NCNN, MNN, PaddlePaddle, etc.)

## 📁 Project Structure

```
project/
├── coco_to_yolo.py          # COCO format converter
├── prepare_dataset.py       # Dataset splitting tool
├── train.py                 # Training script (+ auto ONNX export)
├── validate.py              # Validation script
├── export.py                # Model export (all formats)
├── requirements.txt         # Python dependencies
├── dataset/                 # Source data (create this)
│   ├── images/
│   └── labels/
├── dataset_split/           # Prepared dataset (auto-generated)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── yolo.yaml
└── yolov8x-p2_for_autolabelling/  # Training outputs (auto-generated)
    └── baseline/
        ├── weights/
        │   ├── best.pt
        │   ├── best.onnx      # Auto-exported after training
        │   └── last.pt
        └── results.png
```

## Installation

### Setup

1. Clone or download this project:
```bash
git clone <url>
cd yolo_training
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📖 Usage

### 1. Convert COCO to YOLO Format

If your annotations are in COCO format, convert them first:

```python
python coco_to_yolo.py
```

**Configuration** (edit in `coco_to_yolo.py`):
```python
CONFIG = {
    "labels_dir": "path/to/coco/annotations",  # Folder with JSON files
    "json_file": None,                         # Specific JSON file (or None for all)
    "save_dir": "path/to/output/labels",       # Output folder
    "use_segments": False,                     # Include segmentation masks
    "use_keypoints": False,                    # Include keypoints
    "cls91to80": False,                        # Map 91 COCO classes to 80
    "class_offset": 0,                         # Class ID offset (1 for COCO)
}
```

### 2. Prepare Dataset

Split your dataset into train/val/test sets:

```python
python prepare_dataset.py
```

**Configuration** (edit in `prepare_dataset.py`):
```python
# Split ratios
TRAIN_RATIO = ...
VAL_RATIO = ...
TEST_RATIO = ...

# Source paths
SOURCE_IMAGES = Path("dataset/images")
SOURCE_LABELS = Path("dataset/labels")

# Classes (adjust for your dataset)
CLASSES = {...}
```

The script will:
- Split images randomly with a fixed seed
- Copy images and labels to respective folders
- Generate `yolo.yaml` configuration
- Report statistics

### 3. Train Model

```python
python train.py
```

**Key Configuration** (edit in `train.py`):

```python
TRAINING_CONFIG = {
    "epochs": 100,
    "batch": 4,
    "imgsz": 960,
    # ... інші параметри
}

# Автоматичний експорт в ONNX після навчання
EXPORT_CONFIG = {
    "format": "onnx",
    "imgsz": 960,
    "dynamic": True,
    "simplify": True,
    # ...
}
```

Training outputs:
- Best model: `PROJECT_NAME/baseline/weights/best.pt`
- ONNX model: `PROJECT_NAME/baseline/weights/best.onnx` (auto-exported)
- Last model: `PROJECT_NAME/baseline/weights/last.pt`
- Training plots: `PROJECT_NAME/baseline/results.png`

### 4. Validate Model

Evaluate model performance:

```python
python validate.py
```

**Configuration** (edit in `validate.py`):

```python
VALIDATION_CONFIG = {
    "conf": 0.5,
    "iou": 0.5,
    "imgsz": 576,
    "split": "test",
    # ...
}
```

### 5. Export Model

Export trained model to various formats for deployment:

```python
python export.py
```

**Configuration** (edit in `export.py`):

```python
EXPORT_CONFIG = {
    "format": "onnx",       # onnx, engine, tflite, openvino, coreml, etc.
    "imgsz": 960,
    "half": False,          # FP16 quantization
    "int8": False,          # INT8 quantization (needs calibration data)
    "dynamic": True,        # Dynamic input size
    "simplify": True,       # Simplify ONNX graph
    # ...
}
```

**Supported export formats:**

| Format | Argument | Use Case |
|--------|----------|----------|
| ONNX | `onnx` | Universal, CPU/GPU inference |
| TensorRT | `engine` | NVIDIA GPU (up to 5x speedup) |
| OpenVINO | `openvino` | Intel CPU/GPU (up to 3x speedup) |
| TFLite | `tflite` | Mobile/Edge devices |
| CoreML | `coreml` | Apple devices (macOS/iOS) |
| NCNN | `ncnn` | Mobile/Embedded (ARM) |
| TorchScript | `torchscript` | PyTorch deployment |

**Programmatic usage:**

```python
from export import export_model

# Default export (uses EXPORT_CONFIG)
export_model()

# Custom export
export_model(format="engine", half=True)  # TensorRT FP16
export_model(format="tflite", int8=True, data="yolo.yaml")  # TFLite INT8
```

