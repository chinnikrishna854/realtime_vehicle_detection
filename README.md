# YOLOv11 Real-Time Small Object Detection (UAV Vehicle Detection)

A comparative study of YOLOv11 model variants (nano, small, medium) for real-time small object detection from UAV (drone) imagery, using a vehicle detection dataset from Roboflow.

---

## Overview

This project fine-tunes three YOLOv11 model variants on a UAV vehicle detection dataset and benchmarks them against each other on detection quality and inference speed (FPS). The target use case is detecting small vehicles from aerial/drone perspectives — a challenging task due to the small object sizes and varying viewpoints.

---

## Dataset

**Source:** Roboflow — [`uav-vehicle-detection-1` (version 13)](https://roboflow.com)  
**Workspace:** `developmenttest`  
**Format:** YOLOv11

- Downloaded and extracted to `datasets/`
- Contains `train/`, `valid/`, and `test/` splits with images and labels
- Config file: `datasets/data.yaml`

---

## Models Compared

| Model | Size | Use Case |
|-------|------|----------|
| `yolo11n` | Nano | Fastest, lowest memory |
| `yolo11s` | Small | Balanced speed/accuracy |
| `yolo11m` | Medium | Higher accuracy, slower |

All models start from pretrained ImageNet weights and are fine-tuned on the UAV dataset.

---

## Pipeline

### 1. Install Dependencies
```bash
pip install "ultralytics<=8.3.40" supervision roboflow ray==2.40.0
```

### 2. Download Dataset
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("developmenttest").project("uav-vehicle-detection-1")
dataset = project.version(13).download("yolov11")
```

### 3. Train All Three Models
```bash
yolo task=detect mode=train model=yolo11n.pt data=datasets/data.yaml epochs=100 imgsz=640 plots=True
yolo task=detect mode=train model=yolo11s.pt data=datasets/data.yaml epochs=100 imgsz=640 plots=True
yolo task=detect mode=train model=yolo11m.pt data=datasets/data.yaml epochs=100 imgsz=640 plots=True
```
Trained weights saved to:
- `models/yolo11n/weights/best.pt`
- `models/yolo11s/weights/best.pt`
- `models/yolo11m/weights/best.pt`

### 4. Run Predictions on Test Set
```bash
yolo task=detect mode=predict model=models/yolo11n/weights/best.pt conf=0.25 source=datasets/test/images save=True
yolo task=detect mode=predict model=models/yolo11s/weights/best.pt conf=0.25 source=datasets/test/images save=True
yolo task=detect mode=predict model=models/yolo11m/weights/best.pt conf=0.25 source=datasets/test/images save=True
```

### 5. Visual Comparison
Side-by-side visualization of predictions from all three models on the same test images using matplotlib.

### 6. Inference Speed Benchmark
Measures per-image inference time and FPS for each model variant on a validation image (with warm-up pass).

---

## Training Config

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Image Size | 640×640 |
| Confidence Threshold (predict) | 0.25 |
| Base Weights | Pretrained YOLOv11 (ImageNet) |

---

## Inference Benchmark (Sample)

Timing measured on a single validation image after warm-up:

| Model | Inference Time | FPS |
|-------|---------------|-----|
| yolo11n | ~X.XXs | ~XX |
| yolo11s | ~X.XXs | ~XX |
| yolo11m | ~X.XXs | ~XX |

> Actual values depend on hardware (GPU/CPU). Run the benchmark cell to get results for your environment.

---

## Project Structure

```
├── datasets/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── models/
│   ├── yolo11n/weights/best.pt
│   ├── yolo11s/weights/best.pt
│   ├── yolo11m/weights/best.pt
│   ├── _yolo11n/          ← test predictions
│   ├── _yolo11s/
│   └── _yolo11m/
└── yolov11_real_time_small_object.ipynb
```

---

## Requirements

```
ultralytics<=8.3.40
supervision
roboflow
ray==2.40.0
opencv-python
matplotlib
```

---

## Notes

- A Roboflow API key is required to download the dataset. Replace `"mucvK2QUbJpoJGP1RzY1"` with your own key.
- Training is designed to run on GPU (Colab or local). CPU training will be significantly slower.
- The `ray==2.40.0` dependency is pinned for compatibility with the ultralytics version used.
