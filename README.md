# ☀️ Solar Panel Monitoring System

A multimodal AI system for detecting and diagnosing solar panel faults using
paired **RGB** (visual) and **IR/thermal** images.  The system fuses both
modalities at different levels of abstraction, runs YOLOv8-based object
detection, and applies a rule-based diagnosis engine to generate actionable
maintenance recommendations.

---

## Architecture Overview

```
RGB image ──┐                      ┌─── IR detections ──┐
            ├── Registration ──────┤                     ├── DiagnosisEngine
IR image  ──┘                      └─── RGB detections ──┘
            │
            ├── EarlyFusion   (4-channel concatenation)
            └── MidLevelFusion (dual-stream CNN features)
                      │
               YOLOv8 detector
                      │
               Streamlit dashboard
```

| Component | Description |
|---|---|
| `src/data/dataset.py` | PyTorch `Dataset` for paired RGB + IR images |
| `src/data/augmentation.py` | Shared geometric augmentations + IR-specific CLAHE |
| `src/fusion/registration.py` | Homography-based image alignment (manual or ORB) |
| `src/fusion/fusion.py` | Early fusion (concat) and Mid-Level fusion (dual-stream CNN) |
| `src/models/detector.py` | YOLOv8 wrapper with solar fault class names |
| `src/models/dual_stream.py` | End-to-end dual-stream classification model |
| `src/reasoning/diagnosis.py` | Rule-based diagnosis engine |
| `src/visualization/dashboard.py` | Streamlit monitoring dashboard |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd akvr_project

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
akvr_project/
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # SolarPanelDataset
│   │   └── augmentation.py     # SolarAugmentation
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── registration.py     # ImageRegistrator
│   │   └── fusion.py           # EarlyFusion, MidLevelFusion
│   ├── models/
│   │   ├── __init__.py
│   │   ├── detector.py         # SolarPanelDetector (YOLOv8)
│   │   └── dual_stream.py      # DualStreamModel
│   ├── reasoning/
│   │   ├── __init__.py
│   │   └── diagnosis.py        # DiagnosisEngine
│   └── visualization/
│       ├── __init__.py
│       └── dashboard.py        # Streamlit dashboard
└── tests/
    ├── __init__.py
    ├── test_augmentation.py
    ├── test_registration.py
    ├── test_fusion.py
    ├── test_diagnosis.py
    └── test_dataset.py
```

---

## Usage Examples

### Dataset loading

```python
from src.data.dataset import SolarPanelDataset

ds = SolarPanelDataset(
    root_dir="data/solar_panels",  # must contain rgb/, ir/, labels/
    load_labels=True,
)
print(len(ds))             # number of paired images
item = ds[0]
print(item["rgb"].shape)   # (3, H, W)
print(item["ir"].shape)    # (1, H, W)
print(item["label"])       # Tensor (N, 5) or None
```

### Data augmentation

```python
import numpy as np
from src.data.augmentation import SolarAugmentation

aug = SolarAugmentation(img_size=(640, 640), flip_p=0.5, apply_clahe=True)
rgb_aug, ir_aug = aug(rgb_np, ir_np)   # numpy uint8 arrays
```

### Image registration & fusion

```python
from src.fusion.registration import ImageRegistrator
from src.fusion.fusion import EarlyFusion

# Align IR to RGB
registrator = ImageRegistrator()
registered_ir, H = registrator.register(rgb_np, ir_np)

# Fuse into 4-channel tensor
fuser = EarlyFusion()
rgb_t, ir_t = EarlyFusion.to_tensor(rgb_np, registered_ir)
fused = fuser(rgb_t, ir_t)   # shape (4, H, W)
```

### Detection

```python
from src.models.detector import SolarPanelDetector

detector = SolarPanelDetector(model_path="yolov8n.pt")
detections = detector.detect(bgr_image, conf_threshold=0.3)
# [{'bbox': [x1,y1,x2,y2], 'confidence': 0.87, 'class_id': 0, 'class_name': 'hotspot'}, ...]

hotspots = detector.detect_hotspots(ir_image)
```

### Dual-stream classification

```python
from src.models.dual_stream import DualStreamModel

model = DualStreamModel(num_classes=5)
result = model.predict(rgb_np, ir_np)
print(result["class_name"], result["confidence"])

model.save("checkpoint.pt")
model = DualStreamModel.load("checkpoint.pt")
```

### Diagnosis

```python
from src.reasoning.diagnosis import DiagnosisEngine

engine = DiagnosisEngine()
result = engine.diagnose(ir_detections, rgb_detections)
print(result.diagnosis)   # "Cell defect / Internal short circuit"
print(result.priority)    # "high"
print(result.action)      # "Panel replacement required"
```

---

## Streamlit Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

The dashboard provides:
- Sidebar controls (confidence threshold, fusion method, detection mode)
- Side-by-side image upload (RGB + IR)
- Live detection results table
- Automated diagnosis with color-coded priority (🟢 low → 🚨 critical)

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover augmentation, image registration, fusion, diagnosis logic, and
dataset loading — no GPU or model downloads required.