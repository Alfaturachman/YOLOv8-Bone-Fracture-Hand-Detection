# FracAtlas - yolov8s Fracture Detection

Bone Fracture Localization in X-ray Images Using the yolov8s Deep Learning Object Detection Model on the FracAtlas Dataset

> **Note:** This project has been filtered to focus specifically on **hand fracture detection** only, as per academic supervision requirements.

## Dataset Overview

| Property              | Value / Description                                 |
| :-------------------- | :-------------------------------------------------- |
| **Total Images**      | 4,084 X-rays (original FracAtlas)                   |
| **Filtered Images**   | Hand images only (~1,538 images)                    |
| **Resolution**        | ~1760 x 2140 pixels                                 |
| **Target Class**      | 1 (`fracture`)                                      |
| **Annotation Format** | Bounding box in YOLO (.txt)                         |
| **Training Split**    | 70% Train \| 15% Val \| 15% Test (stratified split) |

### Data Distribution (Original FracAtlas)

- **Fractured images**: 717 (17.6%) images with fracture annotations
- **Non-fractured images**: 3,366 (82.4%) images without fractures
- **Body Part Distribution:**
    - **Hand:** 1,538 ✓ **(Used in this project)**
    - **Leg:** 2,273
    - **Hip:** 338
    - **Shoulder:** 349
    - **Mixed:** 398
- **View Type Distribution:**
    - **Frontal:** 2,503
    - **Lateral:** 1,492
    - **Oblique:** 418
- **Special Categories:**
    - **Hardware:** 99
    - **Multi-scan:** 396

### Hand-Only Dataset Statistics

After filtering for hand images only (see Notebook 01, Section 2.3):

| Category               | Count  | Percentage |
| :--------------------- | :----- | :--------- |
| **Hand Fractured**     | ~267   | ~17.4%     |
| **Hand Non-fractured** | ~1,271 | ~82.6%     |
| **Total Hand**         | ~1,538 | 100%       |

_Note: Exact counts may vary slightly. Run Notebook 01 to see precise statistics._

## Quick Start

### Prerequisites

| Requirement     | Version    | How to Check          |
| --------------- | ---------- | --------------------- |
| Python          | 3.9 - 3.11 | `python --version`    |
| CUDA (optional) | 11.7+      | `nvidia-smi`          |
| GPU Memory      | 6GB+       | Required for training |

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from ultralytics import YOLO; print('YOLO installed')"
```

---

## Implementation Pipeline

The workflow is divided into four sequential stages, each with its own Jupyter Notebook for transparency and reproducibility.

### Step 1 - Data Preparation

- **File:** `notebooks/01_data_preparation.ipynb`
- **Action:** Converts raw files into the standard YOLO directory structure.
- **Output:** `yolo_dataset/` folder with images/labels partitions.

### Step 2 - Model Training

- **File:** `notebooks/02_training_yolov8.ipynb`
- **Action:** Trains the `yolo8s.pt` model with medical-optimized augmentations.
- **Output:** `runs/detect/train/weights/best.pt`.
- **Framework:** Ultralytics
- **Model:** `yolo8s.pt` (Pre-trained on COCO)
- **Hyperparameters:**
    - **Image Size:** 1024 pixels
    - **Epochs:** 100 (with early stopping)
    - **Batch Size:** 8 (adjustable based on VRAM)
    - **Optimizer:** Auto (SGD/AdamW)
    - **Augmentation:** Standard yolov8 augmentations (mosaic, flip, scale).

### Step 3 - Quantitative Evaluation

- **File:** `notebooks/03_evaluation.ipynb`
- **Action:** Validates performance using Precision, Recall, and mAP metrics.
- **Visuals:** Confusion Matrix and PR Curves.

### Step 4 - Visual Inference

- **File:** `notebooks/04_inference.ipynb`
- **Action:** Runs the trained model on unseen test images to verify clinical utility.
- **Output:** Predicted images and annotated X-rays with detected fracture bounding boxes.

---

**Important:**

- Complete each notebook before moving to the next
- Notebook 01 must finish successfully before running Notebook 02
- Each notebook is self-contained with clear outputs

## Project Structure

```
FracAtlas/
├── images/                     # Original Datasets
│   ├── Fractured/              # Fractured X-rays
│   ├── Non_fractured/          # Non-fractured X-rays
├── Annotations/
│   └── YOLO/                   # YOLO format labels
├── splits/
│   ├── train.csv               # Training split
│   ├── val.csv                 # Validation split
│   └── test.csv                # Test split
├── runs/
│   └── detect/                 # Training outputs
├── requirements.txt
├── fricatlas.yaml              # # YOLO dataset config
├── dataset.csv
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_training_yolov8.ipynb
│   ├── 03_evaluation.ipynb
│   ├── 04_inference.ipynb
├── yolo_dataset/               # Datasets Hand Only
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
```

## Configuration

### Model Variants

| Model       | Parameters | Speed    | Accuracy |
| ----------- | ---------- | -------- | -------- |
| `yolo8n.pt` | ~2.5M      | Fastest  | Lower    |
| `yolo8s.pt` | ~7M        | Fast     | Good     |
| `yolo8s.pt` | ~20M       | Balanced | Better   |
| `yolo8l.pt` | ~43M       | Slow     | High     |
| `yolo8x.pt` | ~68M       | Slowest  | Best     |

### Key Hyperparameters

```python
# Learning rate
lr0: 0.01
lrf: 0.01

# Augmentation (medical-image optimized)
degrees: 10.0        # Limited rotation
translate: 0.1       # Small translation
scale: 0.5           # Scale variation
shear: 0.0           # Disabled (anatomical)
flipud: 0.0          # Disabled (anatomical orientation)
fliplr: 0.5          # Horizontal flip OK
mosaic: 1.0          # Mosaic augmentation
mixup: 0.1           # Light mixup
```

## Expected Metrics

---

### Target Performance Metrics

Evaluate the trained model using the following performance metrics to ensure reliable medical object detection results:

| Metric    | Rationale                                                                     |
| --------- | ----------------------------------------------------------------------------- |
| mAP@50    | Measures detection accuracy with IoU threshold 0.50 (high overlap confidence) |
| mAP@50–95 | Measures localization precision across multiple IoU thresholds                |
| Recall    | Ensures minimal false negatives in medical diagnosis                          |
| Precision | Ensures minimal false positives in detection                                  |
| F1-Score  | Provides a balanced measurement between precision and recall                  |

---

### Required Evaluation Visualizations

Generate the following evaluation graphs and visualizations based on the test dataset results:

1. **Precision–Recall Curve**
    - Plot precision vs recall for each class and overall model performance.

2. **Confusion Matrix**
    - Display the confusion matrix showing True Positives, False Positives, False Negatives, and True Negatives for each class.

3. **mAP Curve**
    - Visualize mAP performance across IoU thresholds (0.50–0.95).

4. **F1-Score Curve**
    - Plot F1-score as a function of confidence threshold.

5. **Precision Curve**
    - Plot precision vs confidence threshold.

6. **Recall Curve**
    - Plot recall vs confidence threshold.

7. **Training Metrics Graph**
    - Display training vs validation performance over epochs including:
        - Loss curves (box_loss, cls_loss, dfl_loss)
        - mAP@50
        - mAP@50–95

---

### Output Requirement

Save and export all evaluation graphs as image files for analysis and reporting, including:

- `precision_recall_curve.png`
- `confusion_matrix.png`
- `map_curve.png`
- `f1_curve.png`
- `precision_curve.png`
- `recall_curve.png`
- `training_metrics.png`

These visualizations will be used to analyze model performance and support results in a scientific publication.

---

_Actual performance depends on training duration and hyperparameters_

## Medical Imaging Considerations

### Augmentation Strategy

| Augmentation    | Setting        | Rationale                                      |
| :-------------- | :------------- | :--------------------------------------------- |
| Vertical Flip   | Disabled       | Anatomical orientation matters                 |
| Horizontal Flip | Optional       | Acceptable for symmetric anatomy (arms/legs)   |
| Perspective     | Disabled       | X-rays are projection images                   |
| Shear           | Disabled       | Preserves anatomical structure                 |
| HSV             | Light          | X-rays are grayscale                           |
| Rotation        | Limited (±10°) | Simulates slight patient positioning variation |
| Translation     | Small (≤5%)    | Simulates minor patient shift                  |
| Scaling         | 0.9 – 1.1      | Simulates distance variation                   |

### Class Imbalance Handling

Fracture datasets often contain more normal bones than fractures.

| Technique     | Setting                       |
| ------------- | ----------------------------- |
| Weighted Loss | Enabled                       |
| Data Sampling | Balanced sampling if possible |

### Confidence Thresholds

Different thresholds are used depending on the clinical scenario.

| Use Case  | Confidence  | Goal                                         |
| --------- | ----------- | -------------------------------------------- |
| Screening | 0.10 – 0.20 | Maximize recall to avoid missed fractures    |
| Diagnosis | 0.25 – 0.40 | Balanced precision and recall                |
| Clinical  | ≥0.50       | High precision for clinical decision support |

### Post-Processing

| Method                        | Setting       | Purpose                                |
| ----------------------------- | ------------- | -------------------------------------- |
| Non-Maximum Suppression (NMS) | IoU 0.5 – 0.7 | Remove duplicate detections            |
| Minimum Box Area              | Optional      | Avoid extremely small false detections |

## Hardware Requirements

| Component | Minimum        | Recommended      |
| --------- | -------------- | ---------------- |
| GPU       | GTX 1660 (6GB) | RTX 3080 (10GB+) |
| RAM       | 16 GB          | 32 GB            |
| Storage   | 20 GB free     | SSD preferred    |

### Estimated Training Time

| GPU      | Time (100 epochs) |
| -------- | ----------------- |
| RTX 3050 | ~2-3 hours        |
| RTX 3080 | ~1-1.5 hours      |

## Training Monitoring

Training logs and plots are saved to `runs/detect/fricatlas_yolo8s/`:

- `results.csv` - Epoch-by-epoch metrics
- `results.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last checkpoint

## Troubleshooting

### CUDA Out of Memory

```bash
# In notebook, reduce batch size
batch_size = 8

# Or reduce image size
imgsz = 1024
```

### Poor Performance

1. Verify data preparation step completed successfully
2. Check labels exist in `yolo_dataset/labels/train/`
3. Increase training epochs to 150
4. Adjust learning rate in the hyperparameters cell

### No Detections

- Lower confidence threshold: `conf=0.1`
- Verify model training completed: Check `weights/best.pt` exists
- Ensure correct class mapping in `fricatlas.yaml`

## License

This project is for research and educational purposes. Please ensure compliance with medical data regulations when using patient data.

## Acknowledgments

- **yolov8**: Ultralytics YOLO framework
- **FracAtlas**: Bone fracture X-ray dataset

## Support

For issues or questions:

1. Check existing issues in the repository
2. Review Ultralytics documentation: https://docs.ultralytics.com
3. Verify data preparation completed successfully
