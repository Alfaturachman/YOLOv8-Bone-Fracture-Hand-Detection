# Hand Bone Fracture Detection Using YOLOv8

**Comparative Analysis of YOLOv8s and YOLOv8m with Oversampling Strategy for Hand Bone Fracture Detection on X-Ray Images Using the FracAtlas Dataset**

> This project is developed for a **Sinta 3 journal submission**. The goal is to systematically compare YOLOv8s (Small) and YOLOv8m (Medium) for detecting hand bone fractures on X-ray images from the FracAtlas dataset, using a medically-optimized preprocessing pipeline with oversampling to handle class imbalance.

---

## Research Overview

| Property                     | Value                                                |
| :--------------------------- | :--------------------------------------------------- |
| **Task**                     | Object Detection (Fracture Localization)             |
| **Dataset**                  | FracAtlas — Hand subset only                         |
| **Models Compared**          | YOLOv8s (Small) vs. YOLOv8m (Medium)                 |
| **Class Imbalance Handling** | Oversampling on training set                         |
| **Framework**                | Ultralytics YOLOv8                                   |
| **Evaluation**               | mAP@50, Recall, Precision, F1, FPS, Confusion Matrix |

---

## Dataset

### Source: FracAtlas (Hand Subset Only)

| Category               | Count  | Percentage |
| :--------------------- | :----- | :--------- |
| **Hand Fractured**     | ~267   | ~17.4%     |
| **Hand Non-fractured** | ~1,271 | ~82.6%     |
| **Total Hand Images**  | ~1,538 | 100%       |

Original FracAtlas contains 4,083 X-rays across multiple body parts (Hand, Leg, Hip, Shoulder, Mixed). This project uses **hand images only** to eliminate confounding features from other anatomical regions.

### After Oversampling (Training Set Only)

| Category                  | Count     | Percentage |
| :------------------------ | :-------- | :--------- |
| Fractured (train)         | 920       | 54.4%      |
| Non-fractured (train)     | 770       | 45.6%      |
| **Total training images** | **1,690** | —          |
| Val images                | 362       | —          |
| Test images               | 362       | —          |
| **Total dataset**         | **2,414** | —          |

> Validation and Test sets use the **original unmodified distribution** to ensure unbiased evaluation.

---

## Models

| Property       | YOLOv8s (Small)    | YOLOv8m (Medium)   |
| :------------- | :----------------- | :----------------- |
| Parameters     | ~11.2M             | ~25.9M             |
| GFLOPs (640px) | 28.6               | 79.1               |
| Pretrained on  | COCO (80 classes)  | COCO (80 classes)  |
| Fine-tuned for | 1 class (fracture) | 1 class (fracture) |

---

## Results

### Accuracy Metrics (Val Set — Best Epoch)

| Metric     | YOLOv8s   | YOLOv8m   |
| :--------- | :-------- | :-------- |
| **mAP@50** | **0.847** | 0.838     |
| mAP@50-95  | **0.444** | 0.429     |
| **Recall** | **0.780** | 0.726     |
| Precision  | 0.891     | **0.912** |
| Best Epoch | 100       | 98        |

### Inference Speed (GPU: RTX 3050 4GB)

| Model   | FPS | Latency (ms) |
| :------ | :-- | :----------- |
| YOLOv8s | TBD | TBD          |
| YOLOv8m | TBD | TBD          |

_Results to be filled after running Cell 8 in the training notebooks._

---

## Implementation Pipeline

### Notebook Sequence

```
01b_hand_dataset.ipynb
  → Filter hand-only images from FracAtlas
  → Apply stratified split (70/15/15)
  → Apply oversampling on training set
  → Output: yolo_dataset_hand_oversampled/

02a_training_yolov8s_oversampled.ipynb
  → Train YOLOv8s on oversampled dataset
  → Benchmark FPS (Cell 8)
  → Evaluate on test set + confusion matrix (Cell 9)
  → Qualitative visualization (Cell 10)
  → Output: runs-old-v8s-oversampled/

02a_training_yolov8m_oversampled.ipynb
  → Train YOLOv8m on oversampled dataset
  → Benchmark FPS (Cell 8)
  → Evaluate on test set + confusion matrix (Cell 9)
  → Qualitative visualization (Cell 10)
  → Output: runs-old-v8m-oversampled/
```

### Running Order

1. Run `01b_hand_dataset.ipynb` to prepare dataset _(if not done yet)_
2. Run `02a_training_yolov8s_oversampled.ipynb` — **training already complete**, run Cells 8–10 for journal outputs
3. Run `02a_training_yolov8m_oversampled.ipynb` — **training already complete**, run Cells 8–10 for journal outputs

---

## Project Structure

```
FracAtlas/
│
├── images/                              # [git-ignored] Original FracAtlas images
│   ├── Fractured/
│   └── Non_fractured/
│
├── Annotations/YOLO/                    # YOLO format label files (.txt)
│
├── yolo_dataset_hand_oversampled/       # [git-ignored] Prepared dataset
│   ├── train/images/ & labels/
│   ├── val/images/   & labels/
│   └── test/images/  & labels/
│
├── runs-old-v8s-oversampled/            # [git-ignored] YOLOv8s training outputs
│   └── detect/fracatlas_yolov8s_adamw/
│       ├── weights/best.pt
│       ├── weights/last.pt
│       ├── results.csv
│       └── confusion_matrix.png
│
├── runs-old-v8m-oversampled/            # [git-ignored] YOLOv8m training outputs
│   └── detect/fracatlas_yolov8m_adamw/
│       ├── weights/best.pt
│       ├── weights/last.pt
│       ├── results.csv
│       └── confusion_matrix.png
│
├── notebooks/
│   ├── 01b_hand_dataset.ipynb           # Data preparation + oversampling
│   ├── 02a_training_yolov8s_oversampled.ipynb   # YOLOv8s training + eval
│   └── 02a_training_yolov8m_oversampled.ipynb   # YOLOv8m training + eval
│
├── fracatlas_hand_oversampled.yaml      # YOLO dataset config (oversampled)
├── fracatlas_hand.yaml                  # YOLO dataset config (original)
├── paper_draft.md                       # Research paper draft
├── dataset.csv                          # Full dataset metadata
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Training Configuration

### Hyperparameters (Both Models)

| Parameter       | Value | Notes                                           |
| :-------------- | :---- | :---------------------------------------------- |
| `optimizer`     | AdamW | More stable than SGD for small medical datasets |
| `lr0`           | 0.001 | Lower than default (0.01) to avoid overshooting |
| `lrf`           | 0.01  | Final LR ratio with cosine scheduler            |
| `epochs`        | 100   | With early stopping (`patience=20`)             |
| `imgsz`         | 640   | Balanced detail vs. VRAM constraint (4GB)       |
| `batch`         | 4     | Limited by VRAM                                 |
| `cos_lr`        | True  | Smooth cosine decay                             |
| `warmup_epochs` | 5     | Extended warmup for stability                   |
| `amp`           | True  | Automatic Mixed Precision                       |

### Augmentation Strategy (X-Ray Optimized)

| Parameter       | Value   | Reason                                              |
| :-------------- | :------ | :-------------------------------------------------- |
| `degrees`       | 10.0°   | Simulate patient positioning variation              |
| `fliplr`        | 0.5     | Safe for symmetric hand anatomy                     |
| `flipud`        | **0.0** | Disabled — proximal/distal bone orientation matters |
| `shear`         | **0.0** | Disabled — anatomical distortion                    |
| `perspective`   | **0.0** | Disabled — X-rays are orthogonal projections        |
| `mosaic`        | 0.5     | Reduced from 1.0 — avoids medical image artifacts   |
| `mixup`         | 0.1     | Light regularization                                |
| `copy_paste`    | **0.0** | Disabled — invalid for medical domain               |
| `hsv_h / hsv_s` | **0.0** | Disabled — X-rays are grayscale                     |
| `hsv_v`         | 0.2     | Simulate X-ray exposure variation                   |

---

## Journal Outputs Checklist

The following outputs are required for journal submission (Sinta 3):

| Output                               | Source                 | Status                           |
| :----------------------------------- | :--------------------- | :------------------------------- |
| mAP@50, mAP@50-95, Recall, Precision | `results.csv`          | ✅ Done                          |
| Training loss curves (box/cls/dfl)   | `results.png`          | ✅ Done                          |
| Confusion matrix (val set)           | `confusion_matrix.png` | ✅ Done (auto-generated by YOLO) |
| **Inference FPS**                    | Notebook Cell 8        | ⏳ Run Cell 8                    |
| **Confusion matrix (test set)**      | Notebook Cell 9        | ⏳ Run Cell 9                    |
| **Qualitative prediction samples**   | Notebook Cell 10       | ⏳ Run Cell 10                   |

---

## Environment

| Component   | Version                           |
| :---------- | :-------------------------------- |
| Python      | 3.10.19                           |
| PyTorch     | 2.7.1+cu118                       |
| Ultralytics | 8.4.63                            |
| CUDA        | 11.8                              |
| GPU         | NVIDIA RTX 3050 Laptop (4GB VRAM) |
| OS          | Windows 11                        |

### Setup

```bash
# Create and activate conda environment
conda create -n yolo python=3.10
conda activate yolo

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Medical Imaging Considerations

### Why Recall is Prioritized

In fracture detection, a **False Negative** (missed fracture) is far more dangerous than a **False Positive** (unnecessary follow-up). A model with high Precision but low Recall is clinically unacceptable. This is why:

- Oversampling is applied to prevent the model from biasing toward "non-fracture"
- Recall is the **primary clinical metric** in Results and Discussion

### Confidence Threshold Guidelines

| Scenario           | Threshold   | Goal                                       |
| :----------------- | :---------- | :----------------------------------------- |
| Screening (IGD)    | 0.10 – 0.20 | Maximize recall, minimize missed fractures |
| Diagnostic support | 0.25 – 0.40 | Balanced precision and recall              |
| Clinical decision  | ≥ 0.50      | High precision for direct clinical use     |

---

## References

- **FracAtlas dataset:** Shadmand et al. (2023). _FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation._ Scientific Data.
- **Ultralytics YOLOv8:** Jocher, G. et al. (2023). _Ultralytics YOLOv8._ GitHub.
- **Class imbalance:** Buda, M., Maki, A., & Mazurowski, M.A. (2018). _A systematic study of the class imbalance problem in convolutional neural networks._ Neural Networks, 106, 249–259.

---

## License

For research and educational purposes only. Ensure compliance with institutional data regulations when using patient imaging data.
