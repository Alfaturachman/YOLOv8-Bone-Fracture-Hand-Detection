#!/usr/bin/env python
# coding: utf-8

# # Bounding Box-Based Detection Difficulty Analysis
# **FracAtlas — Comparing YOLOv8m vs YOLOv8s with AdamW**
# 
# Classifies detected objects into **Easy / Moderate / Hard** levels based on
# mathematical thresholds on bounding-box parameters and compares performance
# across the two trained models.

# ## 0. Imports & Settings

# In[ ]:


import os, sys, json, yaml, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.bbox'] = 'tight'

SAVE_DPI = 150
BASE_DIR = Path(r"D:\Project Medical Object Detection\FracAtlas")
OUTPUT_DIR = BASE_DIR / "notebooks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_REF_W = IMG_REF_H = 1024  # training resolution (reference for difficulty)

print("Libraries loaded.")
print(f"Output dir: {OUTPUT_DIR}")


# ## 1. Define Model Configs

# In[ ]:


YAML_PATH = BASE_DIR / "fracatlas_hand_oversampled.yaml"

MODELS = {
    "YOLOv8m-AdamW": {
        "run_dir": BASE_DIR / "runs-old-v8m-adamw" / "detect",
        "run_name": "fracatlas_yolov8m_adamw",
    },
    "YOLOv8s-AdamW": {
        "run_dir": BASE_DIR / "runs-old-v8s-adamw" / "detect",
        "run_name": "fracatlas_yolov8s_adamw",
    },
}

# Load dataset info
with open(YAML_PATH) as f:
    data_cfg = yaml.safe_load(f)
dataset_path = Path(data_cfg["path"])
test_img_dir = dataset_path / "test" / "images"
test_lbl_dir = dataset_path / "test" / "labels"

test_images = sorted(test_img_dir.iterdir())
print(f"Test images: {len(test_images)}")
for name, cfg in MODELS.items():
    best_pt = cfg["run_dir"] / cfg["run_name"] / "weights" / "best.pt"
    assert best_pt.exists(), f"{best_pt} not found"
    print(f"  {name}: {best_pt}")


# ## 2. Helper Functions

# In[ ]:


def yolo_to_abs(x_center, y_center, width, height, img_w, img_h):
    """Convert YOLO normalized coords to absolute pixels for given image size."""
    x1 = (x_center - width / 2) * img_w
    y1 = (y_center - height / 2) * img_h
    x2 = (x_center + width / 2) * img_w
    y2 = (y_center + height / 2) * img_h
    return x1, y1, x2, y2


def bbox_area_abs(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_iou(box1, box2):
    """box = [x1, y1, x2, y2] in absolute pixels."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = bbox_area_abs(*box1)
    area2 = bbox_area_abs(*box2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def classify_by_area(area_ref_px, thresholds=None):
    """
    Classify by bounding-box area scaled to 1024x1024 reference frame.
    Default thresholds (COCO-inspired):
      Easy   (large)   : area > 96x96 = 9216 px²
      Moderate (medium): 32x32 < area <= 96x96  => 1024 < area <= 9216
      Hard   (small)   : area <= 32x32 = 1024 px²
    """
    if thresholds is None:
        thresh_easy = 96 * 96    # 9216
        thresh_mod  = 32 * 32    # 1024
    else:
        thresh_mod, thresh_easy = thresholds
    if area_ref_px > thresh_easy:
        return "Easy"
    elif area_ref_px > thresh_mod:
        return "Moderate"
    else:
        return "Hard"


def classify_by_aspect_ratio(w, h, threshold=3.0):
    """Aspect ratio = max(w,h)/min(w,h). High => harder."""
    if w <= 0 or h <= 0:
        return "Hard"
    ratio = max(w, h) / min(w, h)
    if ratio <= threshold:
        return "Easy"
    elif ratio <= threshold * 2:
        return "Moderate"
    else:
        return "Hard"


def classify_by_confidence(conf, thresholds=None):
    """Confidence-based difficulty (Easy = high conf)."""
    if thresholds is None:
        thresh_easy = 0.7
        thresh_mod  = 0.4
    else:
        thresh_mod, thresh_easy = thresholds
    if conf >= thresh_easy:
        return "Easy"
    elif conf >= thresh_mod:
        return "Moderate"
    else:
        return "Hard"


def composite_difficulty(area_ref_px, aspect_ratio, conf):
    """Composite score combining all three signals."""
    score = 0.0
    max_area = IMG_REF_W * IMG_REF_H
    area_score = min(area_ref_px / max_area * 10, 1.0)
    score += area_score * 0.4
    ar = max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-6))
    ar_score = max(0, 1.0 - (ar - 1.0) / 9.0)
    score += ar_score * 0.3
    score += conf * 0.3
    if score >= 0.7:
        return "Easy", score
    elif score >= 0.4:
        return "Moderate", score
    else:
        return "Hard", score


# ## 3. Run Inference & Collect Predictions

# In[ ]:


def collect_predictions(model, test_images, test_lbl_dir, conf_thresh=0.01):
    rows = []
    for img_path in tqdm(test_images, desc="Predicting"):
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")

        # Get actual image dimensions
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # Ground truth (convert using actual image dimensions)
        gt_boxes_abs = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                xc, yc, w, h = parts[1:5]
                x1, y1, x2, y2 = yolo_to_abs(xc, yc, w, h, img_w, img_h)
                gt_boxes_abs.append([x1, y1, x2, y2, cls_id])

        # Predictions (ultralytics returns coords in original image space)
        preds = model.predict(str(img_path), conf=conf_thresh, iou=0.5, verbose=False)
        pred_boxes = []
        if preds and preds[0].boxes is not None:
            boxes = preds[0].boxes
            for i in range(len(boxes)):
                x1p, y1p, x2p, y2p = boxes.xyxy[i].tolist()
                conf_p = float(boxes.conf[i])
                cls_p = int(boxes.cls[i])
                pred_boxes.append([x1p, y1p, x2p, y2p, conf_p, cls_p])

        num_gt = len(gt_boxes_abs)

        # Match predictions to GT (greedy IoU)
        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for j, gb in enumerate(gt_boxes_abs):
                if j in matched_gt:
                    continue
                iou = compute_iou(pb[:4], gb[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            is_tp = 1 if (best_iou >= 0.5 and best_gt_idx >= 0) else 0
            if is_tp:
                matched_gt.add(best_gt_idx)

            # Scale to reference 1024x1024 for difficulty classification
            scale_x = IMG_REF_W / img_w
            scale_y = IMG_REF_H / img_h
            wp_ref = (pb[2] - pb[0]) * scale_x
            hp_ref = (pb[3] - pb[1]) * scale_y
            area_ref = wp_ref * hp_ref
            ar = max(wp_ref, hp_ref) / max(min(wp_ref, hp_ref), 1)

            rows.append({
                "image": img_path.name,
                "img_w": img_w, "img_h": img_h,
                "x1": pb[0], "y1": pb[1], "x2": pb[2], "y2": pb[3],
                "width": pb[2] - pb[0], "height": pb[3] - pb[1],
                "area_ref_px": area_ref,
                "aspect_ratio": ar,
                "confidence": pb[4],
                "iou": best_iou,
                "is_tp": is_tp,
                "num_gt": num_gt,
            })

        # False negatives: GT boxes not matched
        scale_x = IMG_REF_W / img_w
        scale_y = IMG_REF_H / img_h
        for j, gb in enumerate(gt_boxes_abs):
            if j not in matched_gt:
                wg_ref = (gb[2] - gb[0]) * scale_x
                hg_ref = (gb[3] - gb[1]) * scale_y
                area_ref = wg_ref * hg_ref
                ar = max(wg_ref, hg_ref) / max(min(wg_ref, hg_ref), 1)
                rows.append({
                    "image": img_path.name,
                    "img_w": img_w, "img_h": img_h,
                    "x1": gb[0], "y1": gb[1], "x2": gb[2], "y2": gb[3],
                    "width": gb[2] - gb[0], "height": gb[3] - gb[1],
                    "area_ref_px": area_ref,
                    "aspect_ratio": ar,
                    "confidence": 0.0,
                    "iou": 0.0,
                    "is_tp": 0,
                    "num_gt": num_gt,
                })

    return pd.DataFrame(rows)


all_results = {}
for model_name, cfg in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Processing {model_name}...")
    best_pt = cfg["run_dir"] / cfg["run_name"] / "weights" / "best.pt"
    model = YOLO(str(best_pt))
    df = collect_predictions(model, test_images, test_lbl_dir, conf_thresh=0.01)
    all_results[model_name] = df
    tp = df['is_tp'].sum()
    fn = (df['confidence'] == 0).sum()
    fp = len(df[df['confidence'] > 0]) - tp
    print(f"  Total rows: {len(df)}")
    print(f"  TP: {int(tp)}  FP: {int(fp)}  FN: {int(fn)}")
    print(f"  GT objects: {int(tp + fn)}")
    mean_iou = df[df['is_tp']==1]['iou'].mean() if tp > 0 else 0
    print(f"  Mean IoU (TP): {mean_iou:.4f}")


# ## 4. Assign Difficulty Labels

# In[ ]:


# ---- TUNABLE THRESHOLDS (modify these as needed) ----
# Area thresholds (px²) at 1024x1024 reference frame
AREA_THRESH_MOD = 32 * 32       # 1024 px² -> below = Hard
AREA_THRESH_EASY = 96 * 96      # 9216 px² -> above = Easy
# Aspect ratio threshold
ASPECT_THRESHOLD = 3.0
# Confidence thresholds
CONF_THRESH_MOD = 0.4
CONF_THRESH_EASY = 0.7
# -----------------------------------------------------

for model_name, df in all_results.items():
    df["difficulty_area"] = df["area_ref_px"].apply(
        lambda a: classify_by_area(a, (AREA_THRESH_MOD, AREA_THRESH_EASY)))
    df["difficulty_aspect"] = df["aspect_ratio"].apply(
        lambda a: classify_by_aspect_ratio(a, ASPECT_THRESHOLD))
    df["difficulty_conf"] = df["confidence"].apply(
        lambda c: classify_by_confidence(c, (CONF_THRESH_MOD, CONF_THRESH_EASY)))

    diff_comp = df.apply(
        lambda r: composite_difficulty(r["area_ref_px"], r["aspect_ratio"], r["confidence"]),
        axis=1, result_type="expand")
    df["difficulty_composite"] = diff_comp[0]
    df["composite_score"] = diff_comp[1]

print("Difficulty labels assigned.")
print(f"\nThresholds used (at {IMG_REF_W}x{IMG_REF_H} reference frame):")
print(f"  Area: Hard <= {AREA_THRESH_MOD}, Moderate <= {AREA_THRESH_EASY}, Easy > {AREA_THRESH_EASY}")
print(f"  Aspect ratio: Easy <= {ASPECT_THRESHOLD}, Moderate <= {ASPECT_THRESHOLD*2}, Hard > {ASPECT_THRESHOLD*2}")
print(f"  Confidence: Easy >= {CONF_THRESH_EASY}, Moderate >= {CONF_THRESH_MOD}, Hard < {CONF_THRESH_MOD}")


# In[ ]:


# Quick summary of distribution
for model_name, df in all_results.items():
    print(f"\n{model_name}")
    print("Composite difficulty distribution:")
    print(df["difficulty_composite"].value_counts().to_string())


# ## 5. Performance Metrics by Difficulty Level

# In[ ]:


def compute_metrics_by_difficulty(df, difficulty_col="difficulty_composite"):
    """Compute precision, recall, F1, average IoU per difficulty level."""
    levels = ["Easy", "Moderate", "Hard"]
    rows = []
    for level in levels:
        sub = df[df[difficulty_col] == level]
        if len(sub) == 0:
            continue
        tp = int(sub["is_tp"].sum())
        preds = sub[sub["confidence"] > 0]
        n_pred = len(preds)
        fn = int((sub["confidence"] == 0).sum())
        n_gt = tp + fn  # each GT is either TP or FN
        fp = n_pred - tp

        precision = tp / max(n_pred, 1)
        recall = tp / max(n_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        avg_iou = sub[sub["is_tp"] == 1]["iou"].mean() if tp > 0 else 0.0
        avg_conf = preds["confidence"].mean() if len(preds) > 0 else 0.0

        rows.append({
            "Difficulty": level,
            "GT Objects": n_gt,
            "Predictions": n_pred,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Avg IoU (TP)": avg_iou,
            "Avg Confidence": avg_conf,
        })
    return pd.DataFrame(rows).set_index("Difficulty")


metrics_by_model = {}
for model_name, df in all_results.items():
    metrics_by_model[model_name] = compute_metrics_by_difficulty(df, "difficulty_composite")
    print(f"\n{'='*50}")
    print(f"  {model_name} - Performance by Difficulty")
    print('='*50)
    print(metrics_by_model[model_name].to_string())
    print()


# ## 6. Comparison: Metrics by Area, Aspect Ratio, and Confidence

# In[ ]:


for criterion in ["difficulty_area", "difficulty_aspect", "difficulty_conf"]:
    print(f"\n{'='*50}")
    print(f"Criterion: {criterion}")
    print('='*50)
    for model_name, df in all_results.items():
        print(f"\n--- {model_name} ---")
        m = compute_metrics_by_difficulty(df, criterion)
        print(m[["GT Objects", "Predictions", "Precision", "Recall", "F1-Score", "Avg IoU (TP)"]].to_string())
        print()


# ## 7. Visualizations

# In[ ]:


colors = {"Easy": "#2ecc71", "Moderate": "#f39c12", "Hard": "#e74c3c"}
level_order = ["Easy", "Moderate", "Hard"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, metric in enumerate(["Precision", "Recall", "F1-Score"]):
    ax = axes[0, i]
    for model_name in all_results:
        m = metrics_by_model[model_name]
        vals = [m.loc[lv][metric] if lv in m.index else 0 for lv in level_order]
        ax.plot(level_order, vals, marker='o', linewidth=2.5, markersize=8, label=model_name)
    ax.set_title(f"{metric} by Composite Difficulty", fontsize=13, fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()

for i, (criterion, title) in enumerate([
    ("difficulty_area", "Area-Based"),
    ("difficulty_aspect", "Aspect Ratio-Based"),
    ("difficulty_conf", "Confidence-Based"),
]):
    ax = axes[1, i]
    for model_name, df in all_results.items():
        m = compute_metrics_by_difficulty(df, criterion)
        vals = [m.loc[lv]["F1-Score"] if lv in m.index else 0 for lv in level_order]
        ax.plot(level_order, vals, marker='s', linewidth=2.5, markersize=8, label=model_name)
    ax.set_title(f"F1-Score by {title} Difficulty", fontsize=13, fontweight='bold')
    ax.set_ylabel("F1-Score")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "difficulty_analysis_comparison.png", dpi=SAVE_DPI)
plt.show()
print(f"Saved to {OUTPUT_DIR / 'difficulty_analysis_comparison.png'}")


# In[ ]:


# Distribution of difficulty levels across models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (model_name, df) in zip(axes, all_results.items()):
    counts = df["difficulty_composite"].value_counts().reindex(level_order, fill_value=0)
    bars = ax.bar(level_order, counts.values, color=[colors[l] for l in level_order], edgecolor='white')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts.values)*0.02,
                str(val), ha='center', fontsize=11, fontweight='bold')
    ax.set_title(f"{model_name} — Difficulty Distribution", fontsize=13, fontweight='bold')
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "difficulty_distribution.png", dpi=SAVE_DPI)
plt.show()
print(f"Saved to {OUTPUT_DIR / 'difficulty_distribution.png'}")


# In[ ]:


# Scatter: Area vs Confidence, colored by TP/FP
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (model_name, df) in zip(axes, all_results.items()):
    preds = df[df["confidence"] > 0].copy()
    colors_scatter = preds["is_tp"].map({1: "#2ecc71", 0: "#e74c3c"})
    ax.scatter(preds["area_ref_px"], preds["confidence"], c=colors_scatter,
               alpha=0.5, s=15, edgecolors='none')
    ax.set_xlabel("BBox Area (px² at 1024x1024 ref)")
    ax.set_ylabel("Confidence")
    ax.set_title(f"{model_name} — Area vs Confidence", fontsize=13, fontweight='bold')
    ax.axhline(CONF_THRESH_EASY, color='gray', ls='--', alpha=0.5, label=f'Easy conf >={CONF_THRESH_EASY}')
    ax.axhline(CONF_THRESH_MOD, color='gray', ls=':', alpha=0.5, label=f'Mod conf >={CONF_THRESH_MOD}')
    ax.axvline(AREA_THRESH_EASY, color='green', ls='--', alpha=0.3)
    ax.axvline(AREA_THRESH_MOD, color='orange', ls='--', alpha=0.3)
    ax.set_xscale('symlog')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "area_vs_confidence_scatter.png", dpi=SAVE_DPI)
plt.show()


# In[ ]:


# IoU distribution by difficulty level
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (model_name, df) in zip(axes, all_results.items()):
    tp_df = df[(df["is_tp"] == 1) & (df["difficulty_composite"].isin(level_order))]
    data_by_level = [tp_df[tp_df["difficulty_composite"] == lv]["iou"].values for lv in level_order]
    bp = ax.boxplot(data_by_level, tick_labels=level_order, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors[l] for l in level_order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(f"{model_name} — IoU by Difficulty", fontsize=13, fontweight='bold')
    ax.set_ylabel("IoU")
    ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "iou_by_difficulty.png", dpi=SAVE_DPI)
plt.show()


# ## 8. Summary Comparison Table

# In[ ]:


summary_rows = []
for model_name, df in all_results.items():
    tp = int(df["is_tp"].sum())
    preds_df = df[df["confidence"] > 0]
    n_pred = len(preds_df)
    fn = int((df["confidence"] == 0).sum())
    n_gt = tp + fn
    fp = n_pred - tp
    precision = tp / max(n_pred, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    avg_iou = df[df["is_tp"] == 1]["iou"].mean() if tp > 0 else 0
    avg_area_tp = df[(df["is_tp"] == 1)]["area_ref_px"].mean() if tp > 0 else 0
    avg_area_fn = df[(df["confidence"] == 0)]["area_ref_px"].mean() if fn > 0 else 0

    summary_rows.append({
        "Model": model_name,
        "GT Objects": n_gt,
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1-Score": f"{f1:.4f}",
        "Avg IoU (TP)": f"{avg_iou:.4f}",
        "Avg Area TP (ref px²)": f"{avg_area_tp:.0f}",
        "Avg Area FN (ref px²)": f"{avg_area_fn:.0f}",
    })

df_summary = pd.DataFrame(summary_rows).set_index("Model")
print("=" * 80)
print("OVERALL DETECTION PERFORMANCE")
print("=" * 80)
print(df_summary.to_string())

# Difficulty-level comparison
print("\n" + "=" * 80)
print("PERFORMANCE BY COMPOSITE DIFFICULTY")
print("=" * 80)
for model_name in all_results:
    print(f"\n{model_name}:")
    print(metrics_by_model[model_name][["GT Objects", "Predictions", "Precision", "Recall", "F1-Score", "Avg IoU (TP)"]].to_string())

df_summary.to_csv(OUTPUT_DIR / "difficulty_analysis_summary.csv")
print(f"\nSaved summary to {OUTPUT_DIR / 'difficulty_analysis_summary.csv'}")


# ## 9. Difficulty Threshold Configuration Guide
# 
# Modify the thresholds in **Section 4** to adjust difficulty classification:
# 
# | Parameter | Variable | Default | Notes |
# |---|---|---|---|
# | Area (Hard/Mod) | `AREA_THRESH_MOD` | 1024 px² (32×32) | Below this = Hard |
# | Area (Mod/Easy) | `AREA_THRESH_EASY` | 9216 px² (96×96) | Above this = Easy |
# | Aspect Ratio | `ASPECT_THRESHOLD` | 3.0 | Easy ≤ 3×, Moderate ≤ 6×, Hard > 6× |
# | Confidence (Mod/Easy) | `CONF_THRESH_EASY` | 0.7 | ≥ 0.7 = Easy |
# | Confidence (Hard/Mod) | `CONF_THRESH_MOD` | 0.4 | ≥ 0.4 = Moderate |
# 
# The **composite difficulty** combines all three signals with weights:
# - Area: 40%
# - Aspect Ratio: 30%
# - Confidence: 30%
# 
# All areas are scaled to a reference **1024×1024** frame for consistency across varying image sizes.
