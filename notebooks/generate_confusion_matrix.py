"""
generate_confusion_matrix.py
────────────────────────────
Generate confusion matrix figures untuk jurnal.
Edit bagian CONFIG di bawah sesuai kebutuhan.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG — EDIT SESUAI KEBUTUHAN
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "YOLOv8s-Imbalanced": {"TP": 133, "FP": 42, "FN": 144, "TN": 165},
    "YOLOv8s-Oversampling": {"TP": 200, "FP": 23, "FN": 77, "TN": 165},
    "YOLOv8m-Imbalanced": {"TP": 130, "FP": 32, "FN": 147, "TN": 165},
    "YOLOv8m-Oversampling": {"TP": 225, "FP": 48, "FN": 52, "TN": 165},
}

# Label sumbu — ganti sesuai domain jurnal
CLASS_LABELS = ["Non-Fracture", "Fracture"]   # [negatif, positif]

# Judul plot — {model} akan diganti nama model secara otomatis
TITLE_TEMPLATE = "Confusion Matrix"

# Normalisasi: "none" | "true" (per baris / recall) | "pred" (per kolom / precision)
NORMALIZE = "none"

# Warna colormap: "Blues" | "Greens" | "Oranges" | "RdBu" | "viridis"
CMAP = "Blues"

# Ukuran font
FONT_TITLE   = 14
FONT_LABEL   = 13
FONT_TICK    = 12
FONT_VALUE   = 16    # angka di dalam cell

# Ukuran figure (inch)
FIG_SIZE = (7, 6)

# DPI output
SAVE_DPI = 200

# Folder output (relatif ke lokasi script ini)
OUTPUT_DIR = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════════════════════


def build_matrix(vals: dict) -> np.ndarray:
    """
    Susun 2×2 confusion matrix dari dict {TP, FP, FN, TN}.

    Layout standar (scikit-learn / sklearn):
        rows = Actual,   cols = Predicted
        [[TN, FP],
         [FN, TP]]
    """
    return np.array([
        [vals["TN"], vals["FP"]],
        [vals["FN"], vals["TP"]],
    ], dtype=float)


def normalize_matrix(cm: np.ndarray, mode: str) -> np.ndarray:
    if mode == "true":          # per baris  → recall-like
        row_sum = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, row_sum, where=row_sum != 0)
    elif mode == "pred":        # per kolom  → precision-like
        col_sum = cm.sum(axis=0, keepdims=True)
        return np.divide(cm, col_sum, where=col_sum != 0)
    return cm                   # "none" → raw counts


def plot_cm(cm_raw: np.ndarray, model_name: str, output_path: Path):
    cm_display = normalize_matrix(cm_raw.copy(), NORMALIZE).T
    cm_raw = cm_raw.T
    vmax = 1.0 if NORMALIZE != "none" else None

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    im = ax.imshow(cm_display, interpolation="nearest", cmap=CMAP,
                   vmin=0, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    if NORMALIZE != "none":
        cbar.set_label("Proportion", fontsize=FONT_LABEL - 1)
        cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # Tick labels
    ticks = np.arange(len(CLASS_LABELS))
    ax.set_xticks(ticks)
    ax.set_xticklabels(CLASS_LABELS, fontsize=FONT_TICK)
    ax.set_yticks(ticks)
    ax.set_yticklabels(CLASS_LABELS, fontsize=FONT_TICK, rotation=90, va="center")

    # Axis labels
    ax.set_xlabel("True Label", fontsize=FONT_LABEL, labelpad=8)
    ax.set_ylabel("Predicted Label", fontsize=FONT_LABEL, labelpad=8)

    # Judul
    title = TITLE_TEMPLATE.format(model=model_name)
    ax.set_title(title, fontsize=FONT_TITLE, pad=10)

    # Nilai di tiap cell
    thresh = cm_display.max() / 2.0
    for i in range(2):
        for j in range(2):
            raw   = int(cm_raw[i, j])
            disp  = cm_display[i, j]
            color = "white" if disp > thresh else "black"

            if NORMALIZE != "none":
                label = f"{disp:.1%}\n({raw})"
            else:
                label = str(raw)

            ax.text(j, i, label,
                    ha="center", va="center",
                    fontsize=FONT_VALUE, fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, vals in MODELS.items():
        cm_raw = build_matrix(vals)

        # Hitung metrik ringkas
        tp, fp, fn, tn = vals["TP"], vals["FP"], vals["FN"], vals["TN"]
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        acc    = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

        safe_name = model_name.replace(" ", "_").replace("/", "-")
        out_path  = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"

        print(f"\n[{model_name}]")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  Accuracy  : {acc:.4f}")

        plot_cm(cm_raw, model_name, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
