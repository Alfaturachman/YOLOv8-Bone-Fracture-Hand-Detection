import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

MODELS = {
    "YOLOv8m-AdamW": {"TP": 183, "FP": 32, "FN": 70, "TN": 156},
    "YOLOv8s-AdamW": {"TP": 184, "FP": 26, "FN": 69, "TN": 157},
}

CLASS_LABELS = ["Non-Fracture", "Fracture"]

TITLE_TEMPLATE = "Confusion Matrix"

# 🔥 NORMALIZED MODE (untuk jurnal)
NORMALIZE = "true"   # "true" | "pred"

CMAP = "Blues"

FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK = 10
FONT_VALUE = 10

FIG_SIZE = (7, 6)
SAVE_DPI = 200
OUTPUT_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════

def build_matrix(vals: dict) -> np.ndarray:
    return np.array([
        [vals["TN"], vals["FP"]],
        [vals["FN"], vals["TP"]],
    ], dtype=float)


def normalize_matrix(cm: np.ndarray, mode: str) -> np.ndarray:
    if mode == "true":  # per baris (Recall)
        row_sum = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, row_sum, where=row_sum != 0)

    elif mode == "pred":  # per kolom (Precision)
        col_sum = cm.sum(axis=0, keepdims=True)
        return np.divide(cm, col_sum, where=col_sum != 0)

    return cm


def plot_cm(cm_raw: np.ndarray, model_name: str, output_path: Path):

    # 🔥 SWAP AXIS (TRUE di X-axis)
    cm_norm = normalize_matrix(cm_raw.copy(), NORMALIZE).T

    print(f"\nNormalized Confusion Matrix - {model_name}")
    print(cm_norm)  

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    im = ax.imshow(cm_norm, cmap=CMAP, vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_formatter(
        ticker.PercentFormatter(xmax=1, decimals=0)
    )

    # Labels
    ticks = np.arange(len(CLASS_LABELS))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(CLASS_LABELS, fontsize=FONT_TICK)
    ax.set_yticklabels(CLASS_LABELS, fontsize=FONT_TICK, rotation=90, va="center")

    ax.set_xlabel("True Label", fontsize=FONT_LABEL)
    ax.set_ylabel("Predicted Label", fontsize=FONT_LABEL)

    # Title
    ax.set_title(TITLE_TEMPLATE, fontsize=FONT_TITLE, pad=10)

    # 🔥 Annotation (% only, clean journal style)
    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"

            ax.text(
                j, i,
                f"{val:.2%}",
                ha="center",
                va="center",
                fontsize=FONT_VALUE,
                fontweight="bold",
                color=color
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {output_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for model_name, vals in MODELS.items():
        cm_raw = build_matrix(vals)

        safe_name = model_name.replace(" ", "_").replace("/", "-")
        out_path = OUTPUT_DIR / f"cm_normalized_{safe_name}.png"

        print(f"\n[{model_name}]")

        plot_cm(cm_raw, model_name, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()