import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Baris yang salah impor sudah dihapus

# Data matriks berdasarkan nilai Recall tabel Anda
matrices = {
    "fracatlas_yolov8s_imbalance": np.array([[0.48, 1.00], [0.52, 0.00]]),
    "fracatlas_yolov8m_imbalance": np.array([[0.47, 1.00], [0.53, 0.00]]),
    "fracatlas_yolov8s_oversampled": np.array([[0.72, 1.00], [0.28, 0.00]]),
    "fracatlas_yolov8m_oversampled": np.array([[0.81, 1.00], [0.19, 0.00]]),
}

labels = ["fracture", "background"]

# Melakukan looping untuk membuat dan menyimpan setiap gambar secara terpisah
for title, matrix in matrices.items():
    # Membuat 1 canvas baru ukuran 7x6 inci untuk setiap model
    fig, ax = plt.subplots(figsize=(7, 6))

    # Membuat kustomisasi teks agar angka 0.00 menjadi string kosong ""
    annot_labels = np.where(matrix == 0, "", np.char.mod("%.2f", matrix))

    # Menggambar heatmap dengan anotasi teks yang sudah dikustomisasi
    sns.heatmap(
        matrix,
        annot=annot_labels,  # Menggunakan teks kustom yang mengosongkan angka 0
        fmt="",  # Kosongkan format otomatis karena data sudah berbentuk string
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        cbar=True,
    )

    # Ekstrak nama model saja (yolov8s atau yolov8m) lalu ubah huruf pertamanya jadi kapital (Yolov8s / Yolov8m)
    if "yolov8s" in title:
        model_name = "Yolov8s"
    else:
        model_name = "Yolov8m"

    # Mengatur label sumbu dan judul atas sesuai format yang Anda minta
    ax.set_title(f"Confusion Matrix Normalized - {model_name}", fontsize=12, pad=12)
    ax.set_xlabel("True", fontsize=11, labelpad=10)
    ax.set_ylabel("Predicted", fontsize=11, labelpad=10)

    # Mengatur margin otomatis agar teks tidak terpotong saat disimpan
    plt.tight_layout()

    # Menyimpan gambar menjadi file PNG dengan kualitas tinggi (dpi=300)
    filename = f"{title}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Berhasil menyimpan: {filename}")

    # Menutup plot saat ini agar memori tidak penuh
    plt.close(fig)
