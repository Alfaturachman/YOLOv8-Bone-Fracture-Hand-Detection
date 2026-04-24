# JUDUL
**Optimizing Bone Fracture Detection in Hand X-Ray Images Using YOLOv8m: An Undersampling Approach to Address Class Imbalance on the FracAtlas Dataset**

## Brief Informasi:
Paper ini berfokus pada **deteksi fraktur (patah tulang) tangan pada citra medis X-Ray** dengan menggunakan dataset **FracAtlas**. Jumlah dataset yang digunakan difilter khusus untuk citra bagian tangan (hand) yang terdiri dari **1,538 gambar** (sekitar 267 gambar berlabel fraktur dan 1,271 gambar non-fraktur). 
**Novelty (Kebaruan):** Berbeda dengan literatur deteksi objek medis sebelumnya yang seringkali mengabaikan masalah *class imbalance* (ketidakseimbangan kelas) atau menggunakan model standar, penelitian ini memiliki keunggulan dalam:
1. **Strategi Undersampling Terarah:** Mengatasi ketidakseimbangan kelas yang parah (~17.4% fraktur vs 82.6% non-fraktur) dengan teknik undersampling pada data latih untuk mencegah bias model terhadap kelas dominan (non-fraktur).
2. **Optimasi YOLOv8m (Medium) Khusus Medis:** Menggunakan dan mengoptimasi parameter YOLOv8m yang lebih *robust* dibanding versi dasar YOLOv8s/n, disertai skema augmentasi data khusus citra medis (misalnya menghindari rotasi ekstrem, perspektif, atau shear yang dapat mengubah struktur anatomis tulang).

---

## 1. Introduction
- **Clinical and Technical Background:** Fraktur tangan adalah kasus trauma yang sangat umum terjadi. Diagnosis cepat dan akurat melalui citra X-Ray sangat penting di IGD. Secara teknis, *Deep Learning* (terutama arsitektur YOLO - *You Only Look Once*) telah terbukti efektif untuk deteksi objek *real-time*.
- **Gap in Prior Work:** Banyak studi sebelumnya yang menggunakan model YOLO pada dataset X-Ray gabungan tanpa memperhatikan spesifik *body part* dan tidak menangani *class imbalance* (lebih banyak gambar tulang sehat dibanding patah tulang), sehingga menghasilkan model dengan akurasi tinggi namun *Recall* yang rendah untuk kasus fraktur.
- **Study Objective and Contribution:** Penelitian ini bertujuan meningkatkan akurasi deteksi fraktur tangan dengan model YOLOv8m. Kontribusi utama adalah penerapan teknik *undersampling* pada kelas minoritas (fraktur) vs mayoritas (non-fraktur) dan evaluasi dampaknya terhadap metrik *Recall* dan *mAP*.

## 2. Material dan Method
- **Flowchart & Study Design Overview:** Desain studi kuantitatif eksperimental yang membandingkan performa model YOLO sebelum dan sesudah penanganan *class imbalance*.
- **Dataset Details:** Menggunakan dataset *open-source* FracAtlas yang telah difilter hanya untuk `Hand`. Terdiri dari 1,538 gambar resolusi tinggi (~1760 x 2140 pixels) dengan *bounding box annotations* (format YOLO).
- **Preprocessing:** Meliputi normalisasi resolusi menjadi 1024 piksel, *stratified split* (70% Train, 15% Val, 15% Test), dan strategi *undersampling* pada data *training* (mengurangi jumlah sampel gambar tangan normal agar seimbang dengan jumlah gambar tangan fraktur). Augmentasi yang digunakan difokuskan pada manipulasi HSV dan horizontal flip tanpa merusak orientasi vertikal tulang.
- **Detection Model and Training:** Menggunakan algoritma YOLOv8m (*Medium*). *Training* dilakukan dengan optimasi hyperparameter: *learning rate* (`lr0` dan `lrf`), *momentum*, *weight decay*, dan *optimizer* (AdamW) yang dijalankan hingga batas 100 *epochs* dengan fitur *early stopping*.
- **Evaluation:** Metrik evaluasi utama berfokus pada *mAP@50*, *mAP@50-95*, *Precision*, *Recall*, dan skor F1. *Recall* diutamakan karena pada kasus klinis *false negatives* (gagal mendeteksi tulang patah) sangat berisiko.

## 3. Results
- **Overall Performance:** Menunjukkan nilai *mAP@50* model YOLOv8m setelah menggunakan dataset seimbang. Model dengan *undersampling* menghasilkan deteksi dengan generalisasi yang lebih baik.
- **Specific Performance:** *Recall* (kemampuan model untuk menemukan semua area yang fraktur) mengalami peningkatan yang signifikan pada kelas *fracture* setelah teknik *undersampling* diterapkan dibandingkan dataset asli yang *imbalanced*. Terdapat tabel komparatif performa deteksi *bounding box*.
- **Representative Qualitative Cases:** Visualisasi perbandingan *ground truth* (label asli) dengan *prediction box* dari model pada data Test. Termasuk contoh kasus di mana model berhasil mendeteksi *micro-fracture* yang sulit terlihat kasat mata, serta contoh kegagalan model (False Positives/Negatives).

## 4. Discussion
- **Principal Findings:** Apakah model yang digunakan benar-benar efektif? Ya, pendekatan model YOLOv8m dipadukan dengan data *balancing* (undersampling) sangat efektif meningkatkan sensitivitas deteksi (*Recall*).
- **Why Model (Kenapa ambil model itu):** YOLOv8m dipilih karena memberikan keseimbangan terbaik (*sweet spot*) antara beban komputasi (*speed inference* untuk kemungkinan implementasi *real-time* di IGD) dan akurasi (jumlah parameter ~20M) dibandingkan YOLOv8n yang terlalu sederhana, atau YOLOv8x yang terlalu berat.
- **Comparison with Prior Detection Literature:** Membandingkan hasil *mAP* penelitian ini dengan *paper* lain yang menggunakan algoritma Faster R-CNN, SSD, atau YOLO versi lama (v5/v7) pada dataset FracAtlas atau dataset tulang lainnya. Menyoroti bagaimana penanganan *imbalance* pada *paper* ini membedakannya dari *paper* lain.
- **Strengths and Limitation:** 
  - **Kelebihan:** Fokus spesifik pada area anatomi tangan menghilangkan fitur bias dari anggota tubuh lain. Penanganan dataset tidak seimbang mengurangi *overfitting*.
  - **Kekurangan:** Ukuran data latih berkurang (akibat *undersampling*), serta model masih bisa mengalami kesulitan jika tulang terhalang oleh perangkat medis implant (*Hardware*).

## 5. Conclusion
Penerapan arsitektur **YOLOv8m** yang dipadukan dengan teknik **undersampling** terbukti menjadi solusi yang lebih baik dan efektif dalam menangani kasus deteksi fraktur tulang tangan pada dataset FracAtlas. Pendekatan ini secara krusial berhasil mengatasi *class imbalance*, sehingga mencegah model mendominasi prediksi sebagai "tulang normal". Hasilnya, metrik sensitivitas (*Recall*) meningkat signifikan, memastikan bahwa model kecerdasan buatan dapat lebih diandalkan untuk mendeteksi area fraktur yang sesungguhnya dan mengurangi risiko kesalahan diagnosis di lingkungan klinis dibandingkan menggunakan distribusi data bawaan.

## 6. Declaration
- **Ethics approval:** Tidak diperlukan karena penelitian menggunakan kumpulan data gambar anonim *open-source* (FracAtlas) yang tersedia untuk publik secara bebas.
- **Data availability:** Dataset yang mendasari artikel ini (FracAtlas) tersedia di repositori publik (sebutkan link/sitasi). Skrip pemrosesan untuk sub-dataset *hand* disertakan pada kode penelitian.
- **Code availability:** *Source code* untuk pra-pemrosesan, *training* YOLOv8m, dan evaluasi dapat diakses secara publik pada repositori GitHub penulis: [Link GitHub]
- **Funding:** Tidak ada (*atau sebutkan sumber pendanaan jika ada*).
- **Conflict of interest:** Para penulis menyatakan tidak ada konflik kepentingan terkait penyusunan artikel penelitian ini.
- **Author contributions:** Penulis 1 merancang studi, melakukan eksperimen (*coding* model dan dataset *preprocessing*), dan menulis *draft* utama. Penulis 2 memvalidasi evaluasi klinis dan mengoreksi manuskrip (*Sesuaikan jika perlu*).
- **Acknowledgments:** Terima kasih kepada penyedia *open-source* dataset FracAtlas dan *framework* Ultralytics YOLOv8 yang sangat menunjang penelitian ini.
