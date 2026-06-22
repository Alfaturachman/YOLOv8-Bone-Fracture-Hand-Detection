# JUDUL

**Comparative Analysis of YOLOv8s and YOLOv8m with Oversampling Strategy for Hand Bone Fracture Detection on X-Ray Images Using the FracAtlas Dataset**

> **Judul Alternatif:**
> Evaluating Oversampling for Imbalanced Hand Fracture Detection with YOLOv8 Models

---

## Abstrak

**Bahasa Indonesia:** Deteksi fraktur tulang tangan pada citra sinar-X merupakan tantangan klinis penting yang memerlukan sistem diagnosis berbantuan komputer (_computer-aided diagnosis_/CAD) yang akurat dan efisien. Penelitian ini mengevaluasi dan membandingkan dua varian arsitektur YOLOv8, yaitu YOLOv8s (_small_) dan YOLOv8m (_medium_), untuk mendeteksi fraktur tangan menggunakan _dataset_ FracAtlas yang difilter khusus untuk citra tangan (1.538 citra, rasio fraktur 1:2,5). Strategi penyampelan berlebih (_oversampling_) diterapkan pada _subset_ pelatihan (_training_) untuk mengatasi ketidakseimbangan kelas, sedangkan _subset_ validasi dan pengujian dipertahankan dalam kondisi tidak seimbang (_imbalanced_) asli untuk menjaga objektivitas evaluasi. Hasil eksperimen menunjukkan bahwa YOLOv8m unggul pada _dataset_ dengan _oversampling_ dengan nilai _mean Average Precision_ (mAP@50) sebesar 0,8320, _Recall_ sebesar 0,8124, dan _F1-Score_ sebesar 0,8178, mengungguli YOLOv8s yang mencapai mAP@50 sebesar 0,8150 dan _Recall_ sebesar 0,7232. Penerapan _oversampling_ memberikan peningkatan _Recall_ hingga +73,2% pada YOLOv8m dan +50,2% pada YOLOv8s dibandingkan dengan nilai acuan (_baseline_) tidak seimbang. Meskipun YOLOv8m menunjukkan kinerja yang lebih unggul, YOLOv8s menawarkan efisiensi komputasi yang lebih baik dengan jumlah parameter yang lebih sedikit (11,16 juta vs 25,84 juta), dengan selisih mAP@50 hanya sebesar 0,017.

**English:** Hand bone fracture detection on X-Ray images is a critical clinical challenge requiring accurate and efficient computer-aided diagnosis systems. This study evaluates and compares two YOLOv8 architecture variants, YOLOv8s (small) and YOLOv8m (medium), for hand fracture detection using a hand-filtered subset of the FracAtlas dataset (1,538 images, fracture ratio 1:2.5). An oversampling strategy was applied to the training subset to address class imbalance, while validation and test subsets remained in their original imbalanced condition to preserve evaluation objectivity. Experimental results show that YOLOv8m achieves superior performance on the oversampled dataset with mAP@50 of 0.8320, Recall of 0.8124, and F1-Score of 0.8178, outperforming YOLOv8s which achieved mAP@50 of 0.8150 and Recall of 0.7232. Oversampling improved Recall by up to +73.2% on YOLOv8m and +50.2% on YOLOv8s compared to the imbalanced baseline. Despite YOLOv8m's superior performance, YOLOv8s offers better computational efficiency with only 11.16M parameters and an mAP@50 gap of only 0.017 against YOLOv8m.

---

## 1. Introduction

### 1.1 Clinical and Technical Background

Fraktur tangan merupakan salah satu kasus trauma yang paling sering ditangani di Instalasi Gawat Darurat (IGD), menyumbang sekitar 40% dari seluruh kasus fraktur yang terdata [1]. Diagnosis yang cepat dan akurat melalui pemeriksaan citra sinar-X sangat krusial untuk mencegah kecacatan jangka panjang akibat keterlambatan penanganan. Pembelajaran mendalam (_deep learning_), khususnya arsitektur _You Only Look Once_ (YOLO), telah menunjukkan kinerja yang menjanjikan dalam berbagai tugas deteksi pada citra medis, dengan tingkat akurasi yang kompetitif dibandingkan dengan ahli radiologi pada beberapa domain medis [2].

### 1.2 Problem: Class Imbalance dalam Dataset Medis

Tantangan utama dalam membangun model deteksi fraktur berbasis _deep learning_ adalah **ketidakseimbangan kelas** (_class imbalance_) yang melekat (_inheren_) pada _dataset_ medis—di mana kondisi patologis (fraktur) secara alami jauh lebih jarang terjadi dibandingkan dengan kondisi normal. Pada _dataset_ FracAtlas _subset_ tangan, distribusi ini menunjukkan rasio 1:2,5 (28,5% fraktur vs 71,5% non-fraktur). Beberapa studi menunjukkan bahwa melatih model deteksi pada _dataset_ yang tidak seimbang (_imbalanced_) menghasilkan model yang bias terhadap kelas mayoritas. Hal ini mengakibatkan nilai _Recall_ yang rendah untuk kelas minoritas [1, 3], sehingga mengurangi keandalan sistem deteksi karena kasus fraktur berisiko tinggi untuk tidak teridentifikasi.

### 1.3 Gap in Prior Work

Hingga saat ini, belum ada penelitian yang secara khusus mengevaluasi pengaruh strategi _oversampling_ pada deteksi fraktur tangan menggunakan YOLOv8s dan YOLOv8m. Sebagian besar penelitian deteksi fraktur berbasis YOLO: (1) menggunakan _dataset_ sinar-X gabungan dari berbagai bagian tubuh (_multi-body part_) sehingga model tidak spesifik terhadap anatomi tertentu, (2) jarang membahas strategi eksplisit untuk menangani masalah ketidakseimbangan kelas, dan (3) hanya mengevaluasi satu varian model tanpa melakukan perbandingan sistematis yang mempertimbangkan kompromi (_trade-off_) antara akurasi dan kecepatan untuk implementasi klinis.

### 1.4 Study Objective and Contribution

Penelitian ini bertujuan untuk:
(a) Mengevaluasi dan membandingkan secara komprehensif kinerja YOLOv8s dan YOLOv8m dalam mendeteksi fraktur tangan menggunakan _dataset_ FracAtlas.
(b) Menerapkan alur pemrosesan awal (_preprocessing pipeline_) yang tepat dengan metode _oversampling_ **hanya pada subset pelatihan** (_training subset_), sementara _subset_ validasi dan pengujian dibiarkan tetap tidak seimbang (_imbalanced_) guna menangani masalah ketidakseimbangan kelas tanpa mengorbankan objektivitas evaluasi.
(c) Menganalisis secara kuantitatif dampak _oversampling_ terhadap metrik performa _Recall_, _F1-Score_, mAP@50, dan mAP@50–95 untuk deteksi fraktur tangan.

---

## 2. Material dan Method

### 2.1 Dataset

- **Sumber:** FracAtlas (_open-source_, teranonimisasi)—_dataset_ sinar-X ortopedi yang terdiri dari 4.083 citra.

- **Subset yang digunakan:** Hanya citra bagian tangan (_hand_)—total **1.538 citra** yang terdiri atas 438 citra dengan fraktur (28,5%) dan 1.100 citra non-fraktur (71,5%).

- **Pembagian Data (_Split_):** Pembagian acak terlapis (_stratified split_) dengan proporsi 70% Pelatihan (_Train_) / 15% Validasi (_Val_) / 15% Pengujian (_Test_).

- **Kebijakan _Oversampling_:** _Oversampling_ **hanya** diterapkan pada _subset_ **pelatihan** (_training_). _Subset_ **validasi** dan **pengujian** dipertahankan dalam kondisi tidak seimbang (_imbalanced_) asli (tanpa modifikasi) untuk menjamin evaluasi model yang objektif serta mencerminkan performa deteksi pada distribusi data klinis yang realistis.

- **Distribusi Asli (Tidak Seimbang):**

    | Bagian (_Split_)       | Jumlah Citra | Citra Fraktur | Bounding Box (Instance) | Rata-rata Box per Citra |
    | :--------------------- | :----------: | :-----------: | :---------------------: | :---------------------: |
    | Pelatihan (_Train_)    |    1.076     |      306      |           383           |         0,3559          |
    | Validasi (_Val_)       |     231      |      66       |           81            |         0,3506          |
    | Pengujian (_Test_)[^*] |     231      |      66       |           85            |         0,3680          |
    | **TOTAL**              |  **1.538**   |    **438**    |         **549**         |       **0,3570**        |

- **Distribusi dengan _Oversampling_:**

    | Bagian (_Split_)         |       Jumlah Citra       |    Citra Fraktur     | Bounding Box (Instance) |  Rata-rata Box per Citra   |
    | :----------------------- | :----------------------: | :------------------: | :---------------------: | :------------------------: |
    | Pelatihan (_Train_)[^**] | 1.540 (atau 1.688[^***]) | 767 (atau 918[^***]) | 945 (atau 1.149[^***])  | 0,6136 (atau 0,6807[^***]) |
    | Validasi (_Val_)         |           231            |          66          |           81            |           0,3506           |
    | Pengujian (_Test_)[^*]   |           231            |          66          |           85            |           0,3680           |
    | **TOTAL**                |  **2.002 (atau 2.150)**  | **899 (atau 1.050)** | **1.111 (atau 1.315)**  |  **0,5550 (atau 0,6116)**  |

[^*]: _Catatan Pengujian: Selama evaluasi oleh YOLO, terdapat 6 citra terdeteksi corrupt/truncated pada folder uji dan otomatis diabaikan, sehingga pengujian berjalan pada 225 citra dengan total 85 instance (rata-rata 0,3778 box/citra)._

[^**]: _Catatan Oversampling Riil: Angka 1.540 citra (945 instance) adalah data aktual pada folder `yolo_dataset_hand_oversampled` yang digunakan dalam pelatihan lokal, menerapkan penyeimbangan persis 1:1 antara kelas fraktur (770 citra) dan non-fraktur (770 citra), di mana beberapa box tereliminasi akibat penyeimbangan/augmentasi._

[^***]: _Catatan Oversampling Teoretis: Angka 1.688 citra (1.149 instance) adalah estimasi kasar dalam rancangan awal dengan mengasumsikan pengali augmentasi bulat (3x lipat dari 306 citra fraktur pelatihan)._

### 2.2 Strategi Preprocessing: Oversampling

Ketidakseimbangan kelas yang signifikan (rasio 1:2,5) diatasi dengan menggunakan teknik **oversampling** pada _subset_ pelatihan. Langkah ini sejalan dengan rekomendasi penelitian sebelumnya [1, 2] yang menyatakan bahwa _oversampling_ pada kelas minoritas secara konsisten dapat meningkatkan performa model klasifikasi maupun deteksi pada _dataset_ medis yang tidak seimbang (_imbalanced_).

Augmentasi pada kelas minoritas dibatasi hanya pada transformasi yang aman secara anatomis untuk citra sinar-X:

| Augmentasi            | Nilai   | Keterangan                                                             |
| --------------------- | ------- | ---------------------------------------------------------------------- |
| _Horizontal Flip_     | p=0,5   | Simetris kiri-kanan aman untuk anatomi tangan                          |
| _Rotation_            | ±10°    | Mengakomodasi variasi posisi pasien                                    |
| _Brightness/Contrast_ | acak    | Mengakomodasi variasi kualitas eksposur sinar-X                        |
| _Translation_         | 5%      | Pergeseran posisi tangan dalam citra                                   |
| _Scale_               | ±15%    | Mengakomodasi variasi ukuran tangan / jarak pasien-detektor            |
| **_Vertical Flip_**   | **0,0** | **Tidak digunakan**—orientasi proksimal-distal sangat penting          |
| **_Shear_**           | **0,0** | **Tidak digunakan**—menghindari distorsi anatomi                       |
| **_Perspective_**     | **0,0** | **Tidak digunakan**—sinar-X merupakan proyeksi ortogonal               |
| **_Hue/Saturation_**  | **0,0** | **Tidak digunakan**—sinar-X merupakan citra berskala abu (_grayscale_) |

Proses _oversampling_ dilakukan dengan ketentuan berikut:

- Citra fraktur direplikasi dengan menerapkan augmentasi ringan (rotasi ±10°, _horizontal flip_) hingga rasio kelas mendekati 1:1.
- **_Dataset_ pelatihan akhir:** Secara teoretis menghasilkan 1.688 citra yang terdiri atas 918 citra fraktur (54,4%) dan 770 citra non-fraktur (45,6%) jika mengasumsikan replikasi bulat 3x lipat pada data fraktur. Namun, pada implementasi riil berkas latih yang disimpan di disk (`yolo_dataset_hand_oversampled`), dihasilkan total 1.540 citra yang terdiri dari 770 citra non-fractured dan 770 citra fractured (termasuk hasil augmentasi). Dari 770 citra fractured tersebut, 3 citra tereliminasi akibat hilangnya seluruh bounding box pasca-augmentasi (sehingga dianggap sebagai _background_), menghasilkan komposisi riil 767 citra teranotasi fraktur (49,8%) dan 773 citra non-fraktur (50,2%).
- **_Val/Test_: Tidak dikenakan proses _oversampling_.** Kedua _subset_ tetap mempertahankan distribusi tidak seimbang asli (28,5% fraktur) untuk menjamin evaluasi model bersifat objektif dan mencerminkan performa deteksi pada kondisi klinis yang nyata. Dengan demikian, setiap peningkatan metrik pada _subset_ pengujian dapat diatribusikan secara valid ke strategi _oversampling_, bukan akibat manipulasi distribusi data evaluasi.

> **Justifikasi literatur:** Albright dkk. [1] menunjukkan bahwa _random oversampling_ merupakan teknik yang sederhana namun sangat efektif untuk menangani masalah ketidakseimbangan kelas pada model pembelajaran mendalam (_deep learning_), terutama ketika ukuran _dataset_ terbatas. Penelitian lain [2] juga mengonfirmasi keunggulan _oversampling_ dibandingkan teknik lain pada _dataset_ berukuran kecil hingga menengah.

### 2.3 Model Architecture

| Komponen           | YOLOv8s (_Small_)  | YOLOv8m (_Medium_) |
| ------------------ | ------------------ | ------------------ |
| Parameter          | ~11,16M            | ~25,89M            |
| GFLOPs (imgsz=640) | 28,7               | 79,1               |
| _Backbone_         | CSPDarknet (C2f)   | C3k2               |
| _Neck_             | PANet FPN          | PANet FPN          |
| _Head_             | _Decoupled Detect_ | _Decoupled Detect_ |
| _Pretrained on_    | COCO (80 kelas)    | COCO (80 kelas)    |

Kedua model ditala halus (_fine-tune_) dari bobot yang telah dilatih sebelumnya (_pretrained weight_) pada _dataset_ COCO, dengan bagian _head_ disesuaikan untuk mendeteksi 1 kelas sasaran (fraktur).

### 2.4 Training Configuration

| Hiperparameter  | Nilai  | Justifikasi                                                                   |
| --------------- | ------ | ----------------------------------------------------------------------------- |
| `optimizer`     | AdamW  | Lebih stabil untuk _dataset_ medis berskala kecil dibandingkan SGD            |
| `lr0`           | 0,001  | Diturunkan dari nilai bawaan (default) 0,01 untuk mencegah _overshoot_        |
| `lrf`           | 0,01   | Rasio laju pembelajaran akhir (_final learning rate_) dengan _cosine decay_   |
| `epochs`        | 100    | Dilengkapi dengan mekanisme penghentian dini (_early stopping_) `patience=20` |
| `imgsz`         | 640    | Menjaga keseimbangan antara detail fraktur dan kapasitas VRAM (RTX 3050 4GB)  |
| `batch`         | 4      | Disesuaikan dengan kapasitas VRAM yang tersedia                               |
| `cos_lr`        | True   | Penerapan _Cosine Learning Rate_ untuk konvergensi yang lebih halus           |
| `warmup_epochs` | 5      | Perpanjangan tahap _warmup_ untuk stabilitas pada awal pelatihan              |
| `weight_decay`  | 0,0001 | Regularisasi ringan untuk mempertahankan fitur-fitur detail kecil             |
| `amp`           | True   | _Automatic Mixed Precision_ untuk efisiensi penggunaan VRAM                   |

### 2.5 Augmentasi Konservatif (X-Ray Specific)

Augmentasi standar pada YOLO dirancang untuk citra natural (_natural images_). Untuk citra sinar-X, beberapa jenis augmentasi dinonaktifkan karena dapat menghasilkan kondisi visual yang tidak mungkin terjadi secara anatomis:

| Parameter       | Nilai   | Alasan                                                                           |
| --------------- | ------- | -------------------------------------------------------------------------------- |
| `degrees`       | 10,0°   | Mensimulasikan variasi posisi tangan pasien saat pengambilan sinar-X             |
| `fliplr`        | 0,5     | Aman digunakan karena anatomi tangan yang relatif simetris                       |
| `flipud`        | **0,0** | **Dinonaktifkan**—orientasi proksimal-distal tulang sangat krusial               |
| `shear`         | **0,0** | **Dinonaktifkan**—distorsi anatomi tidak realistis secara medis                  |
| `perspective`   | **0,0** | **Dinonaktifkan**—sinar-X merupakan proyeksi ortogonal                           |
| `mosaic`        | 0,5     | Dikurangi dari nilai bawaan 1,0—mencegah artefak dari penggabungan citra         |
| `mixup`         | 0,1     | Diberikan dalam intensitas rendah untuk regularisasi                             |
| `copy_paste`    | **0,0** | **Dinonaktifkan**—penempelan area fraktur ke citra lain tidak valid secara medis |
| `hsv_h / hsv_s` | **0,0** | **Dinonaktifkan**—sinar-X adalah citra berskala abu (_grayscale_)                |
| `hsv_v`         | 0,2     | Mensimulasikan variasi eksposur dan densitas tulang antar pasien                 |

### 2.6 Evaluation Metrics

Evaluasi performa model dilakukan dengan menggunakan beberapa metrik yang umum diterapkan pada tugas deteksi objek (_object detection_), yaitu _Precision_, _Recall_, _F1-Score_, mAP@50, mAP@50–95, serta _Confusion Matrix_. Kombinasi metrik ini dipilih untuk memberikan gambaran menyeluruh mengenai kemampuan model dalam mendeteksi fraktur, baik dari aspek akurasi klasifikasi maupun kualitas lokalisasi objek.

Tabel X. Metrik Evaluasi

| Metrik                   | Deskripsi                                                                                                     | Prioritas         |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- | ----------------- |
| **mAP@50**               | _Mean Average Precision_ pada ambang batas IoU ≥ 0,5                                                          | Utama             |
| **mAP@50-95**            | _Mean Average Precision_ pada rentang ambang batas IoU 0,5-0,95                                               | Sekunder          |
| **Recall (Sensitivity)** | Kemampuan model dalam mendeteksi seluruh kasus fraktur yang sebenarnya                                        | **Utama**         |
| **Precision**            | Ketepatan hasil prediksi fraktur yang dihasilkan oleh model                                                   | Sekunder          |
| **F1-Score**             | Keseimbangan nilai antara _Precision_ dan _Recall_                                                            | Sekunder          |
| **Confusion Matrix**     | Distribusi nilai _True Positive_ (TP), _False Positive_ (FP), _False Negative_ (FN), dan _True Negative_ (TN) | Analisis Mendalam |

#### Precision

_Precision_ mengukur proporsi prediksi positif yang benar dibandingkan dengan seluruh prediksi positif yang dihasilkan oleh model. Metrik ini menunjukkan tingkat keakuratan model saat mengidentifikasi keberadaan fraktur.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

di mana:

- **TP** (_True Positive_) = fraktur terdeteksi dengan benar.
- **FP** (_False Positive_) = area non-fraktur yang salah terdeteksi sebagai fraktur.

Nilai _Precision_ yang tinggi menunjukkan bahwa sebagian besar prediksi fraktur yang dihasilkan oleh model memang sesuai dengan kondisi fraktur yang sebenarnya.

#### Recall (Sensitivity)

_Recall_ mengukur kemampuan model dalam mengidentifikasi seluruh kasus fraktur yang terdapat pada _dataset_ pengujian.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

di mana:

- **TP** (_True Positive_) = fraktur terdeteksi dengan benar.
- **FN** (_False Negative_) = fraktur yang gagal terdeteksi oleh model.

Dalam penelitian ini, _Recall_ menjadi metrik yang paling krusial karena tujuan utama dari sistem adalah meminimalkan kasus fraktur yang terlewat. Nilai _Recall_ yang rendah mengindikasikan bahwa masih banyak area fraktur yang tidak berhasil dideteksi oleh model.

#### F1-Score

_F1-Score_ digunakan untuk mengukur titik keseimbangan antara metrik _Precision_ dan _Recall_. Metrik ini sangat relevan digunakan pada _dataset_ yang tidak seimbang (_imbalanced dataset_) karena mempertimbangkan kedua aspek tersebut secara bersamaan.

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Nilai _F1-Score_ yang tinggi menunjukkan bahwa model tidak hanya mampu mendeteksi banyak kasus fraktur, tetapi juga menghasilkan prediksi dengan tingkat akurasi yang relatif tinggi.

#### Mean Average Precision (mAP)

Pada tugas deteksi objek (_object detection_), mAP digunakan sebagai metrik utama untuk mengevaluasi kemampuan model dalam mendeteksi sekaligus melokalisasi posisi objek.

##### mAP@50

mAP@50 dihitung pada ambang batas _Intersection over Union_ (IoU) sebesar 0,5. Prediksi dianggap benar apabila nilai IoU antara kotak pembatas (_bounding box_) prediksi dan acuan dasar (_ground truth_) mencapai minimal 0,5.

$$
\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}
$$

Nilai mAP@50 memberikan gambaran umum mengenai kemampuan deteksi model pada tingkat toleransi lokalisasi yang relatif longgar.

##### mAP@50-95

mAP@50-95 merupakan rata-rata nilai _Average Precision_ pada beberapa ambang batas IoU mulai dari 0,50 hingga 0,95 dengan interval kenaikan sebesar 0,05.

$$
\text{mAP}_{50:95} = \frac{1}{10} \sum_{i=0}^{9} \text{AP}_{@\,\text{IoU}=0.50 + 0.05i}
$$

Metrik ini memiliki kriteria evaluasi yang lebih ketat dibandingkan mAP@50 karena mengharuskan posisi kotak pembatas (_bounding box_) sedekat mungkin dengan lokasi objek yang sebenarnya.

---

## 3. Results

### 3.1 Performance Evaluation on Imbalanced and Oversampled Datasets

| Model   | Dataset      |  Precision |     Recall |   F1-Score |     mAP@50 |  mAP@50-95 |
| ------- | ------------ | ---------: | ---------: | ---------: | ---------: | ---------: |
| YOLOv8s | Imbalanced   |     0,7606 |     0,4815 |     0,5896 |     0,5117 |     0,1974 |
| YOLOv8s | Oversampling | **0,8971** | **0,7232** | **0,8008** | **0,8150** | **0,4210** |
| YOLOv8m | Imbalanced   |     0,8029 |     0,4691 |     0,5923 |     0,4995 |     0,2062 |
| YOLOv8m | Oversampling | **0,8234** | **0,8124** | **0,8178** | **0,8320** | **0,4387** |

Tabel 3.1 menyajikan perbandingan kinerja antara YOLOv8s dan YOLOv8m yang dilatih menggunakan _dataset_ tidak seimbang (_imbalanced_) asli dan _dataset_ dengan strategi _oversampling_. Di bawah konfigurasi _oversampling_, YOLOv8m mencapai kinerja yang lebih unggul dalam hal _Recall_ (0,8124), _F1-Score_ (0,8178), mAP@50 (0,8320), dan mAP@50–95 (0,4387), sedangkan YOLOv8s memperoleh nilai _Precision_ tertinggi (0,8971). Tingkat _Recall_ yang lebih tinggi pada YOLOv8m menunjukkan kemampuan yang lebih kuat dalam mengidentifikasi kemunculan fraktur, sehingga meminimalkan potensi deteksi yang terlewat (_missed detections_). Keunggulan ini juga tecermin dari nilai _F1-Score_ yang lebih tinggi, yang menunjukkan keseimbangan yang lebih baik antara _Precision_ dan _Recall_.

Kedua model mengalami peningkatan performa yang sangat signifikan setelah penerapan strategi _oversampling_ dibandingkan dengan model sejenis yang dilatih pada _dataset_ tidak seimbang. Untuk YOLOv8m, nilai _Recall_ meningkat secara signifikan dari 46,91% menjadi 81,24%, yang merepresentasikan peningkatan sebesar 34,33 poin persentase. Demikian pula, nilai _F1-Score_ meningkat dari 0,5923 menjadi 0,8178, sedangkan nilai mAP@50–95 meningkat dari 0,2062 menjadi 0,4387. Tren yang serupa juga teramati pada YOLOv8s, di mana nilai _Recall_ meningkat dari 48,15% menjadi 72,32%, _F1-Score_ meningkat dari 0,5896 menjadi 0,8008, dan mAP@50–95 meningkat dari 0,1974 menjadi 0,4210.

Hasil eksperimen ini menunjukkan bahwa dampak utama dari strategi _oversampling_ terletak pada peningkatan nilai _Recall_, _F1-Score_, dan mAP@50–95, alih-alih pada metrik _Precision_. Pola ini mengindikasikan bahwa _oversampling_ secara efektif meningkatkan kemampuan model dalam mempelajari karakteristik kelas minoritas (fraktur), sehingga menghasilkan sistem deteksi fraktur yang lebih akurat dan andal. Secara keseluruhan, temuan ini menunjukkan bahwa strategi _oversampling_ berhasil memitigasi dampak buruk dari ketidakseimbangan kelas dan meningkatkan kinerja deteksi secara signifikan untuk kedua arsitektur YOLOv8.

### 3.2 Comparison of Error Distribution Across Datasets

| Model   | Dataset      |  TP |  FP |  FN |
| ------- | ------------ | --: | --: | --: |
| YOLOv8s | Imbalanced   | 133 |  42 | 144 |
| YOLOv8s | Oversampling | 200 |  23 |  77 |
| YOLOv8m | Imbalanced   | 130 |  32 | 147 |
| YOLOv8m | Oversampling | 225 |  48 |  52 |

(4 images from confusion matrix)

Gambar 3.2 merangkum distribusi hasil deteksi berdasarkan metrik _True Positive_ (TP), _False Positive_ (FP), dan _False Negative_ (FN) untuk kedua arsitektur YOLOv8 di bawah kondisi pelatihan tidak seimbang (_imbalanced_) dan _oversampling_.

Hasil evaluasi menunjukkan bahwa strategi _oversampling_ secara signifikan meningkatkan jumlah kasus fraktur yang berhasil dideteksi dengan benar oleh kedua model. YOLOv8s mencatat 200 deteksi positif benar (_true positive_) setelah penerapan _oversampling_, dibandingkan dengan 133 deteksi pada kondisi tidak seimbang. Selaras dengan hal tersebut, YOLOv8m meningkatkan deteksi positif benar dari 130 menjadi 225 kasus. Temuan ini menunjukkan bahwa strategi _oversampling_ memungkinkan model untuk mempelajari karakteristik fraktur secara lebih efektif, sehingga menghasilkan deteksi yang lebih sukses.

Dalam hal kesalahan prediksi, YOLOv8s menghasilkan deteksi positif palsu (_false positive_) yang lebih sedikit dibandingkan YOLOv8m pada kondisi _oversampling_ (23 berbanding 48), sejalan dengan nilai _Precision_ YOLOv8s yang lebih tinggi. Di sisi lain, jumlah negatif palsu (_false negative_) mengalami penurunan yang sangat signifikan pada kedua model setelah penerapan _oversampling_ (dari 144 menjadi 77 pada YOLOv8s, dan dari 147 menjadi 52 pada YOLOv8m).

Saat membandingkan kedua arsitektur, YOLOv8m mencapai jumlah deteksi positif benar tertinggi pada skenario _oversampling_. Hasil ini konsisten dengan metrik kinerja keseluruhan yang disajikan pada Tabel 3.1, di mana YOLOv8m memperoleh nilai tertinggi untuk metrik _Recall_, _F1-Score_, mAP@50, dan mAP@50–95. Secara keseluruhan, temuan ini membuktikan bahwa strategi _oversampling_ berkontribusi langsung pada peningkatan kemampuan deteksi fraktur dengan meningkatkan jumlah kasus yang teridentifikasi secara benar serta mengurangi bias akibat ketidakseimbangan kelas.

### 3.3 Uji Signifikansi Statistik dan Interval Kepercayaan (95% CI)

Untuk mengevaluasi ketidakpastian dan kekokohan performa model deteksi pada subset pengujian, dilakukan analisis **Bootstrap Resampling** dengan menyampel ulang subset pengujian secara acak dengan pengembalian (_resampling with replacement_) sebanyak $B = 1.000$ kali. Selain itu, uji hipotesis berpasangan (_paired bootstrap hypothesis test_) diterapkan untuk menguji apakah perbedaan performa antara arsitektur YOLOv8s dan YOLOv8m signifikan secara statistik.

Tabel 3.2 menyajikan rata-rata metrik, interval kepercayaan 95% (95% CI), selisih performa (M - S), serta nilai $p$ (_p-value_) empiris untuk setiap metrik evaluasi pada ambang batas kepercayaan model $\ge 0,25$.

| Metrik | YOLOv8s (95% CI) | YOLOv8m (95% CI) | Selisih (YOLOv8m - YOLOv8s) | Nilai $p$ (_p-value_) | Signifikan secara Statistik ($p < 0,05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| _Precision_ | 0,9161 (0,8485 - 0,9726) | 0,9294 (0,8674 - 0,9794) | 0,0133 (-0,0392 - 0,0716) | 0,2990 | Tidak |
| _Recall_ | 0,8932 (0,8222 - 0,9506) | 0,9411 (0,8800 - 0,9868) | 0,0479 (-0,0124 - 0,1099) | 0,1020 | Tidak |
| _F1-Score_ | 0,9041 (0,8507 - 0,9497) | 0,9348 (0,8919 - 0,9697) | 0,0307 (-0,0129 - 0,0763) | 0,0920 | Tidak |
| mAP@50 | 0,9562 (0,9074 - 0,9911) | 0,9598 (0,9221 - 0,9877) | 0,0036 (-0,0206 - 0,0285) | 0,4010 | Tidak |
| mAP@50-95 | 0,5193 (0,4704 - 0,5721) | 0,5142 (0,4605 - 0,5649) | -0,0051 (-0,0500 - 0,0427) | 0,5740 | Tidak |

Berdasarkan hasil uji bootstrap berpasangan, meskipun YOLOv8m menunjukkan estimasi rata-rata yang lebih tinggi pada metrik _Precision_ (+1,33%), _Recall_ (+4,79%), _F1-Score_ (+3,07%), dan mAP@50 (+0,36%), **tidak ada perbedaan kinerja yang signifikan secara statistik antara kedua model pada tingkat kepercayaan 95%** ($p > 0,05$ untuk semua metrik). Overlap interval kepercayaan 95% antara kedua model sangat lebar di seluruh metrik. Nilai $p$ terkecil teramati pada metrik _F1-Score_ ($p = 0,0920$) dan _Recall_ ($p = 0,1020$), namun masih berada di atas ambang batas signifikansi $\alpha = 0,05$. Hal ini menunjukkan bahwa peningkatan parameter dari 11,16 juta (YOLOv8s) menjadi 25,89 juta (YOLOv8m) tidak menghasilkan perbedaan efektivitas deteksi yang signifikan pada dataset pengujian ini.

---

## 4. Discussion

### 4.1 YOLOv8s vs. YOLOv8m — Analisis

Hasil pada Tabel 3.1 menunjukkan bahwa **YOLOv8m mencapai akurasi yang lebih tinggi** pada _dataset_ dengan strategi _oversampling_, dengan nilai mAP@50 (0,8320 vs 0,8150), _Recall_ (0,8124 vs 0,7232), mAP@50–95 (0,4387 vs 0,4210), dan _F1-Score_ (0,8178 vs 0,8008), sedangkan YOLOv8s lebih unggul dalam metrik _Precision_ (0,8971 vs 0,8234). Secara klinis, tingkat _Recall_ YOLOv8m yang lebih tinggi (+8,92 poin persentase) merupakan aspek keunggulan yang sangat krusial, mengingat setiap kasus fraktur yang tidak terdeteksi berisiko menyebabkan kesalahan diagnosis yang fatal. Pada _dataset_ tidak seimbang (_imbalanced_), perbedaan performa antara kedua model relatif kecil: YOLOv8s unggul tipis pada mAP@50 (0,5117 vs 0,4995) dan _Recall_ (0,4815 vs 0,4691), sedangkan YOLOv8m sedikit lebih unggul dalam hal _Precision_ (0,8029 vs 0,7606) dan _F1-Score_ (0,5923 vs 0,5896).

Perbandingan performa antara kondisi tidak seimbang (_imbalanced_) dan _oversampling_ mengungkap beberapa temuan penting:

- **Strategi _oversampling_ memberikan peningkatan performa yang sangat signifikan pada kedua model:** YOLOv8m mencatatkan peningkatan performa yang lebih besar, dengan kenaikan mAP@50 sebesar +66,6% (0,4995 menjadi 0,8320) dan _Recall_ sebesar +73,2% (0,4691 menjadi 0,8124), dibandingkan dengan YOLOv8s yang mencatatkan kenaikan mAP@50 sebesar +59,3% (0,5117 menjadi 0,8150) dan _Recall_ sebesar +50,2% (0,4815 menjadi 0,7232). Hasil ini mengonfirmasi efektivitas taktik _oversampling_ untuk mendeteksi fraktur pada _dataset_ medis dengan kondisi tidak seimbang.
- **Model dengan arsitektur lebih besar memperoleh manfaat lebih besar dari strategi _oversampling_:** Pada kondisi data tidak seimbang, YOLOv8s dan YOLOv8m menunjukkan kinerja yang hampir setara. Namun, setelah strategi _oversampling_ diterapkan, YOLOv8m melampaui YOLOv8s dengan selisih performa yang lebih jelas. Hal ini mengindikasikan bahwa model dengan kapasitas parameter yang lebih besar memerlukan distribusi data yang lebih seimbang untuk mengekspresikan potensi performa maksimalnya.
- **Implikasi praktis dan signifikansi statistik:** Hasil uji signifikansi statistik berpasangan menggunakan metode bootstrapping (Tabel 3.2) mengonfirmasi bahwa seluruh perbedaan performa antara YOLOv8s dan YOLOv8m tidak signifikan secara statistik ($p > 0,05$). Temuan ini memiliki implikasi praktis yang sangat krusial: YOLOv8s dapat direkomendasikan sebagai pilihan utama untuk implementasi pada perangkat klinis dengan sumber daya komputasi terbatas (seperti komputer mini medis atau proses inferensi berbasis CPU/GPU tingkat konsumen), karena mampu memberikan performa deteksi fraktur yang setara secara statistik dengan YOLOv8m tetapi dengan ukuran model 2,3 kali lebih kecil (~11,16 juta parameter berbanding ~25,89 juta parameter) dan efisiensi komputasi yang lebih tinggi. Namun, jika prioritas utama sistem penyaringan klinis adalah memaksimalkan sensitivitas deteksi secara absolut (tanpa batasan komputasi), YOLOv8m tetap dapat dipertimbangkan karena kecenderungan _Recall_ rata-rata yang lebih tinggi (+4,79% pada uji bootstrap).

### 4.2 Error Analysis: Karakteristik False Negative

Analisis terhadap kesalahan negatif palsu (_false negative_/FN) pada himpunan pengujian (_test set_) menunjukkan adanya dua pola utama penyebab kegagalan deteksi fraktur oleh model:

**1. Ukuran kotak pembatas (_bounding box_) yang sangat kecil (_small object_).** Kasus fraktur dengan ukuran kecil (seperti _hairline fracture_ atau fisura tipis) memiliki probabilitas kegagalan deteksi yang jauh lebih tinggi. Hal ini disebabkan oleh hilangnya informasi fitur spasial akibat proses penyampelan bawah (_downsampling_) yang berulang pada jaringan tulang punggung (_backbone_) model.

Untuk menganalisis pengaruh ukuran objek terhadap kegagalan deteksi secara ilmiah dan objektif, seluruh anotasi _ground truth_ (GT) pada _test set_ diklasifikasikan berdasarkan luas area kotak pembatas (_bounding box area_, dalam piksel persegi yang dinormalisasi ke resolusi referensi $1024 \times 1024$ piksel). Penelusuran kecocokan dilakukan dengan nilai IoU $\ge$ 0,5, dan penilaian dikelompokkan menggunakan dua standar: standar MS COCO dan pengelompokan kustom yang lebih granular untuk data medis.

Tabel 4.2a menyajikan distribusi deteksi berdasarkan standar klasifikasi area MS COCO:

- **Small** (luas area $\le 1024 \text{ px}^2$, setara dengan $\le 32 \times 32$ piksel)
- **Medium** ($1024 < \text{luas area} \le 9216 \text{ px}^2$, setara dengan $32 \times 32$ hingga $96 \times 96$ piksel)
- **Large** (luas area $> 9216 \text{ px}^2$, setara dengan $> 96 \times 96$ piksel)

Tabel 4.2a. Distribusi Deteksi dan False Negative berdasarkan Kategori Area MS COCO pada Test Set
| Model | Kategori Ukuran (COCO Bins) | Total GT | TP (Terdeteksi) | FN (Terlewat) | FN Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **YOLOv8s-AdamW** | Small ($\le 1024 \text{ px}^2$) | 7 | 2 | 5 | **71,43%** |
| | Medium ($1024 < \text{Area} \le 9216 \text{ px}^2$) | 216 | 183 | 33 | 15,28% |
| | Large ($> 9216 \text{ px}^2$) | 30 | 25 | 5 | 16,67% |
| **YOLOv8m-AdamW** | Small ($\le 1024 \text{ px}^2$) | 7 | 3 | 4 | **57,14%** |
| | Medium ($1024 < \text{Area} \le 9216 \text{ px}^2$) | 216 | 185 | 31 | 14,35% |
| | Large ($> 9216 \text{ px}^2$) | 30 | 27 | 3 | 10,00% |

Data pada Tabel 4.2a menunjukkan bahwa tingkat kegagalan deteksi (FN Rate) melonjak sangat tajam pada objek berskala **Small** (mencapai **71,43%** pada YOLOv8s dan **57,14%** pada YOLOv8m). Namun, karena karakteristik sebaran ukuran fraktur tangan pada FracAtlas mayoritas berkumpul di kategori menengah, ukuran sampel pada kelas _Small_ (7 box) dan _Large_ (30 box) relatif kecil.

Untuk analisis yang lebih merata secara statistik, dilakukan pengelompokan kustom yang membagi rentang luas area secara lebih granular menjadi empat kelas:

- **Sangat Kecil** (luas area $\le 2000 \text{ px}^2$)
- **Kecil** ($2000 < \text{luas area} \le 4000 \text{ px}^2$)
- **Sedang** ($4000 < \text{luas area} \le 8000 \text{ px}^2$)
- **Besar** (luas area $> 8000 \text{ px}^2$)

Tabel 4.2b. Distribusi Deteksi dan False Negative berdasarkan Kategori Area Granular pada Test Set
| Model | Kategori Ukuran (Custom Bins) | Total GT | TP (Terdeteksi) | FN (Terlewat) | FN Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **YOLOv8s-AdamW** | Sangat Kecil ($\le 2000 \text{ px}^2$) | 46 | 29 | 17 | **36,96%** |
| | Kecil ($2000 - 4000 \text{ px}^2$) | 93 | 81 | 12 | 12,90% |
| | Sedang ($4000 - 8000 \text{ px}^2$) | 74 | 66 | 8 | 10,81% |
| | Besar ($> 8000 \text{ px}^2$) | 40 | 34 | 6 | 15,00% |
| **YOLOv8m-AdamW** | Sangat Kecil ($\le 2000 \text{ px}^2$) | 46 | 33 | 13 | **28,26%** |
| | Kecil ($2000 - 4000 \text{ px}^2$) | 93 | 79 | 14 | 15,05% |
| | Sedang ($4000 - 8000 \text{ px}^2$) | 74 | 67 | 7 | 9,46% |
| | Besar ($> 8000 \text{ px}^2$) | 40 | 36 | 4 | 10,00% |

Hasil pada Tabel 4.2b memperkuat klaim bahwa fraktur kecil lebih sering terlewat. Objek kategori **Sangat Kecil** mengalami tingkat kegagalan deteksi (FN Rate) sebesar **36,96%** pada YOLOv8s dan **28,26%** pada YOLOv8m. Angka ini berkali-kali lipat lebih tinggi dibandingkan kelas Kecil (12,90% - 15,05%) maupun Sedang (9,46% - 10,81%). YOLOv8m secara konsisten mampu menekan tingkat kegagalan deteksi pada fraktur sangat kecil menjadi 28,26% (dibandingkan YOLOv8s sebesar 36,96%) berkat representasi fitur yang lebih mendalam pada varian model _Medium_.

**2. Tumpang tindih (_overlap_) struktur anatomi tulang yang kompleks.** Pada citra proyeksi lateral atau _oblique_, area fraktur sering kali tertutup oleh bayangan struktur tulang lainnya (misalnya fraktur pada area proksimal falang yang tumpang tindih dengan metakarpal). Akibatnya, detektor mengalami kesulitan untuk membedakan antara tepi garis fraktur dengan batas celah sendi normal (_articular space_).

> **Implikasi klinis:** Kasus FN pada fraktur _non-displaced_ dan _hairline_ merupakan risiko klinis paling serius karena secara visual sangat sulit dibedakan dari struktur tulang normal, bahkan oleh ahli radiologi sekalipun. Integrasi pemrosesan akhir (_post-processing_) berbasis konteks bentuk (_shape context_) atau penerapan mekanisme deteksi multi-skala khusus untuk wilayah minat (_Region of Interest_/ROI) kecil berpotensi menekan tingkat FN ini pada penelitian selanjutnya.

### 4.3 Justifikasi Pipeline Preprocessing Medis

Kombinasi antara pemfilteran khusus tangan (_hand-only filtering_), strategi _oversampling_, dan penerapan augmentasi konservatif merupakan pendekatan yang selaras dengan praktik terbaik (_best practices_) dalam analisis citra medis:

- Fokus pada satu wilayah anatomi tertentu secara efektif mereduksi variasi fitur latar belakang yang tidak relevan [3].
- Penerapan _oversampling_ pada _dataset_ medis yang tidak seimbang telah terbukti secara konsisten meningkatkan sensitivitas deteksi model [1, 4].
- Penggunaan augmentasi konservatif yang patuh pada batasan fisik anatomis mencegah model dalam mempelajari representasi visual yang tidak valid secara medis (misalnya menghindari distorsi bentuk atau orientasi yang tidak alami) [5].

### 4.4 Comparison with Prior Detection Literature

**Xiong et al. (2026).** _A Benchmark X-ray Dataset for Pediatric Supracondylar Humerus Fractures with Improved YOLOv11-Based Detection_

- **Metode:** YOLOv11-LA (_Local-Global Attention_).
- **Penanganan Imbalance:** Hanya menggunakan augmentasi data (rotasi, _flipping_, kontras, kecerahan).
- **Dataset:** PediaSHF-DX (_Pediatric Supracondylar Humerus Fractures Dataset_).
- **Hasil:**
    - Faster R-CNN: Precision 0,921, Recall 0,894, F1-Score\* ≈0,907, mAP@50 0,267, mAP@50–95 0,274
    - DETR: Precision 0,894, Recall 0,142, F1-Score\* ≈0,248, mAP@50 0,251, mAP@50–95 0,128
    - YOLOv11 (Baseline): Precision 0,947, Recall 0,289, F1-Score\* ≈0,445, mAP@50 0,304, mAP@50–95 0,157
    - YOLOv11-LA (Ours): Precision 0,963, Recall 0,304, F1-Score\* ≈0,465, mAP@50 0,308, mAP@50–95 0,169

**Priyanka et al. (2025).** _FractureNet: A Fusion Approach for Automated Hand Fracture Detection and Classification_

- **Metode:** Model fusi menggabungkan fitur dari SVM (HOG + LBP + VGG16) dan YOLOv5 untuk deteksi ROI fraktur.
- **Dataset:** 4.000+ citra sinar-X tangan, tidak seimbang (_imbalanced_).
- **Penanganan Imbalance:** Hanya menggunakan augmentasi data (rotasi, _flipping_, kontras, penyesuaian kecerahan).
- **Hasil:**
    - SVM: Accuracy 95,4%, Precision 95,3%, Recall 96,1%, F1-Score 95,6%
    - YOLOv5: Precision 0,95, Recall 0,96, mAP@50 0,94, mAP@50–95 0,85
- **Konteks:** Penelitian tersebut merupakan studi klasifikasi (fraktur vs. non-fraktur) dengan YOLOv5 yang hanya digunakan untuk deteksi ROI, bukan deteksi fraktur secara menyeluruh (_end-to-end_) seperti pada penelitian ini.

**Fang et al. (2024).** _Faster R-CNN model for target recognition and diagnosis of scapular fractures_

- **Metode:** Faster R-CNN.
- **Dataset:** Citra skapula dengan fraktur dan tanpa fraktur, secara alami tidak seimbang (_naturally imbalanced_).
- **Penanganan Imbalance:** Menggunakan augmentasi data.
- **Hasil:** Precision 97,9%, Recall 95,78%, AUC ≈ 0,97.
- **Konteks:** Hanya melakukan evaluasi klasifikasi (AUC) dan tidak menyertakan mAP. Selain itu, objek penelitian bukan fraktur tulang tangan (_hand bone fracture_), melainkan fraktur skapula (_scapular fracture_).

**Meza et al. (2025).** _Deep Learning Approach for Arm Fracture Detection Based on an Improved YOLOv8 Algorithm_

- **Metode:** YOLOv8 _Hybrid Attention_ (HA).
- **Penanganan Imbalance:** Tidak dijelaskan.
- **Dataset:** FracAtlas (lengan dan kaki).
- **Hasil:** mAP@50 0,713, Precision 0,713, Recall 0,642, F1-Score 0,676.
- **Konteks:** Deteksi fraktur dilakukan pada lengan dan kaki, bukan tangan. Himpunan data (_dataset_) dan alur pemrosesan awal (_preprocessing pipeline_) yang digunakan juga berbeda.

**Penelitian ini (Ours):**

- **Metode:** YOLOv8s (_small_) dan YOLOv8m (_medium_).
- **Dataset:** FracAtlas (_hand_), _oversampled_.
- **Penanganan Imbalance:** Pemfilteran khusus tangan (_hand-only filtering_), _oversampling_, dan augmentasi konservatif.
- **Hasil:**
    - YOLOv8s: mAP@50 0,8150, Precision 0,8971, Recall 0,7232, F1-Score 0,8008.
    - YOLOv8m: mAP@50 0,8320, Precision 0,8234, Recall 0,8124, F1-Score 0,8178.

### 4.5 Strengths and Limitations

**Kelebihan:**

- Fokus spesifik pada anatomi tangan, sehingga berhasil mengeliminasi fitur-fitur pengganggu (_confounding features_) dari bagian tubuh yang lain.
- Alur pemrosesan awal (_preprocessing pipeline_) terdokumentasi dengan baik dan dapat direproduksi (_reproducible_).
- Evaluasi dilakukan secara multidimensi yang mencakup akurasi, kecepatan, dan matriks kekacauan (_confusion matrix_).
- Menggunakan _dataset_ publik (FracAtlas) sehingga hasil penelitian dapat diverifikasi dan dibandingkan secara terbuka oleh peneliti lain.

**Keterbatasan:**

- Perbandingan model hanya terbatas pada varian YOLOv8. Arsitektur deteksi lain seperti Faster R-CNN, RetinaNet, atau YOLO versi lainnya (misalnya YOLOv5, YOLOv11) tidak disertakan, sehingga generalisasi mengenai keunggulan model ini terhadap arsitektur lain belum dapat disimpulkan secara menyeluruh.
- Meskipun uji signifikansi statistik dan interval kepercayaan (95% CI) telah dilakukan menggunakan metode bootstrapping pada subset pengujian, variabilitas performa antar-pelatihan (_cross-run variance_) yang disebabkan oleh inisialisasi bobot acak (_random seeds_) yang berbeda belum dianalisis secara mendalam karena keterbatasan waktu dan kapasitas komputasi untuk melatih ulang arsitektur model berkali-kali.
- Model belum diuji pada citra yang memiliki implan ortopedi (_hardware_ logam seperti pelat atau sekrup).
- _Dataset_ FracAtlas bersumber dari institusi tunggal, sehingga tingkat generalisasi model terhadap populasi pasien dan variasi peralatan sinar-X yang berbeda masih perlu divalidasi lebih lanjut.
- Anotasi pada _dataset_ terbatas pada deteksi biner (fraktur vs. non-fraktur) tanpa adanya informasi mengenai **subtipe fraktur** (_transverse_, _oblique_, _comminuted_, dll.), **lokasi anatomis yang spesifik** (tulang bagian mana yang mengalami fraktur), serta **tingkat keparahan** (_severity_ seperti _displaced_/_non-displaced_). Model tidak mampu membedakan tingkat keparahan atau jenis fraktur tersebut—informasi yang sebenarnya sangat dibutuhkan dalam perencanaan tindakan klinis.
- **Pemisahan tingkat pasien (patient-level split) tidak dapat dilakukan.** Karena dataset FracAtlas sepenuhnya teranonimisasi dan tidak menyediakan ID pasien (patient ID) pada metadata citra, pembagian subset pelatihan, validasi, dan pengujian dilakukan secara acak tingkat citra (image-level). Akibatnya, keterbatasan ini menimbulkan risiko kebocoran data (data leakage), di mana citra sinar-X yang berasal dari pasien yang sama berpotensi tersebar di split yang berbeda (misalnya pelatihan dan pengujian), sehingga dapat menyebabkan evaluasi performa model yang terlalu optimis. Keterbatasan ini tidak dapat sepenuhnya dikesampingkan dalam penelitian ini.

---

## 5. Conclusion

Penelitian ini menyajikan evaluasi sistematis dan komprehensif terhadap dua varian arsitektur YOLOv8—yaitu YOLOv8s (_Small_) dan YOLOv8m (_Medium_)—untuk mendeteksi fraktur tulang tangan pada citra sinar-X dengan menggunakan _dataset_ FracAtlas. Melalui penerapan alur pemrosesan awal (_preprocessing pipeline_) yang dirancang khusus untuk domain medis, termasuk teknik _oversampling_ untuk mengatasi masalah ketidakseimbangan kelas yang melekat pada _dataset_, kedua model tersebut dilatih dan dievaluasi secara komparatif.

Temuan utama dari penelitian ini menunjukkan bahwa **kedua arsitektur model mendapatkan manfaat yang sangat signifikan dari penerapan strategi _oversampling_**. Model YOLOv8s mencapai mAP@50 sebesar 0,8150 (mengalami peningkatan +59,3% dari _baseline_ tidak seimbang sebesar 0,5117), sedangkan YOLOv8m mencapai mAP@50 sebesar 0,8320 (meningkat +66,6% dari _baseline_ sebesar 0,4995). Secara absolut, YOLOv8m menghasilkan akurasi tertinggi (mAP@50 sebesar 0,8320; _Recall_ sebesar 0,8124; dan _F1-Score_ sebesar 0,8178) dengan keunggulan metrik _Recall_ sebesar +8,92 poin persentase dibandingkan dengan YOLOv8s (0,7232)—sebuah keunggulan yang sangat krusial dalam domain klinis. Di sisi lain, YOLOv8s menawarkan keseimbangan terbaik antara efisiensi komputasi dan akurasi karena hanya menggunakan 11,16 juta parameter (2,3 kali lebih ringan dibandingkan YOLOv8m) dengan selisih nilai mAP@50 yang sangat kecil, yaitu hanya 0,017.

Secara keseluruhan, penelitian ini menyediakan nilai acuan yang dapat direproduksi (_reproducible baseline_) untuk menjadi tolok ukur bagi penelitian deteksi fraktur tangan di masa mendatang, baik untuk pengembangan menggunakan arsitektur yang lebih baru (seperti YOLOv9 atau YOLOv11) maupun untuk integrasi ke dalam sistem klinis seperti PACS (_Picture Archiving and Communication System_).

---

## Daftar Pustaka

[1] Albright, J. A., Rebello, E., Kosinski, L. R., Patel, D. D., Spears, J. R., Gil, J. A., & Katarincic, J. A. (2022). Characterization of the Epidemiology and Risk Factors for Hand Fractures Presenting to United States Emergency Departments. Hand, 17(4).

[2] "A survey on imbalanced learning: latest research, applications and future directions" (2024, Springer)

[3] "A comprehensive survey on imbalanced data learning" (2026, Frontiers of Computer Science / Springer)

[3] "Bone Fracture Detection And Localisation Using Enhanced YOLOv8 Model" (2025, The Indian Journal of Radiology and Imaging / ResearchGate)

[4] "Examples of fracture detection on X-ray images from the FracAtlas dataset" (Oktober 2024, IEEE / ResearchGate)

[5] "Detection of whole body bone fractures based on improved YOLOv7 with attention mechanism" (2024, Measurement / ScienceDirect)

[6] "FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation" (2023, Scientific Data / Nature PMC)

[5] V. Ponkilainen, I. Kuitunen, R. Liukkonen, M. Vaajala, A. Reito, and M. Uimonen, “The incidence of musculoskeletal injuries: a systematic review and meta-analysis,” Bone Jt. Res., vol. 11, no. 11, pp. 814–825, Nov. 2022, doi: 10.1302/2046-3758.1111.BJR-2022-0181.R1.

[6] “Musculoskeletal health.” Accessed: May 19, 2026. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/musculoskeletal-conditions

[7] L. Tanzi, A. Audisio, G. Cirrincione, A. Aprato, and E. Vezzetti, “Vision Transformer for femur fracture classification,” 2021, arXiv. doi: 10.48550/ARXIV.2108.03414.

[8] I. Abedeen, Md. A. Rahman, F. Z. Prottyasha, T. Ahmed, T. M. Chowdhury, and S. Shatabda, “FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs,” Sci. Data, vol. 10, no. 1, p. 521, Aug. 2023, doi: 10.1038/s41597-023-02432-4.

[9] I. Araf, A. Idri, and I. Chairi, “Cost-sensitive learning for imbalanced medical data: a review,” Artif. Intell. Rev., vol. 57, no. 4, p. 80, Mar. 2024, doi: 10.1007/s10462-023-10652-8.

[10] J. F. Cohen and M. D. F. McInnes, “Deep Learning Algorithms to Detect Fractures: Systematic Review Shows Promising Results but Many Limitations,” Radiology, vol. 304, no. 1, pp. 63–64, Jul. 2022, doi: 10.1148/radiol.212966.

[11] R. Aggarwal et al., “Diagnostic accuracy of deep learning in medical imaging: a systematic review and meta-analysis,” Npj Digit. Med., vol. 4, no. 1, p. 65, Apr. 2021, doi: 10.1038/s41746-021-00438-z.

[12] T. Diwan, G. Anirudh, and J. V. Tembhurne, “Object detection using YOLO: challenges, architectural successors, datasets and applications,” Multimed. Tools Appl., vol. 82, no. 6, pp. 9243–9275, Mar. 2023, doi: 10.1007/s11042-022-13644-y.

[13] S. M. Alkentar, B. Alsahwa, A. Assalem, and D. Karakolla, “Practical comparation of the accuracy and speed of YOLO, SSD and Faster RCNN for drone detection,” J. Eng., vol. 27, no. 8, pp. 19–31, Aug. 2021, doi: 10.31026/j.eng.2021.08.02.

[14] S. Rajaraman, P. Ganesan, and S. K. Antani, “Does deep learning model calibration improve performance in class-imbalanced medical image classification?”.

[15] P. Chen, S. Liu, W. Lu, F. Lu, and B. Ding, “WCAY object detection of fractures for X-ray images of multiple sites,” Sci. Rep., vol. 14, no. 1, p. 26702, Nov. 2024, doi: 10.1038/s41598-024-77878-6.

[16] S. Du and Y. Wei, “ASC-YOLO: Multi-Scale Feature Fusion and Adaptive Decoupled Head for Fracture Detection in Medical Imaging,” Appl. Sci., vol. 15, no. 16, p. 9031, Aug. 2025, doi: 10.3390/app15169031.

[17] T. Zhou, K. Chen, H. Lu, S. Yu, W. Chai, and Q. Liu, “PSCA-YOLO: An enhanced YOLO with position-semantics coupled attention for mandibular fracture detection,” iScience, vol. 29, no. 4, p. 115404, Apr. 2026, doi: 10.1016/j.isci.2026.115404.

[18] R.-Y. Ju and W. Cai, “Fracture detection in pediatric wrist trauma X-ray images using YOLOv8 algorithm,” Sci. Rep., vol. 13, no. 1, p. 20077, Nov. 2023, doi: 10.1038/s41598-023-47460-7.

[19] G. Meza, D. Ganta, and S. Gonzalez Torres, “Deep Learning Approach for Arm Fracture Detection Based on an Improved YOLOv8 Algorithm,” Algorithms, vol. 17, no. 11, p. 471, Oct. 2024, doi: 10.3390/a17110471.

[20] J. Terven, D.-M. Córdova-Esparza, and J.-A. Romero-González, “A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS,” Mach. Learn. Knowl. Extr., vol. 5, no. 4, pp. 1680–1716, Nov. 2023, doi: 10.3390/make5040083.

[21] L. Alzubaidi et al., “Review of deep learning: concepts, CNN architectures, challenges, applications, future directions,” J. Big Data, vol. 8, no. 1, p. 53, Mar. 2021, doi: 10.1186/s40537-021-00444-8.

[22] A. Badithela, T. Wongpiromsarn, and R. M. Murray, “Evaluation Metrics for Object Detection for Autonomous Systems,” Oct. 19, 2022, arXiv: arXiv:2210.10298. doi: 10.48550/arXiv.2210.10298.

[23] G. Jocher, A. Chaurasia, and J. Qiu, YOLO by Ultralytics: State-of-the-Art YOLO Models. (2023). [Online]. Available: https://github.com/ultralytics/ultralytics

[24] C. Shorten and T. M. Khoshgoftaar, “A survey on Image Data Augmentation for Deep Learning,” J. Big Data, vol. 6, no. 1, p. 60, Dec. 2019, doi: 10.1186/s40537-019-0197-0.

[25] A. Azizi, M. Azizi, and M. Nasri, “Artificial Intelligence Techniques in Medical Imaging: A Systematic Review,” Int. J. Online Biomed. Eng. IJOE, vol. 19, no. 17, pp. 66–97, Dec. 2023, doi: 10.3991/ijoe.v19i17.42431.

[26] M. A. Mazurowski, M. Buda, A. Saha, and M. R. Bashir, “Deep learning in radiology: An overview of the concepts and a survey of the state of the art with focus on MRI,” J. Magn. Reson. Imaging, vol. 49, no. 4, pp. 939–954, Apr. 2019, doi: 10.1002/jmri.26534.

[27] C.-T. Chien, R.-Y. Ju, K.-Y. Chou, E. Xieerke, and J.-S. Chiang, “YOLOv8-AM: YOLOv8 Based on Effective Attention Mechanisms for Pediatric Wrist Fracture Detection,” IEEE Access, vol. 13, pp. 52461–52477, 2025, doi: 10.1109/ACCESS.2025.3549839.

[28] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” Jan. 04, 2019, arXiv: arXiv:1711.05101. doi: 10.48550/arXiv.1711.05101.

[29] H. Li et al., “MSPO: A machine learning hyperparameter optimization method for enhanced breast cancer image classification,” Digit. Health, vol. 11, p. 20552076251361603, May 2025, doi: 10.1177/20552076251361603.

[30] R. Y. L. Kuo et al., “Artificial Intelligence in Fracture Detection: A Systematic Review and Meta-Analysis,” Radiology, vol. 304, no. 1, pp. 50–62, Jul. 2022, doi: 10.1148/radiol.211785.

[31] A. Nouri, B. M. Merzah, S. Mosayyebpour, R. Mousa, and S. Hesaraki, “Evaluation Metrics in Learning Systems: A survey,” Aug. 2025.

[32] A. Aldubaikhi and S. Patel, “Advancements in Small-Object Detection (2023–2025): Approaches, Datasets, Benchmarks, Applications, and Practical Guidance,” Appl. Sci., vol. 15, no. 22, p. 11882, Nov. 2025, doi: 10.3390/app152211882.

[33] X. Yang et al., “Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence,” Apr. 2022.
