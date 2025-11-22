# Face Recognition Algorithms — PCA, LDA, HOG Comparison
Author: Merve Üstünel
University: Kocaeli University — Software Engineering
Academic Year: 2024–2025

This project implements and compares three classical face recognition algorithms:
PCA (Eigenfaces), LDA (Fisherfaces), and HOG (Histogram of Oriented Gradients).
The study includes real-time face detection, feature extraction, histogram comparison,
and PCA/LDA/HOG training pipelines using Python & OpenCV.

============================================================
🇬🇧 1. PROJECT SUMMARY (ENGLISH)
============================================================

This repository contains a full implementation of classical face recognition methods.
The aim is to analyze PCA, LDA, and HOG based on:

- Accuracy
- Robustness to lighting and expression changes
- Feature extraction quality
- Real-time performance

Included Modules:
- Real-time face detection (Haar Cascade)
- Histogram-based recognition
- PCA model training
- LDA model training
- HOG feature extraction
- Confidence score calculation
- Real-time identification

============================================================
🇬🇧 2. TECHNOLOGIES USED
============================================================

Libraries:
- OpenCV
- NumPy
- Pillow (PIL)

Algorithms:
- PCA
- LDA
- HOG
- Histogram Correlation
- Haar Cascade Detector

Tools:
- Python
- Webcam
- Local Dataset (not uploaded)
- trainer/ directory for model files

============================================================
🇬🇧 3. PROJECT STRUCTURE
============================================================

project/
│
├── data/                       # Face images (not uploaded)
├── trainer/
│   ├── trainer.npy
│   ├── pca_model.npy
│   ├── lda_model.yml
│   ├── hog_features.npy
│
├── detect_faces.py
├── train_histogram.py
├── train_pca.py
├── train_lda.py
├── train_hog.py
├── recognize.py
└── README.md

============================================================
🇬🇧 4. WORKFLOW OVERVIEW
============================================================

Camera Input
   ↓
Haar Cascade Face Detection
   ↓
Feature Extraction (Histogram / PCA / LDA / HOG)
   ↓
Model Training
   ↓
Similarity Comparison
   ↓
Predicted Person + Confidence Score

============================================================
🇬🇧 5. METHODS (SHORT VERSION)
============================================================

PCA (Eigenfaces):
- Dimensionality reduction
- Fast, lightweight
- Sensitive to light and pose changes

LDA (Fisherfaces):
- Maximizes class separation
- Good on structured datasets
- Weak when variation is high

HOG:
- Extracts gradient/structural features
- Most robust method
- Highest accuracy in experiments

============================================================
🇬🇧 6. RESULTS
============================================================

Algorithm | Accuracy | Notes
--------- | -------- | -----
HOG       | Highest  | Best robustness
PCA       | Medium   | Fast but unstable
LDA       | Lower    | Works only when classes are separable

Conclusion:
HOG achieved the highest accuracy and consistency.
PCA and LDA performed moderately with limitations.

============================================================
🇬🇧 7. RUNNING THE PROJECT
============================================================

# Install dependencies
pip install opencv-python numpy pillow

# Train models
python train_histogram.py
python train_pca.py
python train_lda.py
python train_hog.py

# Run real-time recognition
python recognize.py

============================================================
🇹🇷 TÜRKÇE BÖLÜM — YÜZ TANIMA RAPOR ÖZETİ
============================================================

Bu proje, PCA, LDA ve HOG gibi klasik yüz tanıma algoritmalarını incelemekte,
karşılaştırmakta ve gerçek zamanlı olarak test etmektedir.

Amaç:
- Yüz algılama
- Özellik çıkarımı (Histogram, PCA, LDA, HOG)
- Model eğitimi
- Gerçek zamanlı tanıma
- Algoritma karşılaştırması

============================================================
🇹🇷 KULLANILAN TEKNOLOJİLER
============================================================

Kütüphaneler:
- OpenCV
- NumPy
- Pillow

Algoritmalar:
- Haar Cascade
- Histogram karşılaştırma
- PCA (Eigenfaces)
- LDA (Fisherfaces)
- HOG

Araçlar:
- Python
- Kamera
- Yerel veri seti
- trainer/ dizini

============================================================
🇹🇷 YÖNTEM ÖZETİ
============================================================

Haar Cascade:
Gerçek zamanlı yüz algılama.

Histogram Tanıma:
Gri tonlama histogramı çıkarılıp normalize edildi, korelasyon ile karşılaştırıldı.

PCA:
Boyut indirgeme yöntemi ile yüz özellikleri çıkarıldı.

LDA:
Sınıflar arası ayrımı maksimize ederek tanıma yapıldı.

HOG:
Yüzün kenar/yönelim özelliklerini çıkararak en yüksek doğruluğu sağladı.

============================================================
🇹🇷 SONUÇLAR
============================================================

Algoritma | Performans
--------- | ----------
HOG       | ⭐ En yüksek doğruluk
PCA       | Orta düzey
LDA       | Düşük doğruluk

Genel Yorum:
HOG yöntemi farklı ışık, açı ve ifadelerde en yüksek başarıyı göstermiştir.

============================================================
🇹🇷 KAYNAKLAR
============================================================

Turk & Pentland — PCA  
Viola & Jones — Haar Cascade  
Dalal & Triggs — HOG  
Ahonen et al. — LBP  
Krizhevsky et al. — CNN  
OpenCV Documentation

============================================================
END OF README
============================================================
