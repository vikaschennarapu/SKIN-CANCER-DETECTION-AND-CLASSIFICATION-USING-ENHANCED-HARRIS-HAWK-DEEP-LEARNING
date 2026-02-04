# Skin Cancer Detection and Classification using Enhanced Harris Hawk Deep Learning
## 🩺 Overview
Early detection of skin cancer is critical for improving survival rates and reducing healthcare costs. This project implements an automated AI-powered diagnostic system that detects and classifies eight different types of skin lesions from dermoscopic images.

The core innovation lies in the use of Enhanced Harris Hawk Optimization (EHHO) to fine-tune the hyperparameters of a Convolutional Neural Network (CNN), resulting in significantly higher accuracy and faster convergence compared to standard deep learning models.

## 🚀 Key Features
Hybrid Architecture: Combines CNN for automated feature extraction with EHHO for optimal parameter tuning (learning rate, dropout, etc.).

High Performance: Achieves 96.2% accuracy, a 0.96 F1-score, and a 0.981 ROC-AUC.

Interpretability: Uses Grad-CAM heatmaps to provide clinical interpretability, allowing doctors to see which areas of the lesion the AI is focusing on.

Efficiency: Reduces analysis time from the typical 30 minutes (manual) to under 1 minute.

Streamlit Web App: A user-friendly interface for real-time image upload and classification.

## 📊 Dataset
The model was trained on 25,331 images from the ISIC 2019 dataset. It classifies the following categories:

Melanoma

Melanocytic Nevus

Basal Cell Carcinoma

Actinic Keratosis

Benign Keratosis-like Lesion

Dermatofibroma

Vascular Lesion

Squamous Cell Carcinoma

## 🛠️ Tech Stack
Language: Python 3.x

Deep Learning: TensorFlow / Keras

Optimization: Enhanced Harris Hawk Optimization (EHHO)

Web Framework: Streamlit

Data Analysis: Pandas, NumPy, Scikit-learn

Visualization: Matplotlib, Seaborn, OpenCV

## 📉 Methodology
Preprocessing: Includes image resizing (28x28), normalization, and data augmentation to handle class imbalance.

Feature Extraction: A deep CNN architecture automatically extracts over 1,000 high-dimensional feature vectors.

EHHO Optimization: A nature-inspired metaheuristic algorithm based on the cooperative hunting behavior of Harris hawks is used to find the global optimum for the model's hyperparameters.

Classification: The optimized model predicts the lesion type with a confidence score.
