# Skin Cancer Detection and Classification using Enhanced Harris Hawk Deep Learning Technique
## 🩺 Overview
Early detection of skin cancer is critical for improving survival rates and reducing healthcare costs. This project implements an automated AI-powered diagnostic system that detects and classifies eight different types of skin lesions from dermoscopic images.

The core innovation lies in the use of Enhanced Harris Hawk Optimization (EHHO) to fine-tune the hyperparameters of a Convolutional Neural Network (CNN), resulting in significantly higher accuracy and faster convergence compared to standard deep learning models.

## 🚀 Key Features
Hybrid Architecture: Combines CNN for automated feature extraction with EHHO for optimal parameter tuning (learning rate, dropout, etc.).

High Performance: Achieves 96.2% accuracy, a 0.96 F1-score, and a 0.981 ROC-AUC.

Interpretability: Uses Grad-CAM heatmaps to provide clinical interpretability, allowing doctors to see which areas of the lesion the AI is focusing on.

Efficiency: Reduces analysis time from the typical 30 minutes (manual) to under 1 minute.

Streamlit Web App: A user-friendly interface for real-time image upload and classification.

## Project Presentation

- <a href="https://github.com/vikaschennarapu/SKIN-CANCER-DETECTION-AND-CLASSIFICATION-USING-ENHANCED-HARRIS-HAWK-DEEP-LEARNING/blob/main/Presentation.pptx">Project Presentation PPT</a>

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

- <a href="https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification">Dataset</a>

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

## Project Python Code
- <a href="https://github.com/vikaschennarapu/SKIN-CANCER-DETECTION-AND-CLASSIFICATION-USING-ENHANCED-HARRIS-HAWK-DEEP-LEARNING/blob/main/skin.ipynb">Jupyter Notes</a>
- <a href="https://github.com/vikaschennarapu/SKIN-CANCER-DETECTION-AND-CLASSIFICATION-USING-ENHANCED-HARRIS-HAWK-DEEP-LEARNING/blob/main/run.py">Jupyter Notes</a>

## 📈 Results
The EHHO-optimized model outperformed standard models by approximately 5%, effectively reducing diagnostic errors from the industry standard of 15–30% to under 4%.

## 🔮 Future Enhancements
Integration with mobile and IoT devices for real-time diagnosis.

Extension of HHO to optimize the entire network architecture (AutoML).

Implementation of Explainable AI (XAI) for higher transparency in clinical settings.

## 👥 Contributors
P. Swetha

C. Vikas

B. Sai Deekshitha

Aatish Kumar

Guidance: Ms. K G Mohanavalli (Assistant Professor, Dept. of CSE)
