# Pneumonia Detection using Modified ResNet

Capstone project from the **Summer School on Deep Learning** at **IIITDM Jabalpur**, ranked **12th out of 600+ participants**.

## 🔍 Overview
Developed a deep learning model using a modified **ResNet-18** architecture to classify chest X-ray images as **Normal** or **Pneumonia**.

## 🛠️ Tech Stack
- Python, PyTorch, Google Colab
- Data Augmentation, Transfer Learning
- Evaluation: Accuracy, Confusion Matrix, Loss Curves

## 🧠 Model Details
- Custom classifier head:  
  `Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→2)`
- Optimizer: Adam | Loss: CrossEntropy | Scheduler: ReduceLROnPlateau

## 📊 Results
- **Baseline ResNet:** ~85–90% accuracy  
- **Modified ResNet:** ~92–94% accuracy  
- Reduced overfitting and improved generalization

## 📂 Dataset
[Chest X-ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## © Copyright
© 2025 Sohang Debnath. This project is for educational and non-commercial use only. Dataset © by original Kaggle authors.

