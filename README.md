# Pneumonia Detection using Modified ResNet

Capstone project from the Summer School on Deep Learning at IIITDM Jabalpur, ranked **12th out of 600+ participants**.

---

# 🔍 Overview

This project focuses on detecting **Pneumonia from Chest X-ray images** using **Deep Learning**.

A **transfer learning approach** is used where a pretrained **ResNet-18** model is fine-tuned to classify X-ray images into two categories:

- **Normal**
- **Pneumonia**

The objective is to improve performance by modifying the **classifier head of ResNet-18** and applying **data augmentation techniques** to improve generalization and reduce overfitting.

---

# 📂 Dataset

Dataset used: **Chest X-Ray Images (Pneumonia) – Kaggle**

The dataset contains labeled chest X-ray images belonging to two classes:

- **Normal**
- **Pneumonia**

Dataset structure:

```
dataset/
   train/
      normal/
      pneumonia/

   test/
      normal/
      pneumonia/

   val/
      normal/
      pneumonia/
```

All images are resized to **224×224 pixels** to match the input requirement of **ResNet-18**.

📷 **Dataset Samples**

[INSERT SAMPLE DATASET IMAGES HERE]

---

# 🛠️ Tech Stack

- Python  
- PyTorch  
- Google Colab  

Techniques used:

- Transfer Learning  
- Data Augmentation  
- Learning Rate Scheduling  

Evaluation metrics:

- Accuracy  
- Confusion Matrix  
- Loss Curves  

---

# 🧪 Data Augmentation

Medical image datasets are usually limited in size. To improve the robustness of the model and reduce overfitting, **data augmentation techniques** were applied during training.

The following transformations were used:

- **Random Horizontal Flip** – introduces mirror variations of images  
- **Random Rotation (small angle)** – helps the model handle orientation variations  
- **Resize to 224×224** – required for ResNet input size  
- **Normalization** – scales pixel values for stable training  

Example preprocessing pipeline:

```python
transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

These augmentations help the model **learn generalized features** instead of memorizing specific training images.

---

# 🧠 Model Architecture

## Baseline Model: ResNet-18

The baseline model uses a pretrained **ResNet-18** architecture.

ResNet introduces the concept of **Residual Learning**, where shortcut connections allow the model to learn residual mappings:

```
F(x) + x
```

This helps solve the **vanishing gradient problem** in deep neural networks and allows training of deeper architectures.

In the baseline model:

- Pretrained ResNet-18 is loaded
- The final fully connected layer is replaced with a classifier for **2 classes**

---

## Modified ResNet Model

To improve performance, the classifier head of the ResNet-18 model was modified.

The original fully connected layer was replaced with a deeper classifier:

```
Linear(512 → 256)
ReLU
Dropout(0.3)
Linear(256 → 2)
```

Purpose of modifications:

- **Extra hidden layer** → learns more complex patterns  
- **ReLU activation** → introduces non-linearity  
- **Dropout (0.3)** → reduces overfitting by randomly dropping neurons during training  

This modified architecture improves **generalization on unseen chest X-ray images**.

---

# ⚙️ Training Setup

Training configuration:

- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Learning Rate Scheduler:** ReduceLROnPlateau  

The scheduler reduces the learning rate when validation performance stops improving.

---

# 🔁 Training Loop

The training loop follows the standard deep learning workflow:

### Step 1 — Forward Pass
Input images are passed through the model to generate predictions.

### Step 2 — Loss Calculation
The predicted outputs are compared with the true labels using **CrossEntropy Loss**.

### Step 3 — Backpropagation
Gradients are computed using backpropagation.

### Step 4 — Parameter Update
The **Adam optimizer** updates model weights.

### Step 5 — Validation
The model is evaluated on the validation dataset after each epoch.

The training process tracks:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

📊 **Training Accuracy Graph**

[INSERT TRAINING ACCURACY GRAPH HERE]

📉 **Training Loss Curve**

[INSERT LOSS GRAPH HERE]

---

# 📊 Evaluation

The model performance was evaluated using a **confusion matrix**, which shows the number of correct and incorrect predictions.

Metrics observed:

- True Positives
- True Negatives
- False Positives
- False Negatives

📊 **Confusion Matrix**

[INSERT CONFUSION MATRIX HERE]

---

# 📈 Results

| Model | Accuracy |
|------|------|
| Baseline ResNet-18 | ~85–90% |
| Modified ResNet-18 | ~92–94% |

Improvements achieved:

- Better feature learning
- Reduced overfitting
- Improved generalization on test data

---

# 🚀 Future Improvements

Possible improvements include:

- Using deeper architectures such as **ResNet-50**
- Applying **Grad-CAM** for model interpretability
- Hyperparameter tuning
- Training on larger medical datasets

---

# © Copyright

© 2025 Sohang Debnath  
This project is for **educational and non-commercial use only**.

Dataset copyright belongs to the **original Kaggle dataset authors**.
