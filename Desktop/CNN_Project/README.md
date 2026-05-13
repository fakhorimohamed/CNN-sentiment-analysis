#  Facial Expression Recognition using CNNs

> **Reconnaissance Faciale à l'aide de Réseaux de Neurones Convolutifs : de l'Apprentissage au Déploiement**

A deep learning project for automatic facial emotion recognition, developed as a final-year project  the **Faculté des Sciences Dhar El Mahraz, Université Sidi Mohamed Ben Abdellah de Fès**.


---

##  Overview

This project explores automatic recognition of the **7 basic facial emotions** (anger, disgust, fear, happiness, sadness, surprise, neutral) using:

- A custom **CNN** model trained from scratch
- A fine-tuned **ResNet50V2** model using transfer learning (pre-trained on ImageNet)
- Cross-dataset transfer from **FER2013** to **CK+**

---

## 🗂️ Datasets

| Dataset | Images | Classes | Notes |
|---------|--------|---------|-------|
| [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) | 35,888 | 7 | 48×48 grayscale, noisy labels, class imbalance |
| [CK+](https://www.kaggle.com/datasets/shawon10/ckplus) | 593 sequences | 7 | Lab-controlled, high quality, FACS-labeled |

---

##  Models

### 1. Custom CNN
- 4 convolutional blocks (Conv2D → MaxPool → Dropout)
- Progressive filters: 128 → 256 → 512
- Fully connected layers: 512 → 256 → 7 (Softmax)
- Trained for 70 epochs, batch size 128
- **Validation accuracy: ~64%**

### 2. CNN + Data Augmentation
- Same architecture with random rotations (±15°), zoom (±10%), horizontal flips
- Improved generalization at the cost of raw validation accuracy (~58%)

### 3. ResNet50V2 (Fine-tuned on FER2013)
- Pre-trained on ImageNet, last 50 layers unfrozen
- Added: Dropout (25%) → BatchNorm → Flatten → Dense(64) → BatchNorm → Dropout (50%) → Dense(7, Softmax)
- Optimizer: Adam | Loss: Categorical Crossentropy
- **Validation accuracy: ~64.95%**

### 4. ResNet50V2 (Fine-tuned on CK+)
- Further fine-tuned from FER2013 weights onto CK+
- Last 6 layers unfrozen, learning rate 1e-4
- EarlyStopping + ReduceLROnPlateau callbacks
- Tested back on FER2013 to verify knowledge retention (~62.99%)

---

## Results Summary

| Model | Validation Accuracy |
|-------|-------------------|
| CNN (baseline) | ~64% |
| CNN + Augmentation | ~58% |
| ResNet50V2 (ImageNet → FER2013) | ~64.95% |
| ResNet50V2 (FER2013 → CK+, tested on FER2013) | ~62.99% |

> **Note:** Models pre-trained on face-specific datasets (e.g. VGGFace2) achieve higher accuracy (~71%) on this task, highlighting the importance of the pre-training domain.

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | Model building & training |
| NumPy / Pandas | Data processing |
| Jupyter Notebook | Model experimentation |
| VS Code | Development environment |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip pip install -r requirements.txt
```

### Train a Model

Open the relevant Jupyter notebook and run all cells. Make sure the FER2013 dataset is placed in the expected directory structure:

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── ...
```

---

## Project Structure

```
├── models/
│   ├── cnn_model.h5
│   ├── resnet50v2_fer.h5
│   └── resnet50v2_ckplus.h5
├── notebooks/
│   ├── 01_CNN_baseline.ipynb
│   ├── 02_CNN_augmentation.ipynb
│   ├── 03_ResNet50V2_FER2013.ipynb
│   └── 04_ResNet50V2_CKPlus.ipynb
├── data/
│   └── (FER2013 and CK+ datasets)
└── README.md
```

---

## Future Work

- Train on additional datasets: **AffectNet**, **JAFFE**
- Explore deeper architectures: **VGG16**, **EfficientNet**
- Integrate multimodal emotion analysis (facial expressions + voice + gestures)
- Pre-train on face-specific datasets like **VGGFace2** for better performance
