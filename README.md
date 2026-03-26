# 🫁 LungVision — Deep Learning for Chest X-Ray Diagnosis

> **Comparative Analysis of InceptionResNetV2 vs VGG19 for Multi-Label Lung Infection Classification**  
> NIH ChestX-ray14 Dataset · Transfer Learning · Keras/TensorFlow · 112,120 Images

---

## 📌 Overview

LungVision is a deep learning research project that tackles automated detection of **13 lung pathologies** from chest X-ray images using two state-of-the-art CNN architectures. Built on the NIH ChestX-ray14 dataset (112,120 images, 30,805 patients), it provides a rigorous side-by-side comparison of **InceptionResNetV2** and **VGG19** via transfer learning.

The goal: replace subjective, time-consuming manual radiograph interpretation with a fast, consistent, and accurate AI-powered diagnostic assistant.

---

## 🏆 Key Results

| Metric | VGG19 (4 epochs) | InceptionResNetV2 (5 epochs) |
|---|---|---|
| Validation Loss | 1.9796 | **1.4179** ✅ |
| Binary Accuracy | 87.72% | **88.94%** ✅ |
| Mean Absolute Error | 0.1228 | **0.1135** ✅ |

> **Winner: InceptionResNetV2** — outperforms VGG19 across all key metrics.

---

## 🦠 Diseases Classified (13 Labels)

| | | | |
|---|---|---|---|
| Atelectasis | Consolidation | Infiltration | Pneumothorax |
| Edema | Emphysema | Fibrosis | Effusion |
| Pneumonia | Pleural Thickening | Cardiomegaly | Nodule/Mass |
| Hernia | | | |

---

## 🗂️ Repository Structure

```
lungvision/
│
├── 📓 InceptionResnet.ipynb       # InceptionResNetV2 model — training & evaluation
├── 📓 VGG.ipynb                   # VGG19 model — training & evaluation
│
├── 📄 ML_Final_Project_Paper.pdf  # Full research paper
├── 📊 Presentation.pptx           # Project presentation slides
│
└── README.md
```

> **Note:** The NIH ChestX-ray14 dataset (~42 GB) is not included. See [Dataset Setup](#-dataset-setup) below.

---

## 🏗️ Architecture

Both models follow the same pipeline:

```
NIH Chest X-Ray Dataset
        │
        ▼
  Data Preprocessing
  (CSV labels → binary encoding, class balancing, augmentation)
        │
   ┌────┴────┐
   ▼         ▼
VGG19   InceptionResNetV2
   │         │
   ▼         ▼
GlobalAveragePooling2D
   │         │
   ▼         ▼
 Dropout → Dense(512) → Dropout → Dense(1024) → Dense(13, sigmoid)
   │         │
   ▼         ▼
  ROC/AUC Evaluation
        │
        ▼
  Best Model Selected
```

**Shared configuration:**
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Metrics: Binary Accuracy, MAE
- Input shape: 224×224×3 (InceptionResNetV2), 128×128×1 (VGG19)
- Epochs: 5 (InceptionResNetV2), 4 (VGG19)

**Model sizes:**
- InceptionResNetV2: ~55.6M parameters (54.3M base + 1.3M custom head)
- VGG19: ~20.8M parameters (20.0M base + 0.8M custom head)

---

## 📊 Dataset Setup

1. Download the **NIH ChestX-ray14** dataset from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
2. Place the images and `Data_Entry_2017.csv` in your working directory:

```
data/
├── images/
│   ├── 00000001_000.png
│   └── ...
└── Data_Entry_2017.csv
```

3. Update the `all_xray_df` path in the notebooks to point to your local data directory.

---

## ⚙️ Installation & Usage

### Prerequisites
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```

### Run InceptionResNetV2
```bash
jupyter notebook InceptionResnet.ipynb
```

### Run VGG19
```bash
jupyter notebook VGG.ipynb
```

Both notebooks are self-contained and walk through:
1. Data loading & preprocessing
2. Class balancing via weighted sampling
3. Data augmentation (rotation, flipping, zoom)
4. Model construction via transfer learning
5. Training with `fit_generator`
6. ROC curve & AUC evaluation
7. Visualization of predictions on the 8 "sickest" patients

---

## 🔬 Methodology Highlights

- **Transfer Learning:** Both architectures are initialized with ImageNet weights. The base layers are frozen; custom classification heads are fine-tuned on the NIH dataset.
- **Class Imbalance Handling:** Labels with < 1,000 cases are filtered out. Weighted sampling downsamples overrepresented classes.
- **Multi-Label Classification:** Each image can have multiple pathology labels simultaneously. Sigmoid activation enables independent probability output per class.
- **Evaluation:** ROC curves + AUC scores computed per disease label. Predictions visualized on the highest-burden test patients.

---

## 📈 Sample AUC Scores (InceptionResNetV2)

| Disease | AUC |
|---|---|
| Edema | 0.75 |
| Effusion | 0.55 |
| Cardiomegaly | 0.57 |
| Infiltration | 0.56 |
| Atelectasis | 0.55 |
| Pneumothorax | 0.51 |
| Fibrosis | 0.40 |

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## 📄 Citation

If you use this work, please cite:

```
Kommareddy, V. (2024). Prediction and Analysis of Lung Infections Using Deep Learning:
A Comparative Study of InceptionResNet and VGG Models.
Machine Learning CSCI 6364, George Washington University.
```

---

## 📬 Contact

**Vivek Kommareddy**  
Machine Learning · George Washington University  

---

*Built with ❤️ for better pulmonary diagnostics.*
