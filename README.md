# 🔥 Fire Detection from Camera Images using Deep Learning (CNN)
### Applied Artificial Neural Network (AANN) — Case Study Project
**Institution:** Chaitanya Bharathi Institute of Technology (CBIT)  
**Subject:** Applied Artificial Neural Network  

---

## 📌 Project Overview
Traditional smoke and heat detectors suffer from critical latency — smoke often needs to travel and accumulate before a detector triggers, and manual alarms rely entirely on human observation.

This project implements an end-to-end Computer Vision pipeline using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras to automatically classify surveillance and camera images as **`Fire`** or **`No Fire`** in real time.

---

## 📊 Dataset & Stratified Splits
The dataset consists of **2,506** camera images labeled across two classes (`fire` and `nofire`). An independent, stratified 70/15/15 train-validation-test split was used to guarantee unbiased evaluation:

| Split | Images | No Fire | Fire |
|---|---|---|---|
| **Training (70%)** | 1,754 | 865 | 889 |
| **Validation (15%)** | 376 | 185 | 191 |
| **Test (15%)** | 376 | 185 | 191 |
| **Total** | **2,506** | **1,235** | **1,271** |

---

## 🏗️ Architecture & Training
- **Convolutional Feature Extractor:** Multi-layer Conv2D blocks with ReLU activations and MaxPooling2D downsampling to extract low- and high-level visual features (flames, smoke textures, color contrast).
- **Regularization:** Dropout layers to prevent overfitting.
- **Classification Head:** Dense layers with a Sigmoid activation function outputting probability $P(\text{fire} \mid x) \in [0, 1]$.
- **Optimization & Callbacks:**
  - Optimizer: Adam
  - Loss: Binary Cross-Entropy
  - Callbacks: `EarlyStopping` (monitoring validation loss) and `ModelCheckpoint` (saving the best model based on validation ROC AUC).

---

## 📈 Evaluation & Performance Metrics

### Final Test Split Results (Hold-out Test Set)
- **Test Accuracy:** **94.41%**
- **Test ROC AUC:** **0.9875**
- **Validation ROC AUC:** **0.9838**

#### Detailed Classification Report (Test Set)
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **No Fire** | 0.91 | 0.98 | 0.95 | 185 |
| **Fire** | **0.98** | **0.91** | **0.94** | 191 |
| **Accuracy** | — | — | **0.94** | 376 |
| **Macro Avg** | 0.95 | 0.94 | 0.94 | 376 |
| **Weighted Avg** | 0.95 | 0.94 | 0.94 | 376 |

---

## 🎯 Threshold Analysis (Sensitivity Calibration)
In mission-critical fire safety systems, the cost of false negatives (missing a real fire) is far higher than false positives. The project analyzed decision thresholds from **0.10 to 0.90**:

- **Baseline Threshold (0.50):** Balanced precision (0.95) and recall (0.91).
- **High-Sensitivity Threshold (0.10):** Maximizes fire recall to **96.3%** ($F_2 = 0.957$) with only 7 false negatives, ideal for early alarm dispatch systems.

---

## 🔍 Model Explainability with Grad-CAM
To verify that the CNN learns actual visual features of combustion rather than background artifacts:
- **Grad-CAM (Gradient-weighted Class Activation Mapping)** was applied to the final convolutional layer.
- Generated heatmaps visually confirm high activation specifically on flame boundaries, glowing embers, and smoke plumes.

---

## 📂 Repository Contents
```text
├── Fire_Detection_CNN_with_Test_Split.ipynb                               # Complete Jupyter Notebook with code & cell outputs
├── Fire_Detection_CNN_with_Test_Split (1).ipynb                           # Notebook replica
├── Fire_Detection_CNN_Project_CBIT_Limitations_Future_Work_Conclusion.pptx # Comprehensive presentation deck
├── sample_images/                                                         # Sample test images for inference verification
│   ├── fire in room.jpg
│   ├── forest fire.webp
│   └── no fire image.webp
├── requirements.txt                                                       # Python package dependencies
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/vignesh124-ai/Fire-Detection-CNN.git
cd Fire-Detection-CNN
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Open in Jupyter or Google Colab
Launch the notebook in Jupyter Notebook, JupyterLab, or Google Colab:
```bash
jupyter notebook "Fire_Detection_CNN_with_Test_Split.ipynb"
```

---

## 📋 Case Study Presentation Summary (`.pptx`)
The accompanying presentation covers:
1. **Problem Statement & Objectives:** Real-world fire delays, smoke detection latency, manual reporting bottlenecks.
2. **Dataset & Stratification Strategy:** Balanced splits ensuring zero test-set leakage.
3. **CNN Architecture & Hyperparameters:** Filter progression, pooling, and regularization.
4. **Error Analysis & Cutoff Exploration:** Sensitivity vs. specificity trade-off curves.
5. **Grad-CAM Visual Verification:** Interpretable AI verification of activations.
6. **Limitations & Future Work:**
   - *Limitation:* Tested on static labeled frames; temporal progression was outside the single-frame scope.
   - *Future Work:* Integration of temporal sequence modeling (e.g., CNN-LSTM / 3D-CNN) for live video stream monitoring and edge deployment on IoT devices (Raspberry Pi / NVIDIA Jetson).

---

## 👨‍💻 Author
- **Bhuma Vignesh** ([@vignesh124-ai](https://github.com/vignesh124-ai))
- CBIT (Chaitanya Bharathi Institute of Technology)
