# WESAD Stress Detection using Graph Attention Networks (GRAM)

## 📌 Project Overview
This project detects human stress levels using physiological data (ECG, EDA, EMG, Temp) from the **WESAD (Wearable Stress and Affect Detection)** dataset. 

It utilizes a **Graph Attention Network (GRAM)** to model the temporal dependencies between sensor readings, effectively filtering noise and identifying stress events with high precision.

## 🚀 Key Features
* **Multimodal Fusion:** Integrates Chest (700Hz) and Wrist (32-64Hz) sensor data.
* **Graph Neural Network:** Implements a GRAM architecture with attention mechanisms.
* **Efficient Training:** Utilizes `NeighborLoader` for mini-batch training on 4.2 million nodes using a single T4 GPU.
* **High Accuracy:** The model effectively distinguishes between Baseline, Stress, and Amusement states.

## 📊 Results
The model was trained for 15 epochs. The confusion matrix below demonstrates near-perfect classification on the test set:

![Confusion Matrix](real_confusion_matrix.png)

* **Stress Detection:** 99.9% Recall (Correctly identified 49,490 out of 49,492 stress instances).
* **Amusement Detection:** 99.9% Recall.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch, PyTorch Geometric
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

## 📂 Dataset
This project uses the WESAD dataset from the University of Siegen.
* **Note:** The training script automatically downloads the `S2.pkl` subject data (~2GB) for training.

---
*Created by Prem Modi*
