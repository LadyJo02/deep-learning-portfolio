# Project 1: PulmoScope

**PulmoScope** is a deep-learning–based assistive system for disease-centered respiratory sound analysis.  
This project investigates the effectiveness of **temporal deep learning models**, with emphasis on a **Hybrid Temporal Convolutional Network–Spiking Neural Network (TCN-SNN)**, for multi-class classification of lung diseases using auscultation sounds.

---

## Final Manuscript

The complete research paper detailing the background, methodology, experimental framework, results, and discussion is provided below:

**[PulmoScope – Final Manuscript (PDF)](DL-FINAL-PROJECT-PULMOSCOPE.pdf)**

> This manuscript presents a comparative evaluation of **TCN-SNN, Pure TCN, LSTM, and Vanilla RNN** architectures under standardized preprocessing, balancing, and evaluation protocols. :contentReference[oaicite:1]{index=1}

---

## Exploratory Data Analysis (EDA)

**EDA Notebook:** `EDA_Pulmoscope.ipynb`

This notebook focuses on:
- Clinical–demographic data exploration  
- Class distribution and imbalance analysis  
- Temporal and spectral characteristics of lung sounds  
- Statistical relationships between age, gender, and diagnoses  

The EDA phase guided key modeling decisions such as **class consolidation**, **balancing strategies**, and **feature design**.

---

## Model Development and Experiments

**Modeling Notebook:** `PulmoScope.ipynb`

This notebook documents:
- Audio preprocessing and feature extraction  
- Hybrid Mel-Spectrogram + MFCC feature stacking  
- Implementation of:
  - Hybrid **TCN-SNN**
  - Pure **TCN**
  - **LSTM**
  - **Vanilla RNN**
- Architecture comparison, hyperparameter tuning, and evaluation  
- Confusion matrix, ROC-AUC, and Grad-CAM interpretability analysis  

The notebook demonstrates an end-to-end deep learning workflow for respiratory disease classification.

---

## Learning Outcomes

Through this project, I learned how to:
- Design disease-centered respiratory sound classification pipelines  
- Handle severe class imbalance in medical audio datasets  
- Compare temporal deep-learning architectures fairly  
- Interpret model decisions using explainability techniques (Grad-CAM)  
- Align deep-learning experiments with clinical relevance and safety
