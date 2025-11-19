# 💳 Credit Card Fraud Detection: A Comparative ML Approach

## Project Overview

This repository contains an end-to-end Machine Learning solution for detecting fraudulent transactions using a highly **imbalanced** dataset of credit card transactions. The primary goal of this project was to establish a robust classification pipeline capable of identifying rare fraud events with high reliability.

The core analysis is detailed within the `Credit_Card_Fraud.ipynb` Jupyter Notebook.

---

## 🔬 Methodology and Technical Challenges

The project's complexity stems from the extreme class imbalance, where fraudulent transactions account for less than $0.2\%$ of the total data. Addressing this required careful methodology:

### 1. Handling Imbalanced Data
* **SMOTE (Synthetic Minority Over-sampling Technique):** Applied to the training data to generate synthetic minority samples, effectively mitigating class bias during model training.
* **Stratified Cross-Validation:** Utilized **Stratified K-Fold** to ensure that the class distribution (the ratio of fraud to non-fraud) was preserved across every fold of the cross-validation process, yielding unbiased performance metrics.

### 2. Model Comparison and Selection
A diverse set of classifiers were evaluated to determine the optimal solution for this high-stakes task:
* Logistic Regression
* Random Forest
* XGBoost and LightGBM (Gradient Boosting Ensembles)
* Support Vector Classifier (SVC)
* K-Nearest Neighbors (KNN)

### 3. Evaluation Strategy
Given the context of fraud detection, **Accuracy is insufficient**. The evaluation strategy was centered on specialized metrics:
* **ROC-AUC (Area Under the Receiver Operating Characteristic Curve):** Measures the model’s discrimination capacity.
* **Average Precision Score (PR-AUC):** The most critical metric for severe imbalance, providing a robust measure of precision across all recall thresholds.

---

## 📊 Performance Summary

The comprehensive cross-validation results are summarized below, highlighting models with the highest discrimination and precision capabilities.

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.954** | **0.887** | 0.999 | 0.94 | 0.93 | 0.93 |
| **XGBoost** | 0.941 | 0.855 | 0.994 | 0.91 | 0.92 | 0.932 |


---

## 🛠️ Data Acquisition and Setup

The dataset used is the anonymized `creditcardfraud` dataset available on Kaggle.

To reproduce the data download step within the notebook, users must have their **Kaggle API key** correctly configured on their local machine (placed as `kaggle.json` in the `~/.kaggle/` directory).

---

## 🧑‍💻 Author and Technical Profile

This project showcases initial expertise in data pipeline construction, model evaluation for imbalanced data, and applied machine learning techniques.

* **Author:** **Ololade Akinsanola**
* **Context:** Developed as the final project for **COMP 479: Introduction to Machine Learning**.
* **Skills Demonstrated:** Python, Pandas, NumPy, Scikit-learn, Imblearn, XGBoost, LightGBM, Data Preprocessing (Standardization), Cross-Validation.
