# Telco Customer Churn Prediction using Data Mining

## Overview
This project aims to predict customer churn for a telecommunications company using various **Data Mining** and **Machine Learning** techniques. Customer Churn (the rate at which customers stop doing business with an entity) is a critical metric. This study compares the performance of **Support Vector Machine (SVM)**, **Naive Bayes**, and **Decision Tree** algorithms to identify customers likely to leave, enabling proactive retention strategies.

## Dataset
The project utilizes the **Telco Customer Churn Dataset** sourced from Kaggle.
* **Samples:** 7,043 Customer Records.
* **Structure:** **Imbalanced Dataset** (73% No Churn vs. 27% Yes Churn).
* **Features:** 21 attributes including demographics (Gender, Senior Citizen), services (Internet, Phone), and account information (Contract, Payment Method).
* **Target:** `Churn` (Yes: Customer Left, No: Customer Stayed).

## Methodology
An end-to-end data mining pipeline was implemented to handle the specific challenges of the dataset:

### 1. Data Preprocessing
* **Data Cleaning:** Handled missing values in `TotalCharges` and dropped irrelevant columns (`customerID`).
* **Encoding:** Applied **Label Encoding** to convert categorical variables (e.g., Partner, Contract) into numeric format.
* **Scaling:** Used **MinMaxScaler** to normalize features to a 0-1 range. This step is crucial for distance-based algorithms like SVM to prevent features with large magnitudes (e.g., TotalCharges) from dominating the model.
* **Handling Imbalance:** Unlike balanced datasets, this project required handling class imbalance. **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the **Training set** to balance the classes and improve the model's ability to detect Churners (Minority Class).

### 2. EDA (Exploratory Data Analysis)
* **Correlation Analysis:** Identified relationships between features and the target variable.
* **Distribution Analysis:** Examined feature distributions to understand customer profiles.
* **Outlier Analysis:** Investigated outliers in numerical columns (`MonthlyCharges`, `TotalCharges`) but decided to retain them and handle them via Scaling to preserve valuable information about high-value customers.

### 3. Modeling
Three supervised learning algorithms were trained and evaluated:
* **Support Vector Machine (SVM):** Configured with **RBF Kernel** to handle non-linear relationships.
* **Naive Bayes:** Used as a probabilistic baseline model.
* **Decision Tree:** Implemented to capture hierarchical decision rules.

## Results
Given the imbalanced nature of the data, the models were evaluated focusing on **Recall** (to minimize missed churners) and **F1-Score**, alongside Accuracy and ROC-AUC.

| Model | Accuracy | Recall | Precision | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | **79.00%** | **84.00%** | **77.00%** | **80.00%** | **0.860** |
| Naive Bayes | 77.00% | 81.00% | 76.00% | 78.00% | 0.840 |
| Decision Tree | 75.00% | 76.00% | 76.00% | 76.00% | 0.750 |

> **Key Finding:** The **SVM model** outperformed others, achieving the highest **Recall (84%)** and **AUC (0.86)**. While Decision Tree and Naive Bayes provided competitive results, SVM demonstrated superior capability in distinguishing between Churn and Non-Churn customers.

# Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/dilademirhan/telco-churn-prediction.git](https://github.com/dilademirhan/telco-churn-prediction.git) Data_Mining_Project
cd Data_Mining_Project
