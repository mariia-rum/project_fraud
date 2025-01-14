# Credit Card Fraud Detection

## Overview

This repository outlines the workflow for deploying a machine learning model as a web application on Heroku to detect credit card fraud. The dataset used for this project was sourced from Kaggle, provided by Vesta Corporation, a leading payment service company aiming to enhance fraud prevention strategies. The dataset is available [here](https://www.kaggle.com/c/ieee-fraud-detection).

---

## Project Workflow

### **Step 1: Data Preprocessing**
- **Data Sources**: The initial data consists of four CSV files: `train_identity`, `test_identity`, `train_transactions`, and `test_transactions`.
- **Data Merging**: These files were combined into a single dataset to streamline the cleaning process.
- **Handling Missing Values**: Rows containing more than 60% missing values were removed.

---

### **Step 2: Feature Engineering**
Several features were engineered to enhance model performance, including:
- **Time-Based Features**: `Weekday` and `Hour of the Day`.
- **Credit Card Features**: Attributes specific to each card.
- **Transaction Distance**: The deviation of a transaction's value from the median value of transactions on the same card, relative to fraud cases.

**Note**: Features derived from user email domains (e.g., email bins such as Gmail) were tested but excluded as they did not improve the model's performance.

---

### **Step 3: Column Transformer**
- **Feature Separation**: The dataset was divided into numerical and categorical features.
- **Numerical Data**: Missing values were imputed based on the proportion of missing data:
  - **15%-60% Missing**: Imputed with the mean.
  - **<15% Missing**: Imputed with the median.
- **Categorical Data**: Missing values were not imputed to avoid potential biases. Instead, missing values were labeled as `"Unknown"`.

---

### **Step 4: Model Training**
Multiple models were trained, and their performance was evaluated based on recall and F1 scores:

| Model           | Recall | F1   |
|------------------|--------|------|
| SVC             | 0.65   | 0.09 |
| XGBoost         | 0.26   | 0.39 |
| Logistic        | 0.71   | 0.00 |
| LightGBM        | 0.17   | 0.28 |
| KNN             | 0.074  | 0.074|
| Random Forest   | 0.32   | 0.48 |

**Selected Model**: Random Forest, demonstrating the best performance, was chosen as the final model.

---

### **Step 5: Web App Production**
- **Deployment**: The finalized model was deployed on Heroku using Flask.
- **Functionality**: The web application enables users to upload a CSV file containing transaction details. The model predicts whether each transaction is fraudulent or not.
- **Web Application**: [Visit the app](https://dash-fraud.herokuapp.com).

---






