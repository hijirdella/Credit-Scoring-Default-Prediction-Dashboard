# Credit Scoring Default Prediction Dashboard
https://credit-scoring-default-prediction-dashboard.streamlit.app/<br>
A Streamlit web application for Credit Scoring and Default Prediction.  
This project demonstrates a complete end-to-end workflow from raw data upload and EDA to machine learning–based probability scoring, risk segmentation, and result export.

---

## Overview

The dashboard enables users to:
- Upload raw credit data (e.g., `combined_df.csv`).
- Perform automated exploratory data analysis (EDA).
- Generate probability of default (PD) scores using a trained model.
- Visualize customer risk through decile segmentation.
- Download the final scored dataset for business use.

This tool is designed for data science and risk analytics practitioners who need an interpretable, deployable, and business-oriented credit scoring application.

---

## Key Features

### 1. Data Upload and EDA
- Upload customer loan data in CSV format.
- View data summary, types, and missing values.
- Explore numeric distributions and demographic variables.
- Analyze default label distribution (if available).

### 2. Model Scoring
- Uses a pre-trained Logistic Regression credit scoring model (`best_credit_scoring_logreg.pkl`).
- Predicts probability of default (PD) per customer.
- Allows adjustment of the decision threshold.
- Displays prediction summary (Good vs Bad ratio).

### 3. Decile Segmentation
- Ranks customers by risk score into ten deciles (1 = lowest risk).
- Shows average probability and customer share per decile.
- Helps simulate portfolio acceptance strategy (e.g., 2% cumulative default rate).

### 4. Export Results
- Download the scored dataset as `scored_customers.csv`.

---

## Model Summary

The included model (`best_credit_scoring_logreg.pkl`) was trained on engineered customer-level features using a Logistic Regression pipeline.  
It handles class imbalance and was evaluated using ROC-AUC, KS, and Gini metrics.  
You can replace this model with another `.pkl` file that follows the scikit-learn pipeline format.

---
