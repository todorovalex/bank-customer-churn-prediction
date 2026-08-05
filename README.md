# Bank Customer Churn Prediction

## Status: In Progress (Aug 2026)

## Overview
Predicting customer churn in digital banking using classification models. 
The goal is to identify customers at high risk of leaving a bank, understand 
the key factors driving churn, and translate those insights into actionable 
retention recommendations.

## Data Source
[Bank Customer Churn Dataset](enlace de Kaggle) — 10,000 customer records from 
a European bank, including demographics, account activity, and engagement 
metrics. Target variable: `Exited` (1 = churned, 0 = retained).

Note: the original dataset includes a `Complain` column, which was excluded 
from modeling due to data leakage — nearly all customers who complained 
subsequently churned, making it an unrealistic predictor available "before 
the fact."

## Approach
1. Exploratory Data Analysis (EDA)
2. Data cleaning and feature encoding (categorical variables: Geography, 
   Gender, Card Type)
3. Handling class imbalance (~20% churn rate)
4. Model training and comparison (Logistic Regression, Random Forest, XGBoost)
5. Evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
6. Feature importance analysis
7. Business recommendations based on key churn drivers

## Results
TBD

## Business Recommendations
TBD

## Tech Stack
Python, pandas, scikit-learn, matplotlib/seaborn, Google Colab

## Repository Structure
```
data/         # dataset (or link to download)
notebooks/    # analysis notebook
src/          # reusable functions (if applicable)
```
