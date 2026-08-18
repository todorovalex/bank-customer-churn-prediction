# Dataset

This project uses the Bank Customer Churn dataset from Kaggle.

Source: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

Download the CSV and place it in this folder as `bank_churn_data.csv` 
to run the notebook.

Note: the `Complain` column was removed before analysis due to data leakage 
(it is nearly perfectly correlated with churn, as it reflects post-departure 
behavior rather than a predictor available beforehand).
