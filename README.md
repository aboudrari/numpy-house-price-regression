🏠 NumPy House Price Regression

An end-to-end House Price Regression project built entirely with NumPy, without relying on high-level machine learning libraries such as Scikit-learn.

The project implements the complete machine learning pipeline from data cleaning and feature engineering to Ordinary Least Squares (OLS) regression, evaluation, and residual analysis.

⸻

🚀 Project Overview

The goal of this project is to predict house prices using a tabular dataset while implementing the core components of a regression pipeline from scratch.

The workflow includes:

Raw Data → Cleaning → Feature Engineering → Encoding → Standardization → Train/Validation/Test Split → OLS Regression → Predictions → Evaluation

The model is trained using the Normal Equation, providing a practical implementation of linear regression using only NumPy.

⸻

🧠 What This Project Covers

* Handling missing values
* Detecting and handling outliers using IQR
* Feature engineering
* Ratio-based features
* Categorical feature encoding
* Feature standardization
* Reproducible dataset splitting
* Linear regression from scratch
* OLS using the Normal Equation
* Model predictions
* Regression evaluation metrics
* Residual analysis

⸻

🛠️ Tech Stack

* Python
* NumPy
* Linear Algebra
* Statistics
* Machine Learning

No Scikit-learn or high-level ML framework is used for the regression pipeline.

⸻

📊 Machine Learning Pipeline

1. Data Cleaning

Missing numerical values are handled using mean imputation.

Outliers are detected using the Interquartile Range (IQR) method and clipped to predefined bounds.

2. Feature Engineering

Additional features are created from the original data, including ratio-based features.

Categorical variables are converted into numerical representations using one-hot encoding.

3. Feature Scaling

Numerical features are standardized before training.

A bias/intercept column is then added to the feature matrix.

4. Dataset Splitting

The dataset is shuffled using a reproducible random process and divided into:

* Training set
* Validation set
* Test set

5. Model Training

The regression model is implemented from scratch using Ordinary Least Squares (OLS) and the Normal Equation.

6. Evaluation

The trained model is evaluated on held-out data using:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — R-squared
* Residual Statistics

⸻

✅ Implementation Checklist

Data Preprocessing

* impute_nan_with_mean
* compute_iqr_bounds
* clip_columns

Feature Engineering

* make_ratio_feature
* append_column
* one_hot_encode

Feature Scaling

* fit_standardizer
* apply_standardizer
* add_bias_column

Dataset Preparation

* make_shuffled_indices
* partition_indices
* subset_xy

Regression Model

* ols_fit
* ols_predict

Evaluation

* mean_absolute_error
* root_mean_squared_error
* r_squared
* residual_summary

Pipeline

* prepare_cleaned_features
* assemble_feature_matrix
* make_train_val_test
* standardize_and_add_bias

⸻

▶️ How to Run

Clone the repository and run:

python scaffold.py

The script executes the complete preprocessing, training, and evaluation pipeline.

⸻

📚 Learning Objectives

This project is designed to strengthen understanding of the mathematical and practical foundations behind regression models.

Instead of using a ready-made implementation, the main components are built using NumPy, making it easier to understand what happens internally during a typical machine learning workflow.

Key concepts include:

* Linear algebra
* Matrix operations
* Feature preprocessing
* One-hot encoding
* Standardization
* Least Squares
* Model evaluation
* Residual analysis

⸻

🎯 Project Status

Status: Completed ✅

All major components of the NumPy regression pipeline have been implemented.

⸻

📖 Reference

Built as part of the Deep-ML learning journey.