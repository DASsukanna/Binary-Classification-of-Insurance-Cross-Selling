
# Binary Classification Using XGBoost
## Overview
This project performs binary classification to predict a response variable using XGBoost. The project includes data preprocessing, feature engineering, and hyperparameter tuning to improve model accuracy. The model was trained on data containing various demographic, behavioral, and financial features to classify outcomes effectively.

## Table of Contents
- [Overview](#overview)
- [Data Preprocessing](#data-processing)
- [Feature Engineering](#feature-engineering)
- [Model Training and Hyperparameter Tuning](#model-training-and-hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Making Predictions](#making-prediction)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Usage](#usage)
  
## Data Preprocessing
### Steps:
- **Loaded the Dataset**: Imported the training data and displayed the first few rows.
- **Checked Null Values**: Verified that there were no missing values across columns.
- **Explored Unique Values**: Counted unique values for each column to understand data variability.
- **Feature Engineering**
  
### Binning:

- **Age**: Divided into bins representing different age ranges (e.g., 10-20, 20-30).
- **Annual Premium**: Binned into ranges for easier categorization and modeling.
  
## Label Encoding:

Encoded categorical variables such as Gender, Vehicle_Age, Vehicle_Damage, and Age_Binned using LabelEncoder to convert them into numerical format.

## Handling Missing Values:

- **Annual Premium Binned**: Filled missing values with the mode and converted the column to integer type for consistency.
- **Model Training and Hyperparameter Tuning**
  
## XGBoost Workflow Overview

- **Model Initialization**: Initialized the XGBoost classifier with a fixed random_state for reproducibility.
- **Hyperparameter Tuning with RandomizedSearchCV**:
Defined parameter ranges for n_estimators, max_depth, learning_rate, gamma, reg_alpha, and reg_lambda.
Performed RandomizedSearchCV with 5-fold cross-validation to find the best parameters for the model.
- **Best Model Evaluation**:
The tuned model’s performance was assessed using accuracy_score, confusion_matrix, and classification_report.
- **Evaluation**:
- Accuracy: Achieved an accuracy of 0.88.
- Confusion Matrix and Classification Report: Provided detailed metrics, including precision, recall, and F1-score, to evaluate model performance on each class.
- **Making Predictions**
- Test Data Preprocessing: Applied the same preprocessing and feature engineering steps to the test data.
- **Prediction**:
Used the best XGBoost model to predict probabilities on the test dataset.
Saved predictions to predictions.csv with id and Response columns.
## File Structure
- **train.csv**: Training data used for model training.
- **test.csv**: Test data for making final predictions.
- **predictions.csv**: Output file containing predicted probabilities for the test data.

## Requirements
- **pandas**
- **numpy**
- **scikit-learn**
- **xgboost**
- **scipy**
