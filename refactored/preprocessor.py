import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Preprocessing 
print("--- Step 1: Data Loading and Preprocessing ---")


try:
    df = pd.read_csv('loans.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'loans.csv' not found. Please ensure the file is in the same directory.")
    exit()

# Initial information about the dataset
print("\nDataFrame Info:")
df.info()

print("\nFirst 5 rows of the DataFrame:")
df.head()

target_col = 'CreditScore'
features = df.drop(columns=[target_col])
target = df[target_col]

# Categorical and numerical columns
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
numerical_cols = [col for col in features.columns if col not in categorical_cols]

# One-Hot Encoding on categorical features
print("\nApplying One-Hot Encoding to categorical features...")
X_encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

print("\nShape of the new features DataFrame after One-Hot Encoding:")
print(X_encoded.shape)
print("\nFirst 5 rows of the encoded features:")
print(X_encoded.head())

X_train, X_test, y_train, y_test = train_test_split(X_encoded, target, test_size=0.2, random_state=42)
print("\nData split into training and testing sets.")
print(f"Training set size: {X_train.shape[0]} records")
print(f"Testing set size: {X_test.shape[0]} records")


