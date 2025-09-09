import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args, _ = parser.parse_known_args()

    # --- 1. Data Loading and Preprocessing ---
    print("--- Step 1: Data Loading and Preprocessing ---")
    input_data_path = os.path.join(args.train, 'loans.csv')
    df = pd.read_csv(input_data_path)

    target_col = 'CreditScore'
    features = df.drop(columns=[target_col])
    target = df[target_col]

    categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
    features_encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

    model_columns = list(features_encoded.columns)
    joblib.dump(model_columns, os.path.join(args.model_dir, "model_columns.joblib"))
    print("Model columns saved.")

    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # --- 2. Model Training and Evaluation ---
    print("\n--- Step 2: Training and Evaluating Models ---")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    best_model = None
    best_rmse = float('inf')
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"  -> {name} RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            print(f"  -> New best model found: {name}")

    print(f"\n--- Best model selected: {type(best_model).__name__} with RMSE: {best_rmse:.4f} ---")

    # --- 3. Save The Best Model ---
    model_path = os.path.join(args.model_dir, "model.joblib")
    print(f"Saving best model to {model_path}...")
    joblib.dump(best_model, model_path)
    print("Model saved successfully.")

