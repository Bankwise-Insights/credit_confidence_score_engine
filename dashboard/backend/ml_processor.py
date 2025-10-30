"""
ML Processor for Credit Score Prediction
Handles machine learning model inference and feature processing
"""

import joblib
import pandas as pd
import numpy as np
import os
import sys

class MLProcessor:
    def __init__(self):
        """Initialize ML processor with trained model"""
        self.model = None
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load the trained ML model"""
        try:
            # Try to load from root directory first
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'recommender_model.joblib')
            if not os.path.exists(model_path):
                # Try alternative path
                model_path = 'recommender_model.joblib'
            
            self.model = joblib.load(model_path)
            
            # Define expected feature columns (based on training data)
            self.feature_columns = [
                'Age', 'Income', 'MonthsEmployed', 'DTIRatio', 'Education_Graduate',
                'Education_High School', 'Education_Postgraduate', 'Education_Undergraduate',
                'EmploymentType_Full-time', 'EmploymentType_Part-time', 'EmploymentType_Salaried',
                'EmploymentType_Self-employed', 'MaritalStatus_Divorced', 'MaritalStatus_Married',
                'MaritalStatus_Single', 'HasMortgage', 'HasDependents', 'HasCoSigner',
                'AvgMonthlyBalance', 'AvgMonthlySavings', 'NumOverdraftsLast12Months',
                'SavingsRate', 'DepositFrequency', 'LastMonthSpending',
                'MinBalanceLast6Months', 'MaxBalanceLast6Months'
            ]
            
            print("✅ ML Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")
            self.model = None
    
    def prepare_features(self, applicant_data):
        """Prepare features for ML model prediction"""
        try:
            # Create a DataFrame with the applicant data
            df = pd.DataFrame([applicant_data])
            
            # Handle categorical variables with one-hot encoding
            categorical_mappings = {
                'Education': ['Graduate', 'High School', 'Postgraduate', 'Undergraduate'],
                'EmploymentType': ['Full-time', 'Part-time', 'Salaried', 'Self-employed'],
                'MaritalStatus': ['Divorced', 'Married', 'Single']
            }
            
            # Create one-hot encoded features
            for category, values in categorical_mappings.items():
                for value in values:
                    column_name = f"{category}_{value}"
                    df[column_name] = (df[category] == value).astype(int)
            
            # Select only the features expected by the model
            feature_df = pd.DataFrame()
            for col in self.feature_columns:
                if col in df.columns:
                    feature_df[col] = df[col]
                else:
                    # Set default value for missing features
                    feature_df[col] = 0
            
            return feature_df
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return None
    
    def predict_credit_score(self, applicant_data):
        """Predict credit score using ML model"""
        try:
            if self.model is None:
                return {
                    "predicted_score": 650,  # Default score
                    "confidence": 0.5,
                    "risk_category": "Medium",
                    "explanation": "ML model not available - using default score",
                    "model_used": "default"
                }
            
            # Prepare features
            features = self.prepare_features(applicant_data)
            if features is None:
                raise Exception("Feature preparation failed")
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get prediction probability/confidence if available
            confidence = 0.8  # Default confidence
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(features)[0]
                    confidence = max(proba)
                except:
                    pass
            
            # Determine risk category
            if prediction >= 700:
                risk_category = "Low"
            elif prediction >= 600:
                risk_category = "Medium"
            else:
                risk_category = "High"
            
            # Generate explanation
            explanation = self.generate_ml_explanation(applicant_data, prediction)
            
            return {
                "predicted_score": round(prediction, 0),
                "confidence": round(confidence, 2),
                "risk_category": risk_category,
                "explanation": explanation,
                "model_used": "trained_ml_model"
            }
            
        except Exception as e:
            print(f"❌ Error in ML prediction: {e}")
            return {
                "predicted_score": 600,  # Conservative default
                "confidence": 0.3,
                "risk_category": "Medium",
                "explanation": f"ML prediction failed: {str(e)}. Using conservative default score.",
                "model_used": "fallback"
            }
    
    def generate_ml_explanation(self, applicant_data, predicted_score):
        """Generate explanation for ML prediction"""
        try:
            explanations = []
            
            # Income factor
            income = applicant_data.get('Income', 0)
            if income > 100000:
                explanations.append("High income positively impacts score")
            elif income < 30000:
                explanations.append("Low income negatively impacts score")
            
            # DTI Ratio factor
            dti = applicant_data.get('DTIRatio', 0)
            if dti == 0:
                explanations.append("No existing debt obligations (excellent)")
            elif dti < 0.3:
                explanations.append("Low debt-to-income ratio (good)")
            elif dti > 0.5:
                explanations.append("High debt-to-income ratio (concerning)")
            
            # Employment stability
            months_employed = applicant_data.get('MonthsEmployed', 0)
            if months_employed > 24:
                explanations.append("Stable employment history")
            elif months_employed < 6:
                explanations.append("Limited employment history")
            
            # Savings behavior
            savings_rate = applicant_data.get('SavingsRate', 0)
            if savings_rate > 0.15:
                explanations.append("Excellent savings rate")
            elif savings_rate < 0.05:
                explanations.append("Low savings rate")
            
            # Overdraft history
            overdrafts = applicant_data.get('NumOverdraftsLast12Months', 0)
            if overdrafts == 0:
                explanations.append("No recent overdrafts")
            elif overdrafts > 3:
                explanations.append("Frequent overdrafts (negative factor)")
            
            if not explanations:
                explanations.append("Score based on overall financial profile analysis")
            
            return f"ML Model Analysis: {'; '.join(explanations)}. Predicted credit score: {predicted_score:.0f}"
            
        except Exception as e:
            return f"ML model predicted credit score: {predicted_score:.0f}"
    
    def get_feature_importance(self, applicant_data):
        """Get feature importance for the prediction (if available)"""
        try:
            if self.model is None or not hasattr(self.model, 'feature_importances_'):
                return None
            
            features = self.prepare_features(applicant_data)
            if features is None:
                return None
            
            importance_dict = {}
            for i, col in enumerate(self.feature_columns):
                if i < len(self.model.feature_importances_):
                    importance_dict[col] = self.model.feature_importances_[i]
            
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_importance[:10]  # Top 10 features
            
        except Exception as e:
            print(f"❌ Error getting feature importance: {e}")
            return None