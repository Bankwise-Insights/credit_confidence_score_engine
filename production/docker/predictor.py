import os
import io
import joblib
import pandas as pd
import google.generativai as genai
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse

# --- Configuration & Model Loading ---
MODEL_DIR = '/opt/ml/model'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
MODEL_COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.joblib')

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()
model, model_columns, gemini_model = None, None, None

@app.on_event("startup")
def load_resources():
    """Load all necessary models and resources at startup."""
    global model, model_columns, gemini_model
    try:
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(MODEL_COLUMNS_PATH)
        print("Scikit-learn model and columns loaded successfully.")
    except Exception as e:
        print(f"Error loading scikit-learn model: {e}")
    
    if GEMINI_API_KEY:
        try:
            gemini_model = genai.GenerativeModel('gemini-pro')
            print("Gemini Pro model initialized successfully.")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")

# --- Helper Functions ---
def generate_gemini_prompt(applicant):
    return f"""
    You are an expert AI Loan Advisor. Based on the following applicant profile, evaluate the 5 C's of credit and make a final loan recommendation.

    The response MUST follow this structured style:
    ⭐ The 5 C's of Credit Evaluation

    CHARACTER X/10
    - Repayment History: ...
    - Financial Behavior: ...

    CAPACITY X/10
    - Income-to-Expense Ratio: ...
    - Debt-to-Income Post-Loan: ...

    CAPITAL X/10
    - Savings Rate: ...
    - Financial Cushion: ...

    COLLATERAL X/10 or N/A
    - Type: ...
    - Security: ...

    CONDITIONS X/10
    - Economic Environment: ...
    - Employment Status: ...

    Overall 5 C's Assessment: [Brief 1–2 line assessment]
    Final Risk Assessment: [LOW | MEDIUM | HIGH] RISK
    LOAN RECOMMENDATION: [YES | NO ]

    Applicant Profile:
    - Age: {applicant['Age']}
    - Income(KES): {applicant['Income']}
    - Months Employed: {applicant['MonthsEmployed']}
    - Debt-to-Income Ratio: {applicant['DTIRatio']}
    - Education: {applicant['Education']}
    - Employment Type: {applicant['EmploymentType']}
    - Marital Status: {applicant['MaritalStatus']}
    - Has Mortgage: {applicant['HasMortgage']}
    - Has Dependents: {applicant['HasDependents']}
    - Loan Purpose: {applicant['LoanPurpose']}
    - Has Cosigner: {applicant['HasCoSigner']}
    - Predicted Credit Score: {applicant['CreditScore']} 
    - Industry: {applicant.get('Industry', 'Not specified')}
    - Economic Environment: {applicant.get('EconomicEnvironment', 'Not specified')}
    - Personal Circumstances: {applicant.get('PersonalCircumstances', 'Not specified')}

    Evaluate and return the structured response only.
    """

# --- API Endpoints ---
@app.get('/ping')
async def ping():
    """SageMaker health check endpoint."""
    if model and model_columns and gemini_model:
        return JSONResponse(status_code=200, content={"status": "healthy"})
    else:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "detail": "A model or resource failed to load"})

@app.post('/invocations')
async def invocations(request: Request):
    """
    Unified endpoint that accepts a batch of applicant data (JSON or CSV),
    predicts credit scores, and returns a full recommendation for each.
    """
    if not all([model, model_columns, gemini_model]):
        raise HTTPException(status_code=503, detail="A required model or resource is not loaded")

    content_type = request.headers.get('Content-Type')
    features_df = None

    # --- Step 1: Parse Input Data into a DataFrame ---
    try:
        if content_type == 'application/json':
            # Expects a list of applicant dictionaries
            applicant_list = await request.json()
            if not isinstance(applicant_list, list):
                raise ValueError("JSON input must be a list of applicant objects.")
            features_df = pd.DataFrame(applicant_list)
        
        elif content_type == 'text/csv':
            body = await request.body()
            data_csv = body.decode('utf-8')
            raw_feature_columns = [
                "Age", "Income", "MonthsEmployed", "DTIRatio", "Education", "EmploymentType", 
                "MaritalStatus", "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner", 
                "Industry", "EconomicEnvironment", "PersonalCircumstances"
            ]
            features_df = pd.read_csv(io.StringIO(data_csv), header=None, names=raw_feature_columns)
            
        else:
            raise HTTPException(status_code=415, detail=f'Unsupported content type: {content_type}. Use application/json or text/csv.')

        if features_df.empty:
            raise ValueError("Received no applicant data.")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing input data: {str(e)}")

    # --- Step 2: Predict Credit Scores for the entire batch ---
    try:
        features_encoded = pd.get_dummies(features_df)
        features_aligned = features_encoded.reindex(columns=model_columns, fill_value=0)
        
        predicted_scores = model.predict(features_aligned)
        features_df['CreditScore'] = [int(round(score)) for score in predicted_scores]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing data for prediction: {str(e)}")
        
    # --- Step 3: Get Recommendation from Gemini for each applicant ---
    results = []
    for _, applicant_row in features_df.iterrows():
        applicant_data = applicant_row.to_dict()
        try:
            prompt = generate_gemini_prompt(applicant_data)
            gemini_response = gemini_model.generate_content(prompt)
            
            results.append({
                "predicted_credit_score": applicant_data['CreditScore'],
                "recommendation": gemini_response.text
            })
        except Exception as e:
            # If one recommendation fails, record the error and continue
            results.append({
                "predicted_credit_score": applicant_data.get('CreditScore', 'Error'),
                "recommendation": f"Error: Failed to get recommendation from Gemini API: {str(e)}"
            })
            
    return JSONResponse(content={"results": results})

