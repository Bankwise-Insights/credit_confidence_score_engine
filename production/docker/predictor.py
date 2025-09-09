import os
import joblib
import pandas as pd
import google.generativeai as genai
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse

# --- Environment and Model Loading ---
# Load the pre-trained model and column names from the SageMaker model directory
model_path = "/opt/ml/model"
model = joblib.load(os.path.join(model_path, "model.joblib"))
model_columns = joblib.load(os.path.join(model_path, "model_columns.joblib"))

# Configure the Gemini API
# The API key is passed as an environment variable from the SageMaker Estimator
try:
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except KeyError:
    gemini_model = None
    print("WARNING: GEMINI_API_KEY environment variable not set. Recommendation will fail.")

# --- FastAPI Application ---
app = FastAPI()

# --- Gemini Prompt Generation ---
def generate_gemini_prompt(applicant):
    prompt = f"""
    You are an expert AI Loan Advisor. Based on the following applicant profile, evaluate the 5 C's of credit and make a final Loan Recommendation.

    The response MUST follow this structured style:
    ⭐ The 5 C's of Credit Evaluation

    CHARACTER ...
    CAPACITY ...
    CAPITAL ...
    COLLATERAL ...
    CONDITIONS ...

    Overall 5 C's Assessment: [Brief 1–2 line assessment of strengths/weaknesses]
    Final Risk Assessment: [LOW | MEDIUM | HIGH] RISK
    LOAN RECOMMENDATION: [YES | NO]

    Applicant Profile:
    - Age: {applicant.get('Age', 'N/A')}
    - Income: {applicant.get('Income', 'N/A')}
    - Months Employed: {applicant.get('MonthsEmployed', 'N/A')}
    - Debt-to-Income Ratio: {applicant.get('DTIRatio', 'N/A')}
    - Education: {applicant.get('Education', 'N/A')}
    - Employment Type: {applicant.get('EmploymentType', 'N/A')}
    - Marital Status: {applicant.get('MaritalStatus', 'N/A')}
    - Has Mortgage: {applicant.get('HasMortgage', 'N/A')}
    - Has Dependents: {applicant.get('HasDependents', 'N/A')}
    - Loan Purpose: {applicant.get('LoanPurpose', 'N/A')}
    - Has Cosigner: {applicant.get('HasCoSigner', 'N/A')}
    - Predicted Credit Score: {applicant['CreditScore']}

    Evaluate and return the structured response only.
    """
    return prompt

# --- Health Check Endpoint ---
@app.get("/ping")
async def ping():
    """Health check endpoint to confirm the server is running."""
    return {"status": "ok"}

# --- Unified Inference Endpoint ---
@app.post("/invocations")
async def invocations(request: Request):
    """
    Unified endpoint to handle both CSV and JSON for bulk predictions and recommendations.
    """
    content_type = request.headers.get("Content-Type", "application/octet-stream")
    
    # 1. Parse Input Data
    if content_type == "application/json":
        try:
            applicant_list = await request.json()
            if not isinstance(applicant_list, list):
                raise ValueError("JSON input must be a list of applicant objects.")
            features_df = pd.DataFrame(applicant_list)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
            
    elif content_type == "text/csv":
        try:
            data = await request.body()
            from io import StringIO
            features_df = pd.read_csv(StringIO(data.decode("utf-8")))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")

    # 2. Preprocess Data for Prediction
    features_encoded = pd.get_dummies(features_df)
    features_aligned = features_encoded.reindex(columns=model_columns, fill_value=0)

    # 3. Predict Credit Scores in Bulk
    predictions = model.predict(features_aligned)
    features_df['CreditScore'] = [int(p) for p in predictions]

    # 4. Get Recommendations from Gemini in Bulk
    results = []
    applicant_data_list = features_df.to_dict(orient='records')

    for applicant_data in applicant_data_list:
        if gemini_model:
            try:
                prompt = generate_gemini_prompt(applicant_data)
                response = gemini_model.generate_content(prompt)
                recommendation_text = response.text
                result = {
                    "credit_score": applicant_data['CreditScore'],
                    "recommendation": recommendation_text
                }
            except Exception as e:
                error_message = f"Failed to get recommendation from Gemini API: {e}"
                print(error_message)
                result = {
                    "credit_score": applicant_data['CreditScore'],
                    "error": error_message
                }
        else:
            result = {
                "credit_score": applicant_data['CreditScore'], 
                "error": "Gemini API client not configured. Check GEMINI_API_KEY."
            }
        results.append(result)

    # 5. Return Combined Results
    return JSONResponse(content={"results": results})

