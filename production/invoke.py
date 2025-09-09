import boto3
import json

# --- Configuration ---
# IMPORTANT: Replace these with your actual endpoint name and AWS region
ENDPOINT_NAME = 'loan-recommender-serverless' 
AWS_REGION = 'us-east-1' 

# Create a SageMaker runtime client
try:
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
except Exception as e:
    print(f"Error creating boto3 client: {e}")
    exit()

def invoke_bulk_recommendations():
    """
    Invokes the endpoint to get a batch of loan recommendations.
    Sends a JSON payload containing a LIST of applicant objects.
    """
    print("--- Invoking for Bulk Recommendations ---")
    
    # MODIFIED: Payload is now a list of dictionaries
    applicant_payload = [
        {
            "Age": 35, "Income": 85000, "MonthsEmployed": 60, "DTIRatio": 0.28, "Education": "Bachelors",
            "EmploymentType": "Full-Time", "MaritalStatus": "Married", "HasMortgage": True, "HasDependents": True,
            "LoanPurpose": "Home Improvement", "HasCoSigner": False, "Industry": "Tech", "EconomicEnvironment": "Stable",
            "PersonalCircumstances": "No recent adverse events"
        },
        {
            "Age": 29, "Income": 45000, "MonthsEmployed": 24, "DTIRatio": 0.35, "Education": "High School",
            "EmploymentType": "Part-Time", "MaritalStatus": "Single", "HasMortgage": False, "HasDependents": False,
            "LoanPurpose": "Debt Consolidation", "HasCoSigner": True, "Industry": "Retail", "EconomicEnvironment": "Unstable",
            "PersonalCircumstances": "Recent job change"
        }
    ]

    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(applicant_payload)
        )
        
        result_str = response['Body'].read().decode()
        response_json = json.loads(result_str)
        
        print("Successfully received batch response.\n")
        
        # MODIFIED: Loop through the 'results' list in the response
        if 'results' in response_json and isinstance(response_json['results'], list):
            for i, result in enumerate(response_json['results']):
                print(f"--- Result for Applicant {i+1} ---")
                predicted_score = result.get('predicted_credit_score', 'N/A')
                print(f"Predicted Credit Score: {predicted_score}\n")
                print(result.get('recommendation', 'No recommendation found in response.'))
                print("-" * 30 + "\n")
        else:
            print("Error: Response format is incorrect. Expected a 'results' key with a list.")
            print("Received:", response_json)


    except Exception as e:
        print(f"Error invoking endpoint for recommendation: {e}")

if __name__ == '__main__':
    if 'your-sagemaker-endpoint-name' in ENDPOINT_NAME:
        print("Please update the ENDPOINT_NAME and AWS_REGION variables in the script before running.")
    else:
        invoke_bulk_recommendations()
