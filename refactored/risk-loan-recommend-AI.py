import os 
import google.generativeai as genai
from dotenv import load_dotenv 
import pandas as pd   

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash') 

# Sample input data from your training set
applicant0 = {
    "Age": 35,
    "Income": 85000,
    "MonthsEmployed": 60,
    "DTIRatio": 0.28,
    "Education": "Bachelors",
    "EmploymentType": "Full-Time",
    "MaritalStatus": "Married",
    "HasMortgage": True,
    "HasDependents": True,
    "LoanPurpose": "Home Improvement",
    "HasCosigner": False,
    "CreditScore": 720,
    "Industry": "Tech",
    "EconomicEnvironment": "Stable",
    "PersonalCircumstances": "No recent adverse events"
}


applicant1 = {
    "Age": 29,
    "Income": 45000,
    "MonthsEmployed": 24,
    "DTIRatio": 0.35,
    "Education": "High School",
    "EmploymentType": "Part-Time",
    "MaritalStatus": "Single",
    "HasMortgage": False,
    "HasDependents": False,
    "LoanPurpose": "Debt Consolidation",
    "HasCosigner": True,
    "CreditScore": 640,
}

applicant2 = {
    "Age": 42,
    "Income": 120000,
    "MonthsEmployed": 110,
    "DTIRatio": 0.22,
    "Education": "Masters",
    "EmploymentType": "Full-Time",
    "MaritalStatus": "Married",
    "HasMortgage": True,
    "HasDependents": True,
    "LoanPurpose": "Home Improvement",
    "HasCosigner": False,
    "CreditScore": 780,
}

applicant3 = {
    "Age": 37,
    "Income": 67000,
    "MonthsEmployed": 96,
    "DTIRatio": 0.31,
    "Education": "Bachelors",
    "EmploymentType": "Self-Employed",
    "MaritalStatus": "Divorced",
    "HasMortgage": True,
    "HasDependents": True,
    "LoanPurpose": "Business",
    "HasCosigner": False,
    "CreditScore": 690,
}

applicant4 = {
    "Age": 50,
    "Income": 95000,
    "MonthsEmployed": 40,
    "DTIRatio": 0.18,
    "Education": "PhD",
    "EmploymentType": "Full-Time",
    "MaritalStatus": "Married",
    "HasMortgage": True,
    "HasDependents": False,
    "LoanPurpose": "Medical",
    "HasCosigner": False,
    "CreditScore": 810,
}

applicant5 = {
    "Age": 33,
    "Income": 58000,
    "MonthsEmployed": 72,
    "DTIRatio": 0.40,
    "Education": "Associates",
    "EmploymentType": "Contract",
    "MaritalStatus": "Single",
    "HasMortgage": False,
    "HasDependents": True,
    "LoanPurpose": "Car Purchase",
    "HasCosigner": True,
    "CreditScore": 670,
}

def generate_gemini_prompt(applicant):
    prompt = f"""
You are an expert AI Loan Advisor. Based on the following applicant profile, evaluate the 5 C's of credit using the style shown below. 
Each of the 5 C's should receive a score out of 10, followed by 3 bullet points explaining the rationale based on the given info of the applicant.
Make a final Loan Recommendation as either 'YES' or 'NO'.

The response MUST follow this structured style:
⭐ The 5 C's of Credit Evaluation

CHARACTER X/10
- Repayment History: ...
- Financial Behavior: ...
- Trustworthiness: ...
- Red Flags: ...

CAPACITY X/10
- Income-to-Expense Ratio: ...
- Available Surplus: ...
- Debt-to-Income Post-Loan: ...
- Payment-to-Surplus Ratio: ...

CAPITAL X/10
- Savings Rate: ...
- Estimated Monthly Savings: ...
- Financial Cushion: ...
- Net Worth: ...

COLLATERAL X/10 or N/A
- Type: ...
- Security: ...
- Risk Mitigation: ...

CONDITIONS X/10
- Economic Environment: ...
- Employment Status: ...
- Personal Circumstances: ...
- Industry Risk: ...

Overall 5 C's Assessment: [Brief 1–2 line assessment of strengths/weaknesses]
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
- Has Cosigner: {applicant['HasCosigner']}
- Credit Score: {applicant['CreditScore']}
- Industry: {applicant.get('Industry', 'Not specified')}
- Economic Environment: {applicant.get('EconomicEnvironment', 'Not specified')}
- Personal Circumstances: {applicant.get('PersonalCircumstances', 'Not specified')}

Evaluate and return the structured response only.
"""
    return prompt


def get_loan_recommendation(applicant):
    prompt = generate_gemini_prompt(applicant)
    response = model.generate_content(prompt)
    return response.text

# Output
result0 = get_loan_recommendation(applicant0)
print(result0)


result1 = get_loan_recommendation(applicant1)
print(result1)

result2 = get_loan_recommendation(applicant2)
print(result2)

result3 = get_loan_recommendation(applicant3)
print(result3)

result4 = get_loan_recommendation(applicant4)
print(result4)

result5 = get_loan_recommendation(applicant5)
print(result5)


