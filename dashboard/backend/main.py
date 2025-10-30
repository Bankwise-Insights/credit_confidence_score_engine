"""
Credit Confidence Score Engine - Main API Server
Comprehensive credit scoring system with ML model and AI assessment
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
import sqlite3
from datetime import datetime
import os
import sys

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import processors
from ml_processor import MLProcessor
from fivecs_processor import FiveCsProcessor
from statement_processor_hybrid import StatementProcessorHybrid
from document_processor import DocumentProcessor

app = FastAPI(title="Credit Confidence Score Engine", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
ml_processor = MLProcessor()
fivecs_processor = FiveCsProcessor()
statement_processor = StatementProcessorHybrid()
document_processor = DocumentProcessor()

# Database setup
def init_db():
    conn = sqlite3.connect('loan_applications.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            applicant_data TEXT,
            ml_score REAL,
            credit_assessment TEXT,
            statement_analysis TEXT,
            document_analysis TEXT,
            final_recommendation TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

@app.get("/")
async def root():
    return {"message": "Credit Confidence Score Engine API", "version": "2.0", "status": "active"}

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        conn = sqlite3.connect('loan_applications.db')
        cursor = conn.cursor()
        
        # Get total applications
        cursor.execute("SELECT COUNT(*) FROM applications")
        total_applications = cursor.fetchone()[0]
        
        # Get applications by status
        cursor.execute("SELECT status, COUNT(*) FROM applications GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        # Get recent applications
        cursor.execute("""
            SELECT id, timestamp, final_recommendation, status 
            FROM applications 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        recent_applications = [
            {
                "id": row[0],
                "timestamp": row[1],
                "recommendation": row[2],
                "status": row[3]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return {
            "total_applications": total_applications,
            "status_counts": status_counts,
            "recent_applications": recent_applications
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/credit-assessment")
async def process_credit_application(
    # Personal Information
    full_name: str = Form(...),
    age: int = Form(...),
    income: float = Form(...),
    months_employed: int = Form(...),
    dti_ratio: float = Form(0.0),
    education: str = Form(...),
    employment_type: str = Form(...),
    marital_status: str = Form(...),
    has_mortgage: int = Form(0),
    has_dependents: int = Form(0),
    has_cosigner: int = Form(0),
    
    # Financial Behavior
    avg_monthly_balance: float = Form(...),
    avg_monthly_savings: float = Form(...),
    num_overdrafts_last_12_months: int = Form(0),
    savings_rate: float = Form(...),
    deposit_frequency: int = Form(...),
    last_month_spending: float = Form(...),
    min_balance_last_6_months: float = Form(...),
    max_balance_last_6_months: float = Form(...),
    
    # Loan Details
    loan_purpose: str = Form(...),
    loan_amount: float = Form(...),
    loan_term_months: int = Form(...),
    interest_rate: float = Form(...),
    monthly_payment: float = Form(...),
    
    # Optional: Custom loan purpose
    custom_loan_purpose: str = Form(""),
    
    # Optional: Collateral and Co-signer
    collateral_type: str = Form(""),
    collateral_value: float = Form(0.0),
    cosigner_name: str = Form(""),
    cosigner_id: str = Form(""),
    
    # Optional: Credit Score
    credit_score: int = Form(0),
    
    # Optional: File uploads
    bank_statement: UploadFile = File(None),
    mpesa_statement: UploadFile = File(None),
    collateral_document: UploadFile = File(None),
    cosigner_document: UploadFile = File(None)
):
    """Process complete credit application with ML scoring and AI assessment"""
    
    try:
        # Prepare applicant data
        applicant_data = {
            "full_name": full_name,
            "Age": age,
            "Income": income,
            "MonthsEmployed": months_employed,
            "DTIRatio": dti_ratio,
            "Education": education,
            "EmploymentType": employment_type,
            "MaritalStatus": marital_status,
            "HasMortgage": has_mortgage,
            "HasDependents": has_dependents,
            "HasCoSigner": has_cosigner,
            "AvgMonthlyBalance": avg_monthly_balance,
            "AvgMonthlySavings": avg_monthly_savings,
            "NumOverdraftsLast12Months": num_overdrafts_last_12_months,
            "SavingsRate": savings_rate,
            "DepositFrequency": deposit_frequency,
            "LastMonthSpending": last_month_spending,
            "MinBalanceLast6Months": min_balance_last_6_months,
            "MaxBalanceLast6Months": max_balance_last_6_months,
            "LoanPurpose": custom_loan_purpose if loan_purpose == "Other-Miscellaneous" else loan_purpose,
            "LoanAmount": loan_amount,
            "LoanTermMonths": loan_term_months,
            "InterestRate": interest_rate,
            "MonthlyPayment": monthly_payment,
            "CollateralType": collateral_type,
            "CollateralValue": collateral_value,
            "CoSignerName": cosigner_name,
            "CoSignerID": cosigner_id,
            "CreditScore": credit_score
        }
        
        # Step 1: ML Credit Score Prediction
        ml_result = ml_processor.predict_credit_score(applicant_data)
        
        # Step 2: Statement Analysis (if files provided)
        statement_analysis = None
        if bank_statement or mpesa_statement:
            bank_content = await bank_statement.read() if bank_statement else None
            mpesa_content = await mpesa_statement.read() if mpesa_statement else None
            
            statement_analysis = statement_processor.analyze_statements(
                bank_file_content=bank_content,
                bank_content_type=bank_statement.content_type if bank_statement else None,
                mpesa_file_content=mpesa_content,
                mpesa_content_type=mpesa_statement.content_type if mpesa_statement else None
            )
        
        # Step 3: Document Analysis (if files provided)
        document_analysis = None
        if collateral_document or cosigner_document:
            collateral_content = await collateral_document.read() if collateral_document else None
            cosigner_content = await cosigner_document.read() if cosigner_document else None
            
            document_analysis = document_processor.analyze_documents(
                collateral_file_content=collateral_content,
                collateral_content_type=collateral_document.content_type if collateral_document else None,
                cosigner_file_content=cosigner_content,
                cosigner_content_type=cosigner_document.content_type if cosigner_document else None
            )
        
        # Step 4: 5Cs Credit Assessment
        credit_assessment = fivecs_processor.evaluate_credit_application(
            applicant_data=applicant_data,
            ml_score=ml_result.get('predicted_score'),
            statement_analysis=statement_analysis,
            document_analysis=document_analysis
        )
        
        # Step 5: Store in database
        conn = sqlite3.connect('loan_applications.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO applications 
            (timestamp, applicant_data, ml_score, credit_assessment, statement_analysis, document_analysis, final_recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(applicant_data),
            ml_result.get('predicted_score'),
            credit_assessment,
            json.dumps(statement_analysis) if statement_analysis else None,
            json.dumps(document_analysis) if document_analysis else None,
            credit_assessment  # Using credit_assessment as final recommendation
        ))
        
        application_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Return comprehensive response
        return {
            "application_id": application_id,
            "ml_prediction": ml_result,
            "credit_assessment": credit_assessment,
            "statement_analysis": statement_analysis,
            "document_analysis": document_analysis,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/api/applications/{application_id}")
async def get_application(application_id: int):
    """Get specific application details"""
    try:
        conn = sqlite3.connect('loan_applications.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM applications WHERE id = ?
        """, (application_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Application not found")
        
        conn.close()
        
        return {
            "id": row[0],
            "timestamp": row[1],
            "applicant_data": json.loads(row[2]),
            "ml_score": row[3],
            "credit_assessment": row[4],
            "statement_analysis": json.loads(row[5]) if row[5] else None,
            "document_analysis": json.loads(row[6]) if row[6] else None,
            "final_recommendation": row[7],
            "status": row[8]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Credit Confidence Score Engine API Server...")
    print("📊 Dashboard: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)