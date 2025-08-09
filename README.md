# Credit Confidence Score Engine

An intelligent credit scoring system that combines machine learning predictions with AI-powered creditworthiness assessment to provide comprehensive credit evaluations for loan applications.

## 🚀 Overview

The Credit Confidence Score Engine is a sophisticated financial assessment tool that:

- **Combines ML & AI**: Uses a trained machine learning model alongside Google's Gemini AI for comprehensive credit evaluation
- **Real-time API**: Provides instant credit assessments through a FastAPI endpoint
- **Comprehensive Analysis**: Evaluates multiple factors including payment history, debt management, savings behavior, income stability, and account management
- **Risk-based Scoring**: Generates detailed confidence scores (0-100) with risk categorization
- **Loan-specific Assessment**: Factors in specific loan details like amount, term, and monthly payments

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Applicant     │───▶│    ML Model      │───▶│   Gemini AI     │
│     Data        │    │  (Credit Score)  │    │  (Confidence    │
└─────────────────┘    └──────────────────┘    │   Assessment)   │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Structured     │
                                               │  Credit Report  │
                                               └─────────────────┘
```

## 📊 Features

### Core Components

1. **Machine Learning Pipeline**
   - Pre-trained model (`recommender_model.joblib`)
   - Handles 25+ features including demographics, financial behavior, and loan details
   - Generates baseline credit scores

2. **AI-Powered Assessment**
   - Google Gemini 2.5 Pro integration
   - Comprehensive system prompt with detailed scoring guidelines
   - Structured output format for consistent evaluations

3. **FastAPI Web Service**
   - RESTful API endpoint (`/predict`)
   - Pydantic data validation
   - Real-time credit assessments

### Scoring Categories

The system evaluates creditworthiness across five key areas:

| Category | Weight | Focus Areas |
|----------|--------|-------------|
| **Payment History** | 30% | Payment patterns, defaults, late payments |
| **Debt Management** | 25% | Debt-to-income ratio, loan affordability |
| **Savings Behavior** | 20% | Savings rate, emergency funds, financial reserves |
| **Income Stability** | 15% | Employment history, income consistency |
| **Account Management** | 10% | Banking relationship, account maintenance |

### Risk Assessment

- **LOW RISK (70-100)**: Excellent to good creditworthiness
- **MEDIUM RISK (40-69)**: Fair creditworthiness
- **HIGH RISK (0-39)**: Poor to very poor creditworthiness

## 🛠️ Installation

### Prerequisites

```bash
pip install flask
pip install -q -U google-genai
pip install pandas
pip install joblib
pip install uvicorn
pip install fastapi
pip install pydantic
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Bankwise-Insights/credit_confidence_score_engine.git
   cd credit_confidence_score_engine
   ```

2. **Configure API Keys**
   - Set up your Google Gemini API key in the notebook
   - Update the `GEMINI_API_KEY` variable

3. **Verify Model File**
   - Ensure `recommender_model.joblib` is in the root directory

## 🚀 Usage

### Starting the API Server

```python
import uvicorn
import threading

def run_server():
    uvicorn.run("__main__:app", host="127.0.0.1", port=5000, reload=False)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
```

### API Endpoint

**POST** `/predict`

**Request Body:**
```json
{
  "Age": 30,
  "Income": 50000.0,
  "MonthsEmployed": 24,
  "DTIRatio": 0.3,
  "Education": "Undergraduate",
  "EmploymentType": "Salaried",
  "MaritalStatus": "Single",
  "HasMortgage": 0,
  "HasDependents": 0,
  "HasCoSigner": 0,
  "AvgMonthlyBalance": 3500.0,
  "AvgMonthlySavings": 500.0,
  "NumOverdraftsLast12Months": 1,
  "SavingsRate": 0.12,
  "DepositFrequency": 2,
  "LastMonthSpending": 2700.0,
  "MinBalanceLast6Months": 1800.0,
  "MaxBalanceLast6Months": 4200.0,
  "AccountFlags": ["No recent overdraft", "Active savings"],
  "LoanPurpose": "Debt Consolidation",
  "LoanAmount": 15000.0,
  "LoanTermMonths": 36,
  "InterestRate": 12.5,
  "MonthlyPayment": 502.84,
  "TotalRepaymentAmount": 18102.24
}
```

**Response:**
```json
{
  "credit_score": "**🎯 Credit Confidence Score**\n**73/100**\n**LOW RISK**\n\n**Payment History (30%)**\n**22/30** - Good payment reliability\n\n**Debt Management (25%)**\n**18/25** - Manageable debt levels\n\n**Savings Behavior (20%)**\n**15/20** - Consistent savings pattern\n\n**Income Stability (15%)**\n**11/15** - Stable employment\n\n**Account Management (10%)**\n**7/10** - Good banking relationship\n\n**Risk Assessment:** Demonstrates solid financial discipline with manageable debt and consistent savings behavior."
}
```

## 📈 Test Scenarios

The system includes comprehensive test scenarios covering:

- **High Risk Young Borrower**: Limited income, high DTI ratio
- **Low Risk Professional**: Stable income, excellent financial management
- **Medium Risk Graduate**: Recent employment, moderate debt
- **Senior Fixed Income**: Retirement income considerations
- **Ultra Low Risk**: High income, premium banking relationship
- **Gig Worker**: Variable income patterns
- **Credit Rebuilding**: Previous financial difficulties
- **First-time Borrower**: Limited credit history
- **Emergency Borrower**: High-stress financial situation
- **Business Owner**: Variable self-employment income

## 🔧 Technical Details

### Input Features

**Personal Information:**
- Age, Income, Employment details
- Education level, Marital status
- Dependents, Mortgage status

**Financial Behavior:**
- Monthly balance patterns
- Savings rate and frequency
- Overdraft history
- Spending patterns

**Loan Specifics:**
- Loan amount and purpose
- Interest rate and term
- Monthly payment calculations

### ML Model Integration

The system uses a pre-trained machine learning model that:
- Processes 25+ features
- Generates numerical credit scores
- Provides baseline risk assessment
- Integrates with AI evaluation (40% ML score + 60% feature analysis)

### AI Assessment Engine

Google Gemini 2.5 Pro provides:
- Structured credit evaluations
- Category-specific scoring
- Risk level determination
- Professional assessment language
- Consistent output formatting

## 🔒 Security & Compliance

- Input validation through Pydantic models
- API key management for Gemini integration
- Structured data handling
- Error handling and logging

## 📝 Output Format

The system generates structured assessments including:
- Overall confidence score (0-100)
- Risk level classification
- Category-specific scores and explanations
- Professional risk assessment summary

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏢 About Bankwise Insights

Developed by Bankwise Insights for intelligent financial decision-making and risk assessment.

---

*For technical support or questions, please open an issue in the GitHub repository.*