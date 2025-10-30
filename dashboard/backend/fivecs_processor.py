"""
5Cs Credit Assessment Processor
Implements comprehensive credit evaluation using Character, Capacity, Capital, Collateral, Conditions
"""

import google.generativeai as genai
import os
from datetime import datetime

class FiveCsProcessor:
    def __init__(self):
        """Initialize 5Cs processor with Gemini AI"""
        self.setup_gemini()
        self.load_prompt()
    
    def setup_gemini(self):
        """Setup Gemini AI configuration"""
        try:
            # Load API key from environment
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                # Try loading from .env file
                env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'refactored', '.env')
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if line.startswith('GEMINI_API_KEY'):
                                api_key = line.split('=')[1].strip()
                                break
            
            if api_key and api_key != '<your_gemini_api_key>':
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                print("✅ Gemini AI configured successfully")
            else:
                print("❌ Gemini API key not found or is placeholder")
                self.model = None
                
        except Exception as e:
            print(f"❌ Error setting up Gemini: {e}")
            self.model = None
    
    def load_prompt(self):
        """Load the 5Cs evaluation prompt"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), 'gemini_5cs_prompt.txt')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
            print("✅ 5Cs prompt loaded successfully")
        except Exception as e:
            print(f"❌ Error loading 5Cs prompt: {e}")
            self.system_prompt = "Evaluate this credit application using the 5Cs framework."
    
    def evaluate_credit_application(self, applicant_data, ml_score=None, statement_analysis=None, document_analysis=None):
        """Evaluate credit application using 5Cs framework"""
        try:
            if self.model is None:
                return self.fallback_assessment(applicant_data, ml_score)
            
            # Prepare comprehensive input for Gemini
            evaluation_input = self.prepare_evaluation_input(
                applicant_data, ml_score, statement_analysis, document_analysis
            )
            
            # Generate assessment using Gemini
            response = self.model.generate_content(
                f"{self.system_prompt}\n\n{evaluation_input}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2000
                )
            )
            
            return response.text
            
        except Exception as e:
            print(f"❌ Error in 5Cs evaluation: {e}")
            return self.fallback_assessment(applicant_data, ml_score)
    
    def prepare_evaluation_input(self, applicant_data, ml_score, statement_analysis, document_analysis):
        """Prepare comprehensive input for 5Cs evaluation"""
        
        input_text = f"""
## APPLICANT PROFILE

**Personal Information:**
- Name: {applicant_data.get('full_name', 'N/A')}
- Age: {applicant_data.get('Age', 'N/A')}
- Education: {applicant_data.get('Education', 'N/A')}
- Employment: {applicant_data.get('EmploymentType', 'N/A')}
- Marital Status: {applicant_data.get('MaritalStatus', 'N/A')}
- Months Employed: {applicant_data.get('MonthsEmployed', 'N/A')}

**Financial Profile:**
- Annual Income: KES {applicant_data.get('Income', 0):,.2f}
- DTI Ratio: {applicant_data.get('DTIRatio', 0):.2%}
- Average Monthly Balance: KES {applicant_data.get('AvgMonthlyBalance', 0):,.2f}
- Average Monthly Savings: KES {applicant_data.get('AvgMonthlySavings', 0):,.2f}
- Savings Rate: {applicant_data.get('SavingsRate', 0):.2%}
- Overdrafts (Last 12 months): {applicant_data.get('NumOverdraftsLast12Months', 0)}

**Loan Details:**
- Purpose: {applicant_data.get('LoanPurpose', 'N/A')}
- Amount: KES {applicant_data.get('LoanAmount', 0):,.2f}
- Term: {applicant_data.get('LoanTermMonths', 0)} months
- Interest Rate: {applicant_data.get('InterestRate', 0):.2%}
- Monthly Payment: KES {applicant_data.get('MonthlyPayment', 0):,.2f}

**Credit Information:**
- Credit Score: {applicant_data.get('CreditScore', 'Not provided')}
"""

        # Add ML Score information
        if ml_score:
            input_text += f"""
**ML Model Prediction:**
- Predicted Credit Score: {ml_score}
- Model Assessment: ML model analyzed financial behavior patterns
"""

        # Add collateral information if provided
        if applicant_data.get('CollateralType') and applicant_data.get('CollateralValue', 0) > 0:
            ltv_ratio = applicant_data.get('LoanAmount', 0) / applicant_data.get('CollateralValue', 1)
            input_text += f"""
**Collateral Information:**
- Type: {applicant_data.get('CollateralType')}
- Value: KES {applicant_data.get('CollateralValue', 0):,.2f}
- Loan-to-Value Ratio: {ltv_ratio:.2%}
"""

        # Add co-signer information if provided
        if applicant_data.get('CoSignerName'):
            input_text += f"""
**Co-signer Information:**
- Name: {applicant_data.get('CoSignerName')}
- ID Number: {applicant_data.get('CoSignerID', 'Not provided')}
"""

        # Add statement analysis if available
        if statement_analysis:
            input_text += f"""
**Statement Analysis:**
- Banking behavior patterns analyzed
- Transaction history reviewed
- Balance trends evaluated
"""

        # Add document analysis if available
        if document_analysis:
            input_text += f"""
**Document Validation:**
- Collateral documents: {'Validated' if document_analysis.get('collateral_valid') else 'Not validated'}
- Co-signer documents: {'Validated' if document_analysis.get('cosigner_valid') else 'Not validated'}
"""

        input_text += f"""

## EVALUATION REQUEST
Please evaluate this loan application using the 5 C's of Credit framework and provide a structured assessment with specific scores for each category.
"""

        return input_text
    
    def fallback_assessment(self, applicant_data, ml_score):
        """Provide fallback assessment when Gemini is not available"""
        
        # Calculate basic scores
        character_score = min(25, (ml_score or 600) / 30) if ml_score else 20
        
        # Capacity based on DTI ratio
        dti = applicant_data.get('DTIRatio', 0)
        if dti == 0:
            capacity_score = 25
        elif dti < 0.2:
            capacity_score = 22
        elif dti < 0.3:
            capacity_score = 18
        elif dti < 0.4:
            capacity_score = 14
        else:
            capacity_score = 10
        
        # Capital based on income
        income = applicant_data.get('Income', 0)
        if income > 150000:
            capital_score = 20
        elif income > 80000:
            capital_score = 16
        elif income > 50000:
            capital_score = 12
        else:
            capital_score = 8
        
        # Collateral
        collateral_score = 10 if applicant_data.get('CollateralValue', 0) > 0 else 5
        
        # Conditions
        conditions_score = 8  # Default
        
        total_score = character_score + capacity_score + capital_score + collateral_score + conditions_score
        
        if total_score >= 70:
            risk_level = "LOW RISK"
        elif total_score >= 40:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "HIGH RISK"
        
        return f"""**🎯 Credit Confidence Score**
**{total_score:.0f}/100**
**{risk_level}**

**Character ({character_score:.0f}/30)**
{character_score:.0f}/30 - Credit assessment based on available data

**Capacity ({capacity_score:.0f}/25)**
{capacity_score:.0f}/25 - DTI ratio: {dti:.1%}, repayment capacity evaluated

**Capital ({capital_score:.0f}/20)**
{capital_score:.0f}/20 - Income level: KES {income:,.0f}

**Collateral ({collateral_score:.0f}/15)**
{collateral_score:.0f}/15 - {'Secured loan' if collateral_score > 5 else 'Unsecured loan'}

**Conditions ({conditions_score:.0f}/10)**
{conditions_score:.0f}/10 - Standard loan conditions

**Risk Assessment:** Basic assessment completed. Comprehensive AI evaluation temporarily unavailable.

**RECOMMENDATION:** {'APPROVE' if total_score >= 60 else 'CONDITIONAL APPROVAL' if total_score >= 40 else 'DECLINE'}"""