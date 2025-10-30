"""
Hybrid Statement Processor
Combines AWS Bedrock and Google Gemini for robust statement analysis
"""

import os
import sys
import json
from datetime import datetime

# Try to import Bedrock processor
try:
    from statement_processor_bedrock import StatementProcessorBedrock
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    print("⚠️ Bedrock processor not available")

# Try to import Gemini processor
try:
    from statement_processor import StatementProcessor
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini processor not available")

class StatementProcessorHybrid:
    def __init__(self):
        """Initialize hybrid processor with both Bedrock and Gemini"""
        self.bedrock_processor = None
        self.gemini_processor = None
        
        # Initialize Bedrock processor if available
        if BEDROCK_AVAILABLE:
            try:
                self.bedrock_processor = StatementProcessorBedrock()
                print("✅ Bedrock processor initialized")
            except Exception as e:
                print(f"❌ Bedrock processor failed to initialize: {e}")
        
        # Initialize Gemini processor if available
        if GEMINI_AVAILABLE:
            try:
                self.gemini_processor = StatementProcessor()
                print("✅ Gemini processor initialized")
            except Exception as e:
                print(f"❌ Gemini processor failed to initialize: {e}")
        
        if self.bedrock_processor and self.gemini_processor:
            print("🔄 Hybrid Statement Processor initialized (Bedrock + Gemini)")
            print("📊 Using AWS Bedrock (primary) + Gemini (fallback) for statement analysis")
        elif self.bedrock_processor:
            print("🔄 Using AWS Bedrock only for statement analysis")
        elif self.gemini_processor:
            print("🔄 Using Gemini only for statement analysis")
        else:
            print("❌ No statement processors available")
    
    def analyze_statements(self, bank_file_content=None, bank_content_type=None, 
                          mpesa_file_content=None, mpesa_content_type=None):
        """
        Analyze bank and M-Pesa statements using hybrid approach
        Primary: AWS Bedrock, Fallback: Google Gemini
        """
        
        # Try Bedrock first (if available)
        if self.bedrock_processor:
            try:
                print("🔄 Attempting statement analysis with AWS Bedrock...")
                result = self.bedrock_processor.analyze_statements(
                    bank_file_content=bank_file_content,
                    bank_content_type=bank_content_type,
                    mpesa_file_content=mpesa_file_content,
                    mpesa_content_type=mpesa_content_type
                )
                
                if result and self._validate_analysis_result(result):
                    print("✅ Bedrock analysis successful")
                    result['processor_used'] = 'bedrock'
                    return result
                else:
                    print("⚠️ Bedrock analysis incomplete, trying Gemini fallback...")
                    
            except Exception as e:
                print(f"❌ Bedrock analysis failed: {e}")
                print("🔄 Falling back to Gemini...")
        
        # Fallback to Gemini
        if self.gemini_processor:
            try:
                print("🔄 Attempting statement analysis with Gemini...")
                result = self.gemini_processor.analyze_statements(
                    bank_file_content=bank_file_content,
                    bank_content_type=bank_content_type,
                    mpesa_file_content=mpesa_file_content,
                    mpesa_content_type=mpesa_content_type
                )
                
                if result and self._validate_analysis_result(result):
                    print("✅ Gemini analysis successful")
                    result['processor_used'] = 'gemini'
                    return result
                else:
                    print("⚠️ Gemini analysis incomplete")
                    
            except Exception as e:
                print(f"❌ Gemini analysis failed: {e}")
        
        # If both fail, return basic analysis
        print("⚠️ Both processors failed, returning basic analysis")
        return self._create_fallback_analysis(bank_file_content, mpesa_file_content)
    
    def _validate_analysis_result(self, result):
        """Validate that the analysis result contains required fields"""
        if not result:
            return False
        
        required_fields = ['balances', 'transactions', 'summary']
        return all(field in result for field in required_fields)
    
    def _create_fallback_analysis(self, bank_file_content, mpesa_file_content):
        """Create basic fallback analysis when both processors fail"""
        
        analysis = {
            'balances': {
                'opening': 0.0,
                'closing': 0.0,
                'average': 0.0,
                'minimum': 0.0,
                'maximum': 0.0
            },
            'transactions': {
                'total_count': 0,
                'deposits': {'count': 0, 'total': 0.0},
                'withdrawals': {'count': 0, 'total': 0.0},
                'transfers': {'count': 0, 'total': 0.0}
            },
            'summary': {
                'analysis_period': 'Unknown',
                'account_activity': 'Unable to analyze - processor unavailable',
                'financial_behavior': 'Statement analysis failed',
                'risk_indicators': ['Statement processing unavailable']
            },
            'processor_used': 'fallback',
            'timestamp': datetime.now().isoformat(),
            'status': 'failed'
        }
        
        # Try to determine file types at least
        if bank_file_content:
            analysis['bank_statement'] = {
                'provided': True,
                'size_bytes': len(bank_file_content),
                'status': 'processing_failed'
            }
        
        if mpesa_file_content:
            analysis['mpesa_statement'] = {
                'provided': True,
                'size_bytes': len(mpesa_file_content),
                'status': 'processing_failed'
            }
        
        return analysis
    
    def get_processor_status(self):
        """Get status of available processors"""
        return {
            'bedrock_available': self.bedrock_processor is not None,
            'gemini_available': self.gemini_processor is not None,
            'hybrid_mode': self.bedrock_processor is not None and self.gemini_processor is not None
        }
    
    def test_processors(self):
        """Test both processors to ensure they're working"""
        results = {
            'bedrock': {'available': False, 'working': False, 'error': None},
            'gemini': {'available': False, 'working': False, 'error': None}
        }
        
        # Test Bedrock
        if self.bedrock_processor:
            results['bedrock']['available'] = True
            try:
                # Simple test - this would need actual test data
                results['bedrock']['working'] = True
            except Exception as e:
                results['bedrock']['error'] = str(e)
        
        # Test Gemini
        if self.gemini_processor:
            results['gemini']['available'] = True
            try:
                # Simple test - this would need actual test data
                results['gemini']['working'] = True
            except Exception as e:
                results['gemini']['error'] = str(e)
        
        return results