# AWS Bedrock Setup for Statement Analysis

## Overview
This document outlines the setup process for integrating AWS Bedrock with the Credit Confidence Score Engine for enhanced statement analysis capabilities.

## Prerequisites
- AWS Account with Bedrock access
- Python 3.8+
- boto3 library

## Installation

```bash
pip install boto3
```

## Configuration

### 1. AWS Credentials
Set up your AWS credentials in `refactored/.env`:

```env
AWS_ACCESS_KEY_ID=<your_aws_access_key>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
AWS_REGION=us-east-1
```

### 2. Bedrock Model Access
Ensure you have access to the required Bedrock models:
- Claude 3 Sonnet
- Claude 3.5 Sonnet

## Implementation
The Bedrock integration is implemented in `dashboard/backend/statement_processor_bedrock.py` and provides:
- Enhanced statement analysis
- Better transaction categorization
- Improved balance extraction
- Fallback to Gemini if Bedrock fails

## Testing
Run the test suite to verify Bedrock integration:

```bash
python test_bedrock.py
```

## Troubleshooting
- Verify AWS credentials are correct
- Check Bedrock model availability in your region
- Ensure proper IAM permissions for Bedrock access