# Quick Fix: "Error Parsing Body"

## Issue
Form submission shows: "Error submitting application, there was an error parsing the body"

## Solution

### Step 1: Check Backend Console
The backend console will show the ACTUAL error. Look for:
```
Error: ...
Traceback: ...
```

### Step 2: Restart Backend
```bash
# Stop current server (Ctrl+C)
cd c:\Users\NTHUMBI\.vscode\credit_confidence_score_engine\dashboard\backend
python main.py
```

### Step 3: Check Console Output
You should see:
```
Hybrid Statement Processor initialized (Bedrock + Gemini)
Using AWS Bedrock (primary) + Gemini (fallback) for statement analysis
```

If you see an error about imports, the hybrid processor isn't loading.

## Common Errors

### Error 1: "No module named 'statement_processor_hybrid'"
**Fix:** The file exists but Python can't find it.
```bash
cd c:\Users\NTHUMBI\.vscode\credit_confidence_score_engine\dashboard\backend
dir statement_processor*.py
```
Should show:
- statement_processor.py
- statement_processor_bedrock.py
- statement_processor_hybrid.py

### Error 2: "No module named 'boto3'"
**Fix:**
```bash
pip install boto3
```

### Error 3: AWS credentials error
**Fix:** Check `refactored/.env` has:
```
AWS_ACCESS_KEY_ID=<your_aws_access_key>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
AWS_REGION=us-east-1
```

## Test Without Frontend

```bash
cd c:\Users\NTHUMBI\.vscode\credit_confidence_score_engine
python test_server.py
```

Should show:
```
Server status: 200
Response: {'total_applications': 68, ...}
```

## Alternative: Use Gemini Only (Temporary)

If Bedrock is causing issues, temporarily revert to Gemini:

Edit `dashboard/backend/main.py`:
```python
# Change this line:
from statement_processor_hybrid import StatementProcessorHybrid

# Back to:
from statement_processor import StatementProcessor

# And change:
statement_processor = StatementProcessorHybrid()

# Back to:
statement_processor = StatementProcessor()
```

Then restart server.

## Next Steps

1. **Check backend console** for actual error
2. **Copy the error message** and we can fix it
3. **Try test_server.py** to verify API is working
4. **If all else fails**, use Gemini-only temporarily

## Quick Test

Open browser console (F12) and run:
```javascript
fetch('http://localhost:8000/api/dashboard/stats')
  .then(r => r.json())
  .then(d => console.log(d))
```

Should show stats. If it works, the API is fine and the issue is form-specific.