End-to-End Loan Recommendation Engine with SageMaker and Gemini
This project demonstrates a complete MLOps workflow for deploying a sophisticated loan recommendation engine on AWS. The architecture combines a classical machine learning model for credit scoring with a generative AI model (Google's Gemini) for detailed, human-readable loan analysis and recommendation.

The entire system is packaged into a custom Docker container and deployed as a scalable, cost-effective SageMaker Serverless Endpoint.

Final Architecture
The final architecture is streamlined for simplicity and performance:

Client Application: A client (e.g., invoke.py) sends a batch of loan applications in JSON format to the SageMaker Endpoint.

SageMaker Serverless Endpoint: A fully managed, auto-scaling endpoint that hosts our custom Docker container.

Custom Docker Container:

Contains a FastAPI web server (predictor.py) to handle inference requests.

Loads a pre-trained scikit-learn model (model.joblib) to predict a credit score for each applicant.

Uses the predicted score and the original applicant data to construct a detailed prompt.

Calls the Gemini API to generate a comprehensive 5 C's credit analysis and a final "YES" or "NO" recommendation.

Response: The endpoint returns a single JSON object containing the list of results, with each result including the predicted credit score and the detailed recommendation from Gemini.

Local Project Structure
This project uses a structured layout to separate the container environment from the deployment logic.

```
sagemaker_training_project/
|
|-- 📂 container/
|   |-- Dockerfile.train         # Blueprint for the training & serving container
|   |-- requirements.txt       # Python dependencies (sklearn, pandas, fastapi, etc.)
|   |-- sagemaker_training.py    # Script for model training and selection
|   |-- predictor.py           # FastAPI server for inference
|   |-- train                  # Executable script to start training
|   └── serve                  # Executable script to start the FastAPI server
|
|-- 📂 data/
|   └── loans.csv              # Raw training dataset
|
└── 📜 launch_training_job.ipynb    # Jupyter Notebook to control the entire workflow
```

Step-by-Step Deployment Guide
Follow these steps to deploy the entire stack from scratch.

1. Prerequisites
AWS Account with appropriate permissions.

AWS CLI configured locally.

Docker installed and running locally.

2. IAM Permissions
The SageMaker Execution Role used by your notebook and training jobs needs permissions to write logs.

Go to the IAM service in the AWS Console.

Find your SageMaker Execution Role (e.g., SageMaker-ExecutionRole-xxxxxxxx).

Ensure it has cloudwatchfullaccess, sagemakerfullaccess, ecrfullaccess, and if necessary s3fullaccesspermissions.

3. Build and Push the Docker Container
These commands build your custom container and upload it to Amazon's Elastic Container Registry (ECR). Choose the set of commands for your operating system.

For Windows (using PowerShell):

# Set your AWS Account ID and Region
```shell
$env:ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
$env:REGION = "eu-north-1" # Change to your region
```

# Log in to ECR
```shell
aws ecr get-login-password --region $env:REGION | docker login --username AWS --password-stdin "$($env:ACCOUNT_ID).dkr.ecr.$($env:REGION).amazonaws.com"
```

# Navigate to your container directory
```shell
cd sagemaker_training_project/container
```

# Build the image
```shell
docker build -t loan-recommender:latest -f Dockerfile.train .
```

# Tag and Push
```shell
docker tag loan-recommender:latest "$($env:ACCOUNT_ID).dkr.ecr.$($env:REGION)[.amazonaws.com/loan-recommender:latest](https://.amazonaws.com/loan-recommender:latest)"
docker push "$($env:ACCOUNT_ID).dkr.ecr.$($env:REGION)[.amazonaws.com/loan-recommender:latest](https://.amazonaws.com/loan-recommender:latest)"
```

4. Upload Data to S3
Ensure your loans.csv dataset from the data/ folder is uploaded to an S3 bucket that your SageMaker role can access.

5. Launch Training & Deployment
Open and run the launch_training_job.ipynb notebook from a SageMaker Notebook Instance. This notebook will:

Define the ECR image_uri using the image you just pushed.

Define the S3 path to your loans.csv data.

Set the GEMINI_API_KEY as an environment variable for the container.

Create a SageMaker Estimator object.

Launch the training job by calling estimator.fit({'train': s3_data_path}).

Deploy the resulting model artifact to a SageMaker Serverless Endpoint by calling estimator.deploy().

How to Invoke the Endpoint
Use the invoke.py script to send requests to your deployed endpoint. The script sends a list of applicant dictionaries and prints the results.

Example invoke.py Usage:
```python

import boto3
import json

ENDPOINT_NAME = 'your-serverless-endpoint-name' 
AWS_REGION = 'eu-north-1' 

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)

# Payload must be a list of applicant dictionaries
applicant_payload = [
    { "Age": 35, "Income": 85000, ... },
    { "Age": 29, "Income": 45000, ... }
]

response = sagemaker_runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType='application/json',
    Body=json.dumps(applicant_payload)
)

# Process and print results...
```
