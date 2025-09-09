# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH="/opt/ml/code:${PATH}"

# Set the working directory
WORKDIR /opt/ml/code

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the necessary scripts for training and serving
COPY credit_scoring.py .
COPY predictor.py .
COPY train .
COPY serve .

# Make the train and serve scripts executable
RUN chmod +x train serve