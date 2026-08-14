# Use lightweight official Python image
FROM python:3.10-slim

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained models
COPY app/ ./app/
COPY models/ ./models/

# Expose port (Cloud Run sets PORT env var dynamically)
ENV PORT=8080

# Run Uvicorn using 0.0.0.0 to accept traffic routed by Cloud Run
CMD ["python", "app/main.py"]