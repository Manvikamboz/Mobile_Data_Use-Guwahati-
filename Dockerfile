# Use official Python 3.12 slim image (Handbook Section 9)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables for clean container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy requirements and install dependencies before copying source code
# This optimizes Docker layer caching during incremental builds
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application, champion model artifact, source code, and configuration
COPY app/ ./app/
COPY models/ ./models/
COPY src/ ./src/
COPY params.yaml .

# Expose container port (PROBLEM.md Section 6 & Handbook Section 9)
EXPOSE 8000

# Start Uvicorn ASGI server binding to 0.0.0.0 on port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
