# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF (fitz)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model file
COPY src/ ./src/
COPY pdf/ ./pdf/
COPY src/rf_model.pkl ./src/rf_model.pkl

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Set entrypoint to main.py, default PDF can be overridden at runtime
ENTRYPOINT ["python", "src/main.py"]
# Example: docker run -v $(pwd)/pdf:/app/pdf myimage ./pdf/sample.pdf
