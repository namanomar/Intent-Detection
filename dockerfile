# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories and set permissions
RUN mkdir -p /app/train/distilbert_model_trained /app/.cache \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy the application code
COPY --chown=appuser:appuser . .

# Download the DistilBERT tokenizer
RUN python -c "from transformers import DistilBertTokenizer; DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='/app/.cache')"

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
