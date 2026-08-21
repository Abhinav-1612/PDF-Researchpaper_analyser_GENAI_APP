FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OCR and PyMuPDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for HuggingFace Spaces (7860)
EXPOSE 7860

# Default command runs the FastAPI server on 7860
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
