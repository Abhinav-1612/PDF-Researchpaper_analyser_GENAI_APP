FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for OCR and PyMuPDF
RUN apt-get update --fix-missing && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Render overrides this, HuggingFace uses 7860)
EXPOSE 7860

# Default command runs the FastAPI server. Uses $PORT if set (Render), else 7860 (HuggingFace)
CMD sh -c "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"
