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

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command runs the FastAPI server
# To run Streamlit instead, override command: streamlit run app.py
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
