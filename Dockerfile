FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (required for OpenCV and PyTorch)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY backend/requirements.txt backend_requirements.txt
# Install Python dependencies (Force CPU-only PyTorch to prevent 3GB+ CUDA downloads)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt -r backend_requirements.txt

# Copy source code and models
COPY . .

# Expose port
EXPOSE 8001

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
