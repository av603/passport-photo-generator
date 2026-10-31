FROM python:3.10-slim

# Install modern system dependencies required by OpenCV, MediaPipe, and scale bar fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user and directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/output && \
    chown -R appuser:appuser /app

# Copy application script
COPY passport_photo_processor.py /app/passport_photo_processor.py

USER appuser

# Default command expecting arguments
ENTRYPOINT ["python", "passport_photo_processor.py"]
