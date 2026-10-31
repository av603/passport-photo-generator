FROM python:3.10-slim

# Install modern system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application script
COPY passport_photo_processor.py /app/passport_photo_processor.py

# Default command expecting arguments
ENTRYPOINT ["python", "passport_photo_processor.py"]
