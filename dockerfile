# Use official PyTorch image (includes CUDA and essential AI packages)
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Install system dependencies required for librosa & audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first (to leverage Docker layer caching)
COPY requirements.txt .

# Install Python dependencies efficiently
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Expose Flask port
EXPOSE 5000

# Use Gunicorn for better production performance
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=4", "app:app"]
