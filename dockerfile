# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only requirements first (for better caching)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose Flask port
EXPOSE 5000

# Use Gunicorn for better performance in production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=4", "app:app"]
