# HuggingFace Spaces requires port 7860
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Create the root mock filesystem if it doesn't exist
# (it should be committed to the repo, but this is a safety net)
RUN mkdir -p root/network root/db root/api

# HuggingFace Spaces requires the app to listen on port 7860
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the FastAPI app (app.py serves the env via HTTP)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
