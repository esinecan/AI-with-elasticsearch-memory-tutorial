# Use Python 3.10 as the base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    llama-index==0.12.16 \
    elasticsearch \
    langchain \
    fastapi \
    uvicorn \
    requests

# Copy the application files
COPY main.py .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
