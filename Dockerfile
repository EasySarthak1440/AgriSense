FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables
ENV FLASK_APP=app/app1.py
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Run the application
CMD ["gunicorn", "--chdir", "app", "-b", "0.0.0.0:7860", "app1:app1"]
