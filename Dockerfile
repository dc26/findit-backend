FROM python:3.11-slim

# Install ffmpeg for audio extraction
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Command to run the Fastapi server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
