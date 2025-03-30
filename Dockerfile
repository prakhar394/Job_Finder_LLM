FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME /app/data
VOLUME /app/results

# Add this to ensure Python can find your modules
ENV PYTHONPATH "${PYTHONPATH}:/app/src"

CMD ["python", "-m", "src.main"]
