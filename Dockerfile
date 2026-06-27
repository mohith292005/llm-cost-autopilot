FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/        ./src/
COPY config/     ./config/
COPY main.py     .
COPY dashboard.py .
COPY classifier.joblib .

# Create directory for SQLite database
RUN mkdir -p /data

# Expose ports
EXPOSE 8000
EXPOSE 8501

ENV DATABASE_URL=sqlite+aiosqlite:////data/autopilot.db
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.router:app", "--host", "0.0.0.0", "--port", "8000"]