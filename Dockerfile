FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libmagic1 \
    file \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set workdir to the actual service folder
WORKDIR /app/GhostUsers

# Copy requirements first (for better caching)
COPY GhostUsers/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project intoimage
COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start fastAPI with uvicorn
CMD ["fastapi","run","app.py"]
