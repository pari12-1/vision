# 1️⃣ Base Python image
FROM python:3.11-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Install system dependencies (for torch & PIL)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Copy requirements first (cache optimization)
COPY requirements.txt .

# 5️⃣ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copy backend code
COPY . .

# 7️⃣ Expose Railway port
EXPOSE 8000

# 8️⃣ Start FastAPI app
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port $PORT"]


