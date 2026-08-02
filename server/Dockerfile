FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer between builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the recipe knowledge base.
COPY app.py rag.py ./
COPY data ./data

EXPOSE 8000

# DEEPSEEK_API_KEY must be supplied at runtime (see docker-compose.yml / README).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
