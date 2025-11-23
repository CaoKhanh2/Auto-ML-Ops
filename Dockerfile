FROM python:3.12-slim

WORKDIR /app

# Copy dependency trước
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code đúng thư mục
COPY src/ ./src
COPY models/ /app/models/
COPY data/ ./data

# Ensure PYTHONPATH includes src
ENV PYTHONPATH="/app/src"

EXPOSE 8000

# Correct module path: src.api.app
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
