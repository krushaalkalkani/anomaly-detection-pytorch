FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch first, then the rest of the stack (skip the CUDA wheel).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir scikit-learn pandas numpy fastapi uvicorn

COPY src/ ./src/
COPY api/ ./api/
COPY results/autoencoder.pt ./results/autoencoder.pt

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/results/autoencoder.pt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
