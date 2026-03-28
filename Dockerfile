FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir openenv-core>=0.2.0 numpy pydantic typing-extensions fastapi uvicorn pyyaml

COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY server/ ./server/
COPY openenv.yaml .
COPY app.py .
COPY inference.py .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]