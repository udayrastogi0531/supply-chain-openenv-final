FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY openenv.yaml .
COPY app.py .
COPY inference.py .

RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]