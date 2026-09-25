# Quickstart

## 1. Install

```bash
uv sync
```

## 2. Run inference

```bash
uv run inference
```

## 3. Start the API

```bash
uv run server --host 0.0.0.0 --port 7860
```

## 4. Validate

```bash
python scripts/validate_openenv.py
python -m pytest -q
```

These commands provide a short local path from a clean checkout to a validated running environment.