# Single container: FastAPI backend + built React frontend on port 8000.
# Works locally (docker compose up) and on any container host (Railway, Render, Fly.io, Cloud Run).

# ---- 1. build the frontend ----
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ARG VITE_MAX_UPLOAD_MB=25
ENV VITE_MAX_UPLOAD_MB=${VITE_MAX_UPLOAD_MB}
RUN npm run build

# ---- 2. runtime ----
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STATIC_DIR=/app/static \
    MAX_UPLOAD_MB=25 \
    PORT=8000
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY backend/artifacts ./artifacts
COPY --from=web /web/dist ./static

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]
