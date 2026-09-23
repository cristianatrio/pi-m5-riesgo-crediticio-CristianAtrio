# Imagen de la API de riesgo crediticio (FastAPI + modelo Random Forest serializado)
# build:  docker build -t riesgo-api:1.3.0 .
# run:    docker run -d --name riesgo-api -p 8000:8000 riesgo-api:1.3.0
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAIZ_PROYECTO=/app

# libgomp1: runtime OpenMP que usan scikit-learn / scipy en Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias primero: capa cacheable, no se reinstala si solo cambia el codigo.
# Lock con hashes (todas las dependencias transitivas verificadas) y solo wheels: no se ejecutan scripts de setup.
COPY requirements-api.lock .
RUN pip install --no-cache-dir --only-binary :all: --require-hashes -r requirements-api.lock

# Codigo y artefactos, solo lo necesario para inferencia
COPY mlops_pipeline/src/ft_engineering.py mlops_pipeline/src/model_deploy.py mlops_pipeline/src/config.json mlops_pipeline/src/
COPY mlops_pipeline/models/modelo_riesgo.joblib mlops_pipeline/models/feature_names.json mlops_pipeline/models/
COPY mlops_pipeline/reports/metrics.json mlops_pipeline/reports/

# Sin root en runtime
RUN useradd --create-home api && chown -R api:api /app
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "model_deploy:app", "--app-dir", "mlops_pipeline/src", "--host", "0.0.0.0", "--port", "8000"]
