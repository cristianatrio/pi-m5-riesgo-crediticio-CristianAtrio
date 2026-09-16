# Imagen de la app Streamlit (prediccion, explicacion, lote y monitoreo)
# build:  docker build -f Dockerfile.app -t riesgo-app:1.3.0 .
# run:    docker run -d --name riesgo-app -p 8501:8501 riesgo-app:1.3.0
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAIZ_PROYECTO=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# La app necesita el historico (defaults del formulario y perfil de pagadores), el modelo,
# las metricas, las importancias y los reportes de drift
COPY Base_de_datos.csv .
COPY .streamlit/ .streamlit/
COPY mlops_pipeline/src/ft_engineering.py mlops_pipeline/src/app_streamlit.py mlops_pipeline/src/config.json mlops_pipeline/src/
COPY mlops_pipeline/models/modelo_riesgo.joblib mlops_pipeline/models/feature_names.json mlops_pipeline/models/
COPY mlops_pipeline/reports/metrics.json mlops_pipeline/reports/importancia_variables.csv mlops_pipeline/reports/
COPY mlops_pipeline/reports/drift/*.json mlops_pipeline/reports/drift/

RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "mlops_pipeline/src/app_streamlit.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
