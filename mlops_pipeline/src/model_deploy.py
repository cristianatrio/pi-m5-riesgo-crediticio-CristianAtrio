"""
API REST del modelo de riesgo crediticio (PI M5) con FastAPI.

Expone el pipeline serializado ``mlops_pipeline/models/modelo_riesgo.joblib`` (preprocesador
+ modelo ganador). La API recibe las 20 variables crudas del solicitante; toda la limpieza,
imputacion y feature engineering ocurre dentro del pipeline.

Endpoints (los marcados con * exigen el header ``X-API-Key``):
- GET  /health           estado del servicio y del modelo (publico: lo usa el HEALTHCHECK de Docker)
- GET  /model/info     * metricas, umbral, features y columnas requeridas
- POST /predict        * un solicitante -> probabilidad de mora, clase, nivel y decision
- POST /predict/batch  * hasta 1.000 solicitantes -> predicciones + resumen
- GET  /docs             Swagger UI (boton "Authorize" para cargar la clave)

Autenticacion: la clave valida se lee de la variable de entorno ``API_KEY``. Si el servidor no
la tiene configurada, los endpoints protegidos responden 503 (falla cerrada: nunca quedan
abiertos por olvido de configuracion).

Uso local::

    set API_KEY=mi-clave-local        (PowerShell: $env:API_KEY = "mi-clave-local")
    uvicorn model_deploy:app --app-dir mlops_pipeline/src --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import COLUMNAS_REQUERIDAS, MODELS_DIR, RAIZ, validar_esquema  # noqa: E402

VERSION_API = "1.5.0"
API_KEY_ENV = "API_KEY"
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description="Clave de acceso. El servidor la lee de la variable de entorno API_KEY.")
MAX_BATCH = 1000
MODEL_PATH = MODELS_DIR / "modelo_riesgo.joblib"
METRICS_PATH = RAIZ / "mlops_pipeline" / "reports" / "metrics.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("model_deploy")

ESTADO: dict = {}  # modelo, umbral, metricas; se llena en el lifespan


# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------
EJEMPLO_SOLICITANTE = {
    "tipo_credito": 4, "capital_prestado": 1921920.0, "plazo_meses": 10, "edad_cliente": 42,
    "tipo_laboral": "Empleado", "salario_cliente": 3000000, "total_otros_prestamos": 1000000,
    "cuota_pactada": 182863, "puntaje_datacredito": 791, "cant_creditosvigentes": 5, "huella_consulta": 4,
    "saldo_mora": 0, "saldo_total": 16178, "saldo_principal": 14442, "saldo_mora_codeudor": 0,
    "creditos_sectorFinanciero": 2, "creditos_sectorCooperativo": 0, "creditos_sectorReal": 1,
    "promedio_ingresos_datacredito": 1204496, "tendencia_ingresos": "Creciente",
}


class Solicitante(BaseModel):
    """Las 20 variables crudas que recibe el pipeline (mismos nombres que Base_de_datos.csv)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EJEMPLO_SOLICITANTE]})

    tipo_credito: int = Field(ge=0, description="Codigo del producto de credito (4 y 9 son los habituales)")
    capital_prestado: float = Field(gt=0, description="Monto solicitado")
    plazo_meses: int = Field(ge=1, le=120, description="Plazo en meses")
    edad_cliente: int = Field(ge=18, le=99, description="Edad del solicitante")
    tipo_laboral: Literal["Empleado", "Independiente"]
    salario_cliente: float = Field(ge=0, description="Ingreso mensual declarado (0 = no informado)")
    total_otros_prestamos: float = Field(ge=0, description="Deuda declarada en otros prestamos")
    cuota_pactada: float = Field(gt=0, description="Cuota mensual del credito")
    puntaje_datacredito: float | None = Field(default=None, description="Score de la central (150-950); fuera de rango o None se imputa")
    cant_creditosvigentes: int = Field(ge=0)
    huella_consulta: int = Field(ge=0, description="Consultas recientes a la central")
    saldo_mora: float | None = Field(default=None, ge=0, description="None = sin dato en la central (se imputa 0)")
    saldo_total: float | None = Field(default=None, ge=0)
    saldo_principal: float | None = Field(default=None, ge=0)
    saldo_mora_codeudor: float | None = Field(default=None, ge=0)
    creditos_sectorFinanciero: int = Field(ge=0)
    creditos_sectorCooperativo: int = Field(ge=0)
    creditos_sectorReal: int = Field(ge=0)
    promedio_ingresos_datacredito: float | None = Field(default=None, ge=0, description="None = la central no tiene estimacion")
    tendencia_ingresos: Literal["Creciente", "Estable", "Decreciente"] | None = Field(default=None, description="None = sin dato")


class Prediccion(BaseModel):
    probabilidad_mora: float = Field(description="Score de riesgo del modelo (entrenado con clases balanceadas)")
    clase: int = Field(description="1 = mora (probabilidad >= umbral), 0 = paga")
    nivel: Literal["bajo", "medio", "alto"]
    decision: Literal["aprobar", "revisar", "rechazar"]
    umbral: float


class LoteSolicitantes(BaseModel):
    solicitantes: list[Solicitante] = Field(min_length=1, max_length=MAX_BATCH)


class RespuestaLote(BaseModel):
    n: int
    n_riesgo: int
    tasa_riesgo: float
    probabilidad_media: float
    predicciones: list[Prediccion]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    modelo: str | None
    version_api: str
    umbral: float | None


class InfoModelo(BaseModel):
    modelo: str
    version_api: str
    umbral_decision: float
    criterio_seleccion: str
    metricas_test: dict[str, float]
    metricas_cv: dict[str, dict[str, float]]
    n_train: int
    tasa_mora_train: float
    columnas_requeridas: list[str]
    features_modelo: list[str]
    top_features: list[str]


# ---------------------------------------------------------------------------
# Ciclo de vida y logica
# ---------------------------------------------------------------------------
def cargar_artefactos() -> None:
    """Carga modelo y metricas una sola vez. Si falla, la API arranca en modo degradado (503)."""
    try:
        ESTADO["modelo"] = joblib.load(MODEL_PATH)
        ESTADO["metricas"] = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        ESTADO["umbral"] = float(ESTADO["metricas"]["umbral_decision"])
        logger.info("Modelo cargado: %s | umbral %.4f", ESTADO["metricas"]["modelo_ganador"], ESTADO["umbral"])
    except Exception:  # noqa: BLE001 - se registra la traza y se sirve 503
        logger.exception("No se pudieron cargar los artefactos desde %s", MODELS_DIR)
        ESTADO.clear()


@asynccontextmanager
async def lifespan(_: FastAPI):
    cargar_artefactos()
    yield
    ESTADO.clear()


app = FastAPI(
    title="API de riesgo crediticio - PI M5",
    version=VERSION_API,
    description="Probabilidad de mora de un solicitante de credito. Pipeline de feature engineering embebido; "
                "recibe datos crudos. Los endpoints de prediccion e informacion del modelo exigen el header X-API-Key.",
    lifespan=lifespan,
)


def verificar_api_key(clave: Annotated[str | None, Security(API_KEY_HEADER)] = None) -> None:
    """Compara la clave recibida con la del servidor en tiempo constante (evita ataques de timing)."""
    esperada = os.environ.get(API_KEY_ENV)
    if not esperada:
        logger.error("Variable %s sin configurar: se rechazan las solicitudes protegidas", API_KEY_ENV)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Servicio sin API key configurada")
    if not clave or not secrets.compare_digest(clave.encode(), esperada.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key invalida o ausente",
                            headers={"WWW-Authenticate": "API-Key"})


PROTEGIDO = [Depends(verificar_api_key)]


def modelo_o_503():
    if "modelo" not in ESTADO:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modelo no disponible; revisar logs del servicio")
    return ESTADO["modelo"]


def clasificar(proba: float, umbral: float) -> tuple[int, str, str]:
    if proba >= umbral:
        return 1, "alto", "rechazar"
    if proba >= umbral * 0.6:
        return 0, "medio", "revisar"
    return 0, "bajo", "aprobar"


def predecir_df(df: pd.DataFrame) -> np.ndarray:
    modelo = modelo_o_503()
    try:
        validar_esquema(df)
        return modelo.predict_proba(df[COLUMNAS_REQUERIDAS])[:, 1]
    except (ValueError, TypeError) as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Datos invalidos para el pipeline: {e}") from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Fallo de inferencia")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error interno al calcular la prediccion") from e


def a_prediccion(proba: float) -> Prediccion:
    umbral = ESTADO["umbral"]
    clase, nivel, decision = clasificar(proba, umbral)
    return Prediccion(probabilidad_mora=round(float(proba), 4), clase=clase, nivel=nivel, decision=decision, umbral=umbral)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=Health, tags=["servicio"])
def health():
    if "modelo" not in ESTADO:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modelo no cargado")
    return Health(status="ok", modelo=ESTADO["metricas"]["modelo_ganador"], version_api=VERSION_API, umbral=ESTADO["umbral"])


@app.get("/model/info", response_model=InfoModelo, tags=["servicio"], dependencies=PROTEGIDO)
def model_info():
    modelo_o_503()
    m = ESTADO["metricas"]
    return InfoModelo(
        modelo=m["modelo_ganador"], version_api=VERSION_API, umbral_decision=ESTADO["umbral"],
        criterio_seleccion=m["criterio_seleccion"], metricas_test=m["test"], metricas_cv=m["cv"],
        n_train=m["n_train"], tasa_mora_train=m["tasa_mora_train"], columnas_requeridas=list(COLUMNAS_REQUERIDAS),
        features_modelo=m["features_modelo"], top_features=m["top_features"],
    )


@app.post("/predict", response_model=Prediccion, tags=["prediccion"], dependencies=PROTEGIDO)
def predict(solicitante: Solicitante):
    df = pd.DataFrame([solicitante.model_dump()])
    return a_prediccion(predecir_df(df)[0])


@app.post("/predict/batch", response_model=RespuestaLote, tags=["prediccion"], dependencies=PROTEGIDO)
def predict_batch(lote: LoteSolicitantes):
    df = pd.DataFrame([s.model_dump() for s in lote.solicitantes])
    probas = predecir_df(df)
    preds = [a_prediccion(p) for p in probas]
    n_riesgo = sum(p.clase for p in preds)
    return RespuestaLote(
        n=len(preds), n_riesgo=n_riesgo, tasa_riesgo=round(n_riesgo / len(preds), 4),
        probabilidad_media=round(float(probas.mean()), 4), predicciones=preds,
    )


@app.get("/", include_in_schema=False)
def raiz():
    return {"servicio": app.title, "version": VERSION_API, "docs": "/docs", "health": "/health"}
