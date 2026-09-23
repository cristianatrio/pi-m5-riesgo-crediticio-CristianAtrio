"""Pruebas de la API (model_deploy.py) y del contrato del pipeline (ft_engineering.py)."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import model_deploy
from ft_engineering import COLUMNAS_REQUERIDAS, validar_esquema
from model_deploy import EJEMPLO_SOLICITANTE, METRICS_PATH, app

CLAVE = "clave-de-prueba"
HEADERS = {"X-API-Key": CLAVE}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    """La API lee la clave valida de la variable de entorno en cada request."""
    monkeypatch.setenv(model_deploy.API_KEY_ENV, CLAVE)


@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers=HEADERS) as c:  # el lifespan carga el modelo una vez por modulo
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    ganador = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["modelo_ganador"]
    assert body["status"] == "ok"
    assert body["modelo"] == ganador
    assert 0 < body["umbral"] < 1


@pytest.mark.usefixtures("client")
def test_health_es_publico_sin_clave():
    assert TestClient(app).get("/health").status_code == 200


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["columnas_requeridas"] == list(COLUMNAS_REQUERIDAS)
    assert 0.5 < body["metricas_test"]["roc_auc"] < 1


def test_predict_ok(client):
    r = client.post("/predict", json=EJEMPLO_SOLICITANTE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probabilidad_mora"] <= 1
    assert body["clase"] in (0, 1)
    assert body["nivel"] in ("bajo", "medio", "alto")
    assert body["decision"] in ("aprobar", "revisar", "rechazar")


@pytest.mark.usefixtures("client")
@pytest.mark.parametrize("metodo, ruta, cuerpo", [
    ("GET", "/model/info", None),
    ("POST", "/predict", EJEMPLO_SOLICITANTE),
    ("POST", "/predict/batch", {"solicitantes": [EJEMPLO_SOLICITANTE]}),
])
def test_endpoints_protegidos_sin_clave_401(metodo, ruta, cuerpo):
    sin_clave = TestClient(app)  # sin "with": reutiliza el modelo ya cargado por el fixture del modulo
    r = sin_clave.request(metodo, ruta, json=cuerpo)
    assert r.status_code == 401


def test_predict_clave_incorrecta_401(client):
    r = client.post("/predict", json=EJEMPLO_SOLICITANTE, headers={"X-API-Key": "otra-clave"})
    assert r.status_code == 401


def test_servidor_sin_clave_configurada_503(client, monkeypatch):
    monkeypatch.delenv(model_deploy.API_KEY_ENV)
    r = client.post("/predict", json=EJEMPLO_SOLICITANTE)
    assert r.status_code == 503
    assert "API key" in r.json()["detail"]


def test_predict_riesgo_alto_sube_probabilidad(client):
    bueno = client.post("/predict", json=EJEMPLO_SOLICITANTE).json()["probabilidad_mora"]
    malo = {**EJEMPLO_SOLICITANTE, "puntaje_datacredito": 600, "huella_consulta": 15, "edad_cliente": 22,
            "plazo_meses": 36, "tendencia_ingresos": "Decreciente"}
    peor = client.post("/predict", json=malo).json()["probabilidad_mora"]
    assert peor > bueno


def test_predict_campo_faltante_422(client):
    datos = {k: v for k, v in EJEMPLO_SOLICITANTE.items() if k != "huella_consulta"}
    assert client.post("/predict", json=datos).status_code == 422


def test_predict_campo_extra_422(client):
    assert client.post("/predict", json={**EJEMPLO_SOLICITANTE, "puntaje": 95.2}).status_code == 422


def test_predict_tipo_laboral_invalido_422(client):
    assert client.post("/predict", json={**EJEMPLO_SOLICITANTE, "tipo_laboral": "Jubilado"}).status_code == 422


def test_predict_nulos_de_la_central_ok(client):
    datos = {**EJEMPLO_SOLICITANTE, "promedio_ingresos_datacredito": None, "tendencia_ingresos": None, "saldo_mora": None}
    assert client.post("/predict", json=datos).status_code == 200


def test_predict_batch(client):
    lote = {"solicitantes": [EJEMPLO_SOLICITANTE, {**EJEMPLO_SOLICITANTE, "edad_cliente": 25}]}
    r = client.post("/predict/batch", json=lote)
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2
    assert len(body["predicciones"]) == 2
    assert 0 <= body["tasa_riesgo"] <= 1


def test_predict_batch_vacio_422(client):
    assert client.post("/predict/batch", json={"solicitantes": []}).status_code == 422


def test_raiz_apunta_a_docs(client):
    assert client.get("/").json()["docs"] == "/docs"


def test_error_de_datos_del_pipeline_400(client, monkeypatch):
    def rechaza(_df):
        raise ValueError("columna rota")

    monkeypatch.setattr(model_deploy, "validar_esquema", rechaza)
    r = client.post("/predict", json=EJEMPLO_SOLICITANTE)
    assert r.status_code == 400
    assert "columna rota" in r.json()["detail"]


def test_fallo_de_inferencia_500_sin_traza(client, monkeypatch):
    class ModeloRoto:
        def predict_proba(self, _X):
            raise RuntimeError("detalle interno que no debe filtrarse")

    monkeypatch.setitem(model_deploy.ESTADO, "modelo", ModeloRoto())
    r = client.post("/predict", json=EJEMPLO_SOLICITANTE)
    assert r.status_code == 500
    assert "detalle interno" not in r.text


def test_modo_degradado_503_si_no_hay_modelo(monkeypatch, tmp_path):
    estado_previo = dict(model_deploy.ESTADO)  # el lifespan de este cliente vacia el estado compartido
    monkeypatch.setattr(model_deploy, "MODEL_PATH", tmp_path / "no_existe.joblib")
    try:
        with TestClient(app, headers=HEADERS) as c:
            assert c.get("/health").status_code == 503
            assert c.get("/model/info").status_code == 503
            assert c.post("/predict", json=EJEMPLO_SOLICITANTE).status_code == 503
    finally:
        model_deploy.ESTADO.update(estado_previo)


def test_validar_esquema_detecta_faltantes():
    df = pd.DataFrame([EJEMPLO_SOLICITANTE]).drop(columns=["saldo_total"])
    with pytest.raises(ValueError, match="saldo_total"):
        validar_esquema(df)
