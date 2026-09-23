"""Pruebas de la API (model_deploy.py) y del contrato del pipeline (ft_engineering.py)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import model_deploy
from ft_engineering import COLUMNAS_REQUERIDAS, validar_esquema
from model_deploy import EJEMPLO_SOLICITANTE, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # el lifespan carga el modelo una vez por modulo
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["modelo"] == "Random Forest"
    assert 0 < body["umbral"] < 1


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
        with TestClient(app) as c:
            assert c.get("/health").status_code == 503
            assert c.get("/model/info").status_code == 503
            assert c.post("/predict", json=EJEMPLO_SOLICITANTE).status_code == 503
    finally:
        model_deploy.ESTADO.update(estado_previo)


def test_validar_esquema_detecta_faltantes():
    df = pd.DataFrame([EJEMPLO_SOLICITANTE]).drop(columns=["saldo_total"])
    with pytest.raises(ValueError, match="saldo_total"):
        validar_esquema(df)
