"""Pruebas de la app Streamlit (app_streamlit.py).

- AppTest ejecuta el script completo como lo haria `streamlit run`: carga de artefactos, las 4
  pestanas, toggle de tema y envio del formulario, sin excepciones.
- La pestana de lote usa st.file_uploader, que AppTest no simula: se prueba llamando a la funcion
  con el uploader reemplazado por un CSV en memoria.
"""

import io

import pytest
from streamlit.testing.v1 import AppTest

import app_streamlit as app
from conftest import SRC_DIR

TIMEOUT = 180  # la primera corrida carga el dataset y predice el historico completo


@pytest.fixture(scope="module")
def at():
    prueba = AppTest.from_file(str(SRC_DIR / "app_streamlit.py"), default_timeout=TIMEOUT)
    prueba.run()
    return prueba


def test_app_carga_sin_errores(at):
    assert not at.exception
    assert at.title[0].value == "Modelo de riesgo crediticio"
    assert [t.label for t in at.tabs] == ["Prediccion", "Explicacion del modelo", "Lote (CSV)", "Monitoreo de drift"]


def test_formulario_devuelve_resultado(at):
    at.button[0].click().run()
    assert not at.exception
    etiquetas = [m.label for m in at.metric]
    assert "Probabilidad de mora" in etiquetas
    assert "resultado" in at.session_state


def test_modo_oscuro(at):
    at.toggle(key="modo_oscuro").set_value(True).run()
    assert not at.exception
    assert at.session_state["_tema_aplicado"] == "oscuro"


@pytest.mark.parametrize("proba, etiqueta", [(0.9, "ALTO RIESGO"), (0.4, "RIESGO MEDIO"), (0.1, "BAJO RIESGO")])
def test_nivel_riesgo(proba, etiqueta):
    assert app.nivel_riesgo(proba, umbral=0.5)[0] == etiqueta


def _subir(monkeypatch, contenido: str):
    monkeypatch.setattr(app.st, "file_uploader", lambda *a, **k: io.BytesIO(contenido.encode("utf-8")))


def test_tab_lote_predice_y_ofrece_descarga(monkeypatch, datos, modelo):
    _subir(monkeypatch, datos.head(20).to_csv(index=False))
    descargas = []
    monkeypatch.setattr(app.st, "download_button", lambda *a, **k: descargas.append(a))
    app.tab_lote(modelo, umbral_modelo=0.5)
    assert descargas
    assert b"probabilidad_mora" in descargas[0][1]


def test_tab_lote_esquema_invalido_muestra_error(monkeypatch, datos, modelo):
    _subir(monkeypatch, datos.head(5).drop(columns=["huella_consulta"]).to_csv(index=False))
    errores = []
    monkeypatch.setattr(app.st, "error", errores.append)
    app.tab_lote(modelo, umbral_modelo=0.5)
    assert errores
    assert "huella_consulta" in errores[0]


def test_tab_monitoreo_sin_reportes_avisa(monkeypatch):
    avisos = []
    monkeypatch.setattr(app, "cargar_drift", dict)
    monkeypatch.setattr(app.st, "warning", avisos.append)
    app.tab_monitoreo("claro")
    assert avisos
    assert "model_monitoring.py" in avisos[0]
