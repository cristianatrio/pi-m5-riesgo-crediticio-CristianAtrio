"""Pruebas del monitoreo de data drift (model_monitoring.py).

Se importa el modulo con alias: sus funciones test_ks / test_chi2 no deben quedar en el
namespace de pytest, que las tomaria como pruebas.
"""

import json

import numpy as np
import pandas as pd
import pytest

import model_monitoring as mm


@pytest.fixture
def drift_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(mm, "DRIFT_DIR", tmp_path)
    return tmp_path


def test_psi_numerico_misma_distribucion_es_casi_cero():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert mm.calcular_psi_numerico(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.02


def test_psi_numerico_distribucion_desplazada_es_critico():
    rng = np.random.default_rng(0)
    psi = mm.calcular_psi_numerico(pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(1.5, 1, 5000)))
    assert mm.nivel_psi(psi) == "critico"


def test_psi_numerico_variable_casi_constante_y_vacia():
    constante = pd.Series([0.0] * 100)
    assert mm.calcular_psi_numerico(constante, constante) == pytest.approx(0.0)
    assert np.isnan(mm.calcular_psi_numerico(pd.Series([], dtype=float), constante))


def test_psi_categorico_detecta_categoria_nueva():
    ref = pd.Series(["a"] * 90 + ["b"] * 10)
    assert mm.calcular_psi_categorico(ref, ref) == pytest.approx(0.0)
    assert mm.calcular_psi_categorico(ref, pd.Series(["c"] * 100)) > mm.CFG["psi_critico"]


def test_tests_estadisticos():
    ref = pd.Series(np.arange(100, dtype=float))
    _, p_igual = mm.test_ks(ref, ref)
    assert p_igual == pytest.approx(1.0)
    _, p_distinto = mm.test_chi2(pd.Series(["a"] * 50 + ["b"] * 50), pd.Series(["b"] * 100))
    assert p_distinto < 0.05
    assert mm.test_chi2(pd.Series(["a"] * 10), pd.Series(["a"] * 10)) == (0.0, 1.0)


@pytest.mark.parametrize("psi, nivel", [(0.05, "ok"), (0.15, "alerta"), (0.3, "critico"), (float("nan"), "sin_datos")])
def test_nivel_psi(psi, nivel):
    assert mm.nivel_psi(psi) == nivel


def test_simular_escenarios():
    temporal = mm.simular_escenario("temporal")
    assert (temporal["fecha_prestamo"] >= pd.Timestamp(mm.CFG["fecha_corte_simulacion"])).all()
    assert len(mm.simular_escenario("sintetico")) == 2000
    with pytest.raises(ValueError, match="desconocido"):
        mm.simular_escenario("otro")


def test_detectar_drift_sin_cambio_no_alerta(datos):
    tabla = mm.detectar_drift(datos, datos)
    assert len(tabla) == len(mm.COLUMNAS_REQUERIDAS)
    assert (tabla["nivel"] == "ok").all()


def test_resumen_global_reglas_de_alerta():
    tabla = pd.DataFrame({"feature": ["a", "b", "c"], "nivel": ["alerta", "alerta", "ok"]})
    assert not mm.resumen_global(tabla, {"nivel": "ok"})["hay_drift"]
    assert mm.resumen_global(tabla, {"nivel": "critico"})["hay_drift"]
    tabla.loc[2, "nivel"] = "critico"
    assert mm.resumen_global(tabla, {"nivel": "ok"})["features_critico"] == ["c"]


def test_drift_target_solo_si_viene_el_target(datos):
    assert mm.drift_target(datos, datos.drop(columns=[mm.TARGET])) is None
    assert mm.drift_target(datos, datos)["diferencia_pp"] == pytest.approx(0.0)


def test_generar_reporte_escribe_json_md_csv_y_figuras(drift_dir, datos, modelo):
    actual = mm.simular_escenario("sintetico")
    rep = mm.generar_reporte(datos, actual, "prueba", modelo, umbral=0.5)
    assert rep["resumen"]["hay_drift"]
    for nombre in ("drift_report_prueba.json", "drift_report_prueba.md", "drift_features_prueba.csv",
                   "psi_prueba.png", "distribuciones_prueba.png"):
        assert (drift_dir / nombre).exists(), nombre
    assert "Estado global" in (drift_dir / "drift_report_prueba.md").read_text(encoding="utf-8")


def test_main_escenario_sintetico_estricto(drift_dir, capsys):
    assert mm.main(["--escenario", "sintetico"]) == 0
    assert mm.main(["--escenario", "sintetico", "--strict"]) == 1
    assert "DRIFT DETECTADO" in capsys.readouterr().out


def test_main_con_csv_de_produccion(drift_dir, datos, tmp_path):
    ventana = tmp_path / "ventana.csv"
    datos.sample(500, random_state=1).to_csv(ventana, index=False)
    assert mm.main(["--nuevos", str(ventana)]) == 0
    rep = json.loads((drift_dir / "drift_report_produccion.json").read_text(encoding="utf-8"))
    assert rep["n_actual"] == 500


def test_main_falla_si_el_detector_no_marca_el_sintetico(drift_dir, monkeypatch, datos):
    monkeypatch.setattr(mm, "simular_escenario", lambda _nombre: datos.sample(2000, random_state=2))
    assert mm.main(["--escenario", "sintetico"]) == 2
