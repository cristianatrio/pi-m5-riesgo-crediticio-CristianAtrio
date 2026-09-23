"""Pruebas de la ingenieria de caracteristicas (ft_engineering.py)."""

import numpy as np
import pandas as pd
import pytest

import ft_engineering as fe


def test_buscar_raiz_respeta_variable_de_entorno(monkeypatch, tmp_path):
    monkeypatch.setenv("RAIZ_PROYECTO", str(tmp_path))
    assert fe.buscar_raiz() == tmp_path.resolve()


def test_buscar_raiz_sin_dataset_cae_en_la_raiz_del_repo(monkeypatch):
    monkeypatch.delenv("RAIZ_PROYECTO", raising=False)
    assert fe.buscar_raiz("no_existe.csv") == fe.SRC_DIR.parents[1]


def test_validar_esquema_rechaza_no_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        fe.validar_esquema([{"a": 1}])


def test_separar_target_invierte_la_etiqueta(datos):
    X, y = fe.separar_target(datos)
    assert fe.TARGET not in X.columns
    assert y.name == "mora"
    assert (y == 1 - datos[fe.TARGET]).all()
    assert 0.04 < y.mean() < 0.06  # 4,75% de mora documentado en el EDA


def test_separar_target_sin_target_falla(datos):
    sin_target = datos.drop(columns=[fe.TARGET])
    with pytest.raises(ValueError, match=fe.TARGET):
        fe.separar_target(sin_target)


def test_dividir_train_test_estratificado(datos):
    X, y = fe.separar_target(datos)
    X_train, X_test, y_train, y_test = fe.dividir_train_test(X, y)
    assert len(X_test) == pytest.approx(0.2 * len(X), abs=1)
    assert abs(y_train.mean() - y_test.mean()) < 0.005
    assert set(X_train.index).isdisjoint(X_test.index)


def test_limpieza_convierte_valores_imposibles_en_nulos():
    crudo = pd.DataFrame([{c: 1 for c in fe.COLUMNAS_REQUERIDAS}] * 3)
    crudo["edad_cliente"] = [121, 40, 30]
    crudo["puntaje_datacredito"] = [800, 1200, 100]
    crudo["salario_cliente"] = [0, 2_000_000, 3_000_000]
    crudo["tendencia_ingresos"] = ["Creciente", "123", None]
    crudo["tipo_laboral"] = "Empleado"
    crudo["puntaje"] = 95.0  # excluida por fuga de informacion

    limpio = fe.LimpiezaCredito().fit(crudo).transform(crudo)

    assert "puntaje" not in limpio.columns
    assert limpio["edad_cliente"].isna().tolist() == [True, False, False]
    assert limpio["puntaje_datacredito"].isna().tolist() == [False, True, True]
    assert np.isnan(limpio.loc[0, "salario_cliente"])
    assert limpio["tendencia_ingresos"].isna().tolist() == [False, True, True]
    assert limpio["tipo_credito"].map(type).eq(str).all()


def test_features_credito_crea_ratios_y_flags():
    X = pd.DataFrame({
        "salario_cliente": [1_000_000, 0], "cuota_pactada": [100_000, 50_000],
        "total_otros_prestamos": [2_000_000, 0], "capital_prestado": [5_000_000, 1_000_000],
        "plazo_meses": [36, 10], "saldo_mora": [10, 0], "creditos_sectorFinanciero": [0, 2],
        "creditos_sectorCooperativo": [1, 0], "creditos_sectorReal": [1, 1],
    })
    paso = fe.FeaturesCredito().fit(X)
    out = paso.transform(X)
    assert out.loc[0, "ratio_cuota_salario"] == pytest.approx(0.1)
    assert out.loc[1, "ratio_cuota_salario"] == pytest.approx(50_000)  # salario 0 -> se acota a 1, sin division por cero
    assert out["plazo_largo"].tolist() == [1, 0]
    assert out["tiene_mora_previa"].tolist() == [1, 0]
    assert out["sin_historial_financiero"].tolist() == [1, 0]
    assert out["creditos_total_sectores"].tolist() == [2, 3]
    assert list(paso.get_feature_names_out()) == list(out.columns)


def test_preprocesador_sin_nulos_y_sin_columnas_excluidas(muestra):
    X, y = fe.separar_target(muestra)
    prep = fe.construir_preprocesador().fit(X, y)
    Xt = prep.transform(X)
    assert not Xt.isna().any().any()
    assert not {"puntaje", "fecha_prestamo"} & set(Xt.columns)
    assert list(prep.named_steps["limpieza"].get_feature_names_out()) == [
        c for c in X.columns if c not in fe.COLS_EXCLUIDAS
    ]


def test_preprocesador_sin_escalado_no_agrega_standard_scaler():
    assert "escalado" not in fe.construir_preprocesador(escalar=False).named_steps


def test_main_genera_los_splits(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fe, "DATA_DIR", tmp_path)
    fe.main()
    assert {p.name for p in tmp_path.glob("*.parquet")} == {
        "X_train.parquet", "X_test.parquet", "y_train.parquet", "y_test.parquet",
    }
    assert "descartadas por correlacion" in capsys.readouterr().out
