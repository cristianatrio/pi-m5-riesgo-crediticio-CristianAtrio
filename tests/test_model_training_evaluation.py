"""Pruebas de entrenamiento y evaluacion (model_training_evaluation.py).

El entrenamiento real (5 modelos, RF de 500 arboles) tarda minutos; aca se valida la logica con
modelos rapidos y se corre main() completo con dos candidatos y 2 folds sobre carpetas temporales.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import ft_engineering as fe
import model_training_evaluation as mte


def test_definir_modelos_compensa_el_desbalance():
    modelos = mte.definir_modelos(ratio_desbalance=20.0)
    assert set(modelos) == {"Dummy (siempre paga)", "Regresion Logistica", "Random Forest", "HistGradientBoosting", "XGBoost"}
    assert modelos["XGBoost"].get_params()["scale_pos_weight"] == 20.0
    assert modelos["Random Forest"].get_params()["class_weight"] == "balanced_subsample"


def test_umbral_optimo_f1_separa_clases_perfectas():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert mte.umbral_optimo_f1(y, proba) == pytest.approx(0.8)


def test_umbral_optimo_f1_sin_variacion_falla(monkeypatch):
    monkeypatch.setattr(mte, "precision_recall_curve", lambda *_: (np.ones(1), np.ones(1), np.array([])))
    with pytest.raises(ValueError, match="umbral"):
        mte.umbral_optimo_f1(np.array([0, 1]), np.array([0.5, 0.5]))


def test_calcular_metricas_valores_conocidos():
    y = np.array([0, 0, 1, 1])
    m = mte.calcular_metricas(y, np.array([0.1, 0.6, 0.4, 0.9]), umbral=0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_calcular_metricas_largos_distintos_falla():
    with pytest.raises(ValueError, match="misma cantidad"):
        mte.calcular_metricas(np.array([0, 1]), np.array([0.2]), 0.5)


def test_seleccionar_ganador_ignora_dummy_y_desempata():
    tabla = pd.DataFrame({
        "modelo": ["Dummy (siempre paga)", "A", "B"],
        "cv_roc_auc_mean": [0.99, 0.70, 0.70],
        "cv_pr_auc_mean": [0.9, 0.12, 0.15],
        "cv_f1_mean": [0.0, 0.2, 0.1],
    })
    assert mte.seleccionar_ganador(tabla) == "B"


def test_evaluar_modelo_devuelve_cv_y_test(muestra):
    X, y = fe.separar_target(muestra)
    X_train, X_test, y_train, y_test = fe.dividir_train_test(X, y)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    clf = LogisticRegression(class_weight="balanced", max_iter=500)

    fila, pipe, proba = mte.evaluar_modelo("Regresion Logistica", clf, X_train, y_train, X_test, y_test, cv)

    assert proba.shape == (len(X_test),) and ((proba >= 0) & (proba <= 1)).all()
    assert {f"cv_{k}_mean" for k in mte.METRICAS} <= set(fila)
    assert {f"test_{k}" for k in mte.METRICAS} <= set(fila)
    assert 0 < fila["umbral"] < 1
    assert "preprocesador" in pipe.named_steps


def test_main_entrena_selecciona_y_serializa(monkeypatch, tmp_path):
    reports, figs, models = tmp_path / "reports", tmp_path / "reports" / "figures", tmp_path / "models"
    monkeypatch.setattr(mte, "REPORTS_DIR", reports)
    monkeypatch.setattr(mte, "FIG_DIR", figs)
    monkeypatch.setattr(mte, "MODELS_DIR", models)
    monkeypatch.setattr(mte, "CV_FOLDS", 2)
    monkeypatch.setattr(mte, "definir_modelos", lambda _ratio: {
        "Dummy (siempre paga)": DummyClassifier(strategy="most_frequent"),
        "Regresion Logistica": LogisticRegression(class_weight="balanced", max_iter=500),
    })

    mte.main()

    metricas = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    assert metricas["modelo_ganador"] == "Regresion Logistica"
    assert metricas["test"]["roc_auc"] > 0.6
    assert (models / "modelo_riesgo.joblib").exists()
    assert json.loads((models / "feature_names.json").read_text(encoding="utf-8")) == metricas["features_modelo"]
    assert {p.name for p in figs.glob("*.png")} == {
        "comparacion_modelos.png", "curvas_roc_pr.png", "matrices_confusion.png", "importancia_variables.png",
    }
    assert len(pd.read_csv(reports / "comparacion_modelos.csv")) == 2
