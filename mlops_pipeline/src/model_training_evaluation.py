"""
Entrenamiento y evaluacion de modelos supervisados para riesgo crediticio (PI M5).

Flujo:
1. Carga el dataset crudo y hace el split estratificado (ft_engineering).
2. Para cada modelo arma ``Pipeline(preprocesador, clasificador)`` -> el fit del
   preprocesador ocurre dentro de cada fold, sin fuga de imputaciones ni percentiles.
3. Validacion cruzada estratificada (5 folds) con UN solo ajuste por fold: de las
   probabilidades out-of-fold (OOF) salen ROC-AUC y PR-AUC por fold, el umbral de
   decision (maximo F1 sobre el OOF completo) y precision / recall / F1 por fold a
   ese mismo umbral. CV y test quedan bajo la misma regla de decision.
   Limitacion conocida: el umbral se elige sobre el mismo OOF con el que se reporta
   F1 en CV, lo que introduce un sesgo optimista leve. Una validacion anidada lo
   eliminaria a costa de 5x tiempo; el test (nunca usado para elegir) es la
   estimacion honesta.
4. Evaluacion en test, tabla comparativa, curvas ROC / PR, matrices de confusion.
5. Seleccion del ganador por ROC-AUC medio en CV (calidad del ranking, independiente
   del umbral), desempate por PR-AUC y F1 en CV. El umbral es una decision operativa
   posterior, no un criterio de seleccion.
6. Importancia por permutacion sobre test: analisis post-hoc para explicar el modelo
   elegido, no evidencia adicional de performance.
7. Serializacion del pipeline completo (preprocesador + modelo) en
   ``mlops_pipeline/models/modelo_riesgo.joblib``; es el unico artefacto de inferencia.

Uso::

    python mlops_pipeline/src/model_training_evaluation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # sin ventanas: el script corre en CI / consola
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import (  # noqa: E402
    CONFIG,
    MODELS_DIR,
    RAIZ,
    RANDOM_STATE,
    cargar_datos,
    construir_preprocesador,
    dividir_train_test,
    separar_target,
)

REPORTS_DIR = RAIZ / "mlops_pipeline" / "reports"
FIG_DIR = REPORTS_DIR / "figures"
CV_FOLDS = CONFIG["cv_folds"]
METRICAS = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Modelos candidatos
# ---------------------------------------------------------------------------
def definir_modelos(ratio_desbalance: float) -> dict[str, object]:
    """Clasificadores a comparar. Todos compensan el desbalance (4,75% de mora).

    ratio_desbalance = negativos / positivos, se usa como scale_pos_weight en XGBoost
    y como class_weight en los demas. LightGBM se descarto: la version 4.7 falla con
    numpy 2.5 en Windows (access violation) y HistGradientBoosting cubre el mismo enfoque.
    """
    return {
        "Dummy (siempre paga)": DummyClassifier(strategy="most_frequent"),
        "Regresion Logistica": LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_depth=4, l2_regularization=1.0,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=ratio_desbalance,
            eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Utilidades de evaluacion
# ---------------------------------------------------------------------------
def umbral_optimo_f1(y_true, proba) -> float:
    """Umbral que maximiza F1 de la clase mora sobre probabilidades out-of-fold."""
    precision, recall, umbrales = precision_recall_curve(y_true, proba)
    if umbrales.size == 0:
        raise ValueError("No se pudo calcular un umbral: las probabilidades no tienen variacion.")
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    return float(umbrales[int(np.argmax(f1))])


def calcular_metricas(y_true, proba, umbral: float) -> dict[str, float]:
    """Metricas de ranking (independientes del umbral) y de decision (al umbral dado)."""
    if len(y_true) != len(proba):
        raise ValueError("y_true y proba deben tener la misma cantidad de observaciones.")
    pred = (proba >= umbral).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def probabilidades_oof(pipe: Pipeline, X, y, cv) -> tuple[np.ndarray, list[np.ndarray]]:
    """Un ajuste por fold. Devuelve las probabilidades out-of-fold y los indices de cada fold."""
    proba = np.zeros(len(y), dtype=float)
    folds = []
    for idx_train, idx_val in cv.split(X, y):
        modelo_fold = clone(pipe).fit(X.iloc[idx_train], y.iloc[idx_train])
        proba[idx_val] = modelo_fold.predict_proba(X.iloc[idx_val])[:, 1]
        folds.append(idx_val)
    if not np.isfinite(proba).all():
        raise RuntimeError("La validacion OOF produjo probabilidades no finitas.")
    return proba, folds


def evaluar_modelo(nombre, clasificador, X_train, y_train, X_test, y_test, cv):
    """CV (un fit por fold) + umbral OOF + ajuste final + test. Devuelve (fila, pipeline, proba_test)."""
    pipe = Pipeline([("preprocesador", construir_preprocesador()), ("modelo", clasificador)])
    t0 = time.perf_counter()

    proba_oof, folds = probabilidades_oof(pipe, X_train, y_train, cv)
    es_dummy = nombre.startswith("Dummy")
    umbral = 0.5 if es_dummy else umbral_optimo_f1(y_train, proba_oof)

    # Metricas por fold, todas al mismo umbral que se usara en test
    y_arr = y_train.to_numpy()
    por_fold = pd.DataFrame([calcular_metricas(y_arr[idx], proba_oof[idx], umbral) for idx in folds])

    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]

    fila = {"modelo": nombre, "umbral": round(umbral, 4)}
    for k in METRICAS:
        fila[f"cv_{k}_mean"] = por_fold[k].mean()
        fila[f"cv_{k}_std"] = por_fold[k].std(ddof=0)
    fila.update({f"test_{k}": v for k, v in calcular_metricas(y_test, proba_test, umbral).items()})
    fila["segundos"] = round(time.perf_counter() - t0, 1)
    print(f"  {nombre:<22} CV ROC-AUC {fila['cv_roc_auc_mean']:.3f}+-{fila['cv_roc_auc_std']:.3f} | "
          f"CV PR-AUC {fila['cv_pr_auc_mean']:.3f} | test ROC-AUC {fila['test_roc_auc']:.3f} | "
          f"test recall {fila['test_recall']:.2f} @ umbral {umbral:.2f} | {fila['segundos']}s")
    return fila, pipe, proba_test


def seleccionar_ganador(tabla: pd.DataFrame) -> str:
    """Criterio: ROC-AUC medio en CV (calidad del ranking, no depende del umbral);
    desempate por PR-AUC (sensible al desbalance) y luego F1 en CV al umbral operativo.

    El accuracy no participa: el Dummy tendria 95% y seria inutil. El umbral se decide
    despues, como parametro operativo del modelo ya elegido.
    """
    candidatos = tabla[~tabla["modelo"].str.startswith("Dummy")]
    orden = ["cv_roc_auc_mean", "cv_pr_auc_mean", "cv_f1_mean"]
    return candidatos.sort_values(orden, ascending=False).iloc[0]["modelo"]


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def graficar_comparacion(tabla: pd.DataFrame) -> None:
    cols = ["cv_roc_auc_mean", "cv_pr_auc_mean", "cv_recall_mean", "cv_precision_mean", "cv_f1_mean"]
    largo = tabla.melt(id_vars="modelo", value_vars=cols, var_name="metrica", value_name="valor")
    largo["metrica"] = largo["metrica"].str.replace("cv_", "").str.replace("_mean", "").str.upper()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=largo, x="metrica", y="valor", hue="modelo", ax=ax)
    ax.set_title(f"Comparacion de modelos: media en CV ({CV_FOLDS} folds, clase mora, umbral optimizado por modelo)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparacion_modelos.png", dpi=120)
    plt.close(fig)


def graficar_curvas(y_test, probas: dict[str, np.ndarray]) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    base = y_test.mean()
    for nombre, proba in probas.items():
        if nombre.startswith("Dummy"):
            continue
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, label=f"{nombre} (AUC {roc_auc_score(y_test, proba):.3f})")
        ax_pr.plot(rec, prec, label=f"{nombre} (AP {average_precision_score(y_test, proba):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="azar")
    ax_pr.axhline(base, color="k", ls="--", lw=1, label=f"azar ({base:.3f})")
    ax_roc.set(xlabel="Tasa de falsos positivos", ylabel="Recall (mora detectada)", title="Curva ROC en test")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall en test")
    ax_roc.legend(fontsize=8)
    ax_pr.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "curvas_roc_pr.png", dpi=120)
    plt.close(fig)


def graficar_matrices(y_test, probas: dict[str, np.ndarray], umbrales: dict[str, float]) -> None:
    nombres = [n for n in probas if not n.startswith("Dummy")]
    fig, axes = plt.subplots(1, len(nombres), figsize=(3.6 * len(nombres), 3.8))
    for ax, nombre in zip(np.atleast_1d(axes), nombres, strict=True):
        pred = (probas[nombre] >= umbrales[nombre]).astype(int)
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["paga", "mora"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"{nombre}\numbral {umbrales[nombre]:.2f}", fontsize=9)
    fig.suptitle("Matrices de confusion en test (umbral optimizado por F1 en train)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "matrices_confusion.png", dpi=120)
    plt.close(fig)


def importancia_variables(pipe: Pipeline, X_test, y_test) -> pd.DataFrame:
    """Importancia por permutacion sobre las features ya transformadas (agnostica al modelo).

    Analisis post-hoc sobre test para explicar el ganador; no interviene en la seleccion.
    """
    X_t = pipe.named_steps["preprocesador"].transform(X_test)
    res = permutation_importance(
        pipe.named_steps["modelo"], X_t, y_test, scoring="roc_auc", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    imp = (pd.DataFrame({"feature": X_t.columns, "importancia": res.importances_mean, "std": res.importances_std})
           .sort_values("importancia", ascending=False))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.barplot(data=imp.head(20), x="importancia", y="feature", ax=ax, color="steelblue")
    ax.set_title("Top 20 features por importancia de permutacion (caida de ROC-AUC en test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "importancia_variables.png", dpi=120)
    plt.close(fig)
    return imp


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = separar_target(cargar_datos())
    X_train, X_test, y_train, y_test = dividir_train_test(X, y)
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Train {X_train.shape} | Test {X_test.shape} | mora train {y_train.mean():.2%} | ratio neg/pos {ratio:.1f}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    filas, pipelines, probas, umbrales = [], {}, {}, {}
    print(f"\nValidacion cruzada ({CV_FOLDS} folds) + evaluacion en test:")
    for nombre, clf in definir_modelos(ratio).items():
        fila, pipe, proba = evaluar_modelo(nombre, clf, X_train, y_train, X_test, y_test, cv)
        filas.append(fila)
        pipelines[nombre], probas[nombre], umbrales[nombre] = pipe, proba, fila["umbral"]

    tabla = pd.DataFrame(filas)
    tabla.to_csv(REPORTS_DIR / "comparacion_modelos.csv", index=False)
    graficar_comparacion(tabla)
    graficar_curvas(y_test, probas)
    graficar_matrices(y_test, probas, umbrales)

    ganador = seleccionar_ganador(tabla)
    fila_g = tabla.set_index("modelo").loc[ganador]
    pipe_g = pipelines[ganador]
    imp = importancia_variables(pipe_g, X_test, y_test)
    imp.to_csv(REPORTS_DIR / "importancia_variables.csv", index=False)

    pred_g = (probas[ganador] >= umbrales[ganador]).astype(int)
    cm = confusion_matrix(y_test, pred_g)
    features_modelo = list(pipe_g.named_steps["preprocesador"].transform(X_test.head(1)).columns)
    metricas = {
        "modelo_ganador": ganador,
        "criterio_seleccion": "mayor ROC-AUC medio en CV; desempate por PR-AUC y F1 en CV. Accuracy excluido por desbalance.",
        "umbral_decision": float(umbrales[ganador]),
        "nota_umbral": "Elegido maximizando F1 sobre probabilidades out-of-fold del train; CV y test se reportan a este umbral.",
        "cv": {k: {"mean": float(fila_g[f"cv_{k}_mean"]), "std": float(fila_g[f"cv_{k}_std"])} for k in METRICAS},
        "test": {k: float(fila_g[f"test_{k}"]) for k in METRICAS},
        "matriz_confusion_test": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "top_features": imp.head(10)["feature"].tolist(),
        "nota_importancia": "Permutacion sobre test, analisis post-hoc; no participa de la seleccion.",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "tasa_mora_train": float(y_train.mean()),
        "features_modelo": features_modelo,
        "random_state": RANDOM_STATE,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(features_modelo, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(pipe_g, MODELS_DIR / "modelo_riesgo.joblib")

    print("\n=== Tabla comparativa ===")
    cols = ["modelo", "cv_roc_auc_mean", "cv_pr_auc_mean", "cv_recall_mean", "cv_precision_mean", "cv_f1_mean",
            "test_roc_auc", "test_pr_auc", "test_recall", "test_precision", "test_f1", "umbral"]
    print(tabla[cols].round(3).to_string(index=False))
    print(f"\nGanador: {ganador} | umbral {umbrales[ganador]:.2f} | matriz test tn={cm[0,0]} fp={cm[0,1]} fn={cm[1,0]} tp={cm[1,1]}")
    print("Top 10 features:", ", ".join(metricas["top_features"]))
    print(f"Modelo guardado en {MODELS_DIR / 'modelo_riesgo.joblib'} | reportes en {REPORTS_DIR}")


if __name__ == "__main__":
    main()
