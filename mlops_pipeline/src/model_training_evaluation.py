"""
Entrenamiento y evaluacion de modelos supervisados para riesgo crediticio (PI M5).

Flujo:
1. Carga el dataset crudo y excluye los creditos censurados: los que todavia no vencieron
   ni tienen la ventana minima de observacion (ft_engineering.filtrar_censura, EDA 3.6).
2. Split estratificado train / test y, aparte, split out-of-time (historico vs. ventana
   mas reciente) para medir como rinde el modelo con creditos futuros.
3. Para cada modelo arma ``Pipeline(preprocesador, clasificador)`` -> el fit del
   preprocesador ocurre dentro de cada fold, sin fuga de imputaciones ni percentiles.
4. Validacion cruzada estratificada (5 folds) con UN solo ajuste por fold. De las
   probabilidades out-of-fold (OOF) salen ROC-AUC, PR-AUC, KS y Gini por fold y el
   umbral de decision: el que MINIMIZA EL COSTO ESPERADO de negocio (moras aprobadas
   x perdida + buenos clientes rechazados x margen, ponderado por capital; supuestos en
   config.json -> "costos"). Precision / recall / F1 por fold se reportan a ese umbral.
   Tambien se registra el umbral de maximo F1 como referencia.
   Limitacion conocida: el umbral se elige sobre el mismo OOF con el que se reportan
   las metricas de decision en CV (sesgo optimista leve); test y out-of-time, que nunca
   intervienen en la eleccion, son las estimaciones honestas.
5. Evaluacion en test y out-of-time, tabla comparativa, curvas ROC / PR, matrices de
   confusion y curva de costo por umbral del ganador.
6. Seleccion del ganador por ROC-AUC medio en CV (calidad del ranking, independiente
   del umbral), desempate por PR-AUC y F1 en CV.
7. Importancia por permutacion sobre test: analisis post-hoc del modelo elegido.
8. Serializacion del pipeline completo (preprocesador + modelo) en
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
    dividir_temporal,
    dividir_train_test,
    filtrar_censura,
    separar_target,
)

REPORTS_DIR = RAIZ / "mlops_pipeline" / "reports"
FIG_DIR = REPORTS_DIR / "figures"
CV_FOLDS = CONFIG["cv_folds"]
COSTOS = CONFIG["costos"]
METRICAS = ["roc_auc", "pr_auc", "ks", "gini", "precision", "recall", "f1"]
UMBRALES_CANDIDATOS = np.round(np.arange(0.01, 1.0, 0.01), 2)
UMBRAL_DUMMY = 0.5
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
        "Dummy (siempre paga)": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "Regresion Logistica": LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample",
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
# Umbral de decision: F1 (referencia) y costo de negocio (operativo)
# ---------------------------------------------------------------------------
def umbral_optimo_f1(y_true, proba) -> float:
    """Umbral que maximiza F1 de la clase mora sobre probabilidades out-of-fold."""
    precision, recall, umbrales = precision_recall_curve(y_true, proba)
    if umbrales.size == 0:
        raise ValueError("No se pudo calcular un umbral: las probabilidades no tienen variacion.")
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    return float(umbrales[int(np.argmax(f1))])


def costo_por_umbral(y_true, proba, capital, umbrales=UMBRALES_CANDIDATOS) -> np.ndarray:
    """Costo esperado para cada umbral, ponderado por el capital de cada credito.

    Mora aprobada (falso negativo): se pierde ``perdida_por_mora`` x capital.
    Buen pagador rechazado (falso positivo): se pierde ``margen_por_credito_sano`` x capital.
    """
    y = np.asarray(y_true)
    cap = np.asarray(capital, dtype=float)
    rechaza = np.asarray(proba)[None, :] >= np.asarray(umbrales)[:, None]
    moras_aprobadas = (~rechaza & (y == 1)) @ cap
    sanos_rechazados = (rechaza & (y == 0)) @ cap
    return COSTOS["perdida_por_mora"] * moras_aprobadas + COSTOS["margen_por_credito_sano"] * sanos_rechazados


def umbral_optimo_costo(y_true, proba, capital) -> float:
    """Umbral candidato con el menor costo esperado sobre las probabilidades OOF."""
    return float(UMBRALES_CANDIDATOS[int(np.argmin(costo_por_umbral(y_true, proba, capital)))])


def resumen_costo(y_true, proba, capital, umbral: float) -> dict[str, float]:
    """Costo con el modelo vs. aprobar a todos (la politica sin modelo) y tasa de rechazo."""
    y = np.asarray(y_true)
    cap = np.asarray(capital, dtype=float)
    costo = float(costo_por_umbral(y, proba, cap, [umbral])[0])
    sin_modelo = float(COSTOS["perdida_por_mora"] * cap[y == 1].sum())
    return {
        "costo": costo,
        "costo_sin_modelo": sin_modelo,
        "ahorro_pct": 1 - costo / sin_modelo if sin_modelo > 0 else 0.0,
        "tasa_rechazo": float((np.asarray(proba) >= umbral).mean()),
    }


# ---------------------------------------------------------------------------
# Utilidades de evaluacion
# ---------------------------------------------------------------------------
def calcular_metricas(y_true, proba, umbral: float) -> dict[str, float]:
    """Metricas de ranking (independientes del umbral) y de decision (al umbral dado).

    KS y Gini son las metricas estandar de scoring crediticio: KS es la maxima separacion
    entre las distribuciones acumuladas de morosos y cumplidores; Gini = 2 x ROC-AUC - 1.
    """
    if len(y_true) != len(proba):
        raise ValueError("y_true y proba deben tener la misma cantidad de observaciones.")
    pred = (proba >= umbral).astype(int)
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    return {
        "roc_auc": auc,
        "pr_auc": average_precision_score(y_true, proba),
        "ks": float(np.max(tpr - fpr)),
        "gini": 2 * auc - 1,
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


def evaluar_oot(pipe: Pipeline, oot: tuple, umbral: float) -> dict[str, float]:
    """Reentrena con el historico y evalua en la ventana mas reciente (out-of-time)."""
    x_hist, x_oot, y_hist, y_oot = oot
    proba = clone(pipe).fit(x_hist, y_hist).predict_proba(x_oot)[:, 1]
    resultado = {f"oot_{k}": v for k, v in calcular_metricas(y_oot, proba, umbral).items()}
    resultado["oot_ahorro_pct"] = resumen_costo(y_oot, proba, x_oot["capital_prestado"], umbral)["ahorro_pct"]
    return resultado


def evaluar_modelo(nombre, clasificador, X_train, y_train, X_test, y_test, cv, oot=None):
    """CV (un fit por fold) + umbrales OOF + ajuste final + test (+ out-of-time si se pasa).

    Devuelve (fila, pipeline, proba_test).
    """
    # memory=None explicito: sin cache de transformadores (cada fold reajusta el preprocesador)
    pipe = Pipeline([("preprocesador", construir_preprocesador()), ("modelo", clasificador)], memory=None)
    t0 = time.perf_counter()

    proba_oof, folds = probabilidades_oof(pipe, X_train, y_train, cv)
    es_dummy = nombre.startswith("Dummy")
    umbral_f1 = UMBRAL_DUMMY if es_dummy else umbral_optimo_f1(y_train, proba_oof)
    umbral = UMBRAL_DUMMY if es_dummy else umbral_optimo_costo(y_train, proba_oof, X_train["capital_prestado"])

    # Metricas por fold, todas al mismo umbral operativo que se usara en test y produccion
    y_arr = y_train.to_numpy()
    por_fold = pd.DataFrame([calcular_metricas(y_arr[idx], proba_oof[idx], umbral) for idx in folds])

    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]

    fila = {"modelo": nombre, "umbral": round(umbral, 4), "umbral_f1": round(umbral_f1, 4)}
    for k in METRICAS:
        fila[f"cv_{k}_mean"] = por_fold[k].mean()
        fila[f"cv_{k}_std"] = por_fold[k].std(ddof=0)
    fila.update({f"test_{k}": v for k, v in calcular_metricas(y_test, proba_test, umbral).items()})
    costo_test = resumen_costo(y_test, proba_test, X_test["capital_prestado"], umbral)
    fila["test_ahorro_pct"] = costo_test["ahorro_pct"]
    fila["test_tasa_rechazo"] = costo_test["tasa_rechazo"]
    if oot is not None:
        fila.update(evaluar_oot(pipe, oot, umbral))
    fila["segundos"] = round(time.perf_counter() - t0, 1)
    _imprimir_fila(fila)
    return fila, pipe, proba_test


def _imprimir_fila(fila: dict) -> None:
    oot = f" | OOT ROC-AUC {fila['oot_roc_auc']:.3f}" if "oot_roc_auc" in fila else ""
    print(f"  {fila['modelo']:<22} CV ROC-AUC {fila['cv_roc_auc_mean']:.3f}+-{fila['cv_roc_auc_std']:.3f} | "
          f"test ROC-AUC {fila['test_roc_auc']:.3f}{oot} | umbral costo {fila['umbral']:.2f} "
          f"(F1 {fila['umbral_f1']:.2f}) | ahorro test {fila['test_ahorro_pct']:.1%} | {fila['segundos']}s")


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
    cols = ["cv_roc_auc_mean", "cv_pr_auc_mean", "cv_ks_mean", "cv_recall_mean", "cv_precision_mean", "cv_f1_mean"]
    largo = tabla.melt(id_vars="modelo", value_vars=cols, var_name="metrica", value_name="valor")
    largo["metrica"] = largo["metrica"].str.replace("cv_", "").str.replace("_mean", "").str.upper()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=largo, x="metrica", y="valor", hue="modelo", ax=ax)
    ax.set_title(f"Comparacion de modelos: media en CV ({CV_FOLDS} folds, clase mora, umbral de minimo costo)")
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
    fig.suptitle("Matrices de confusion en test (umbral de minimo costo esperado, elegido en train)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "matrices_confusion.png", dpi=120)
    plt.close(fig)


def graficar_costo(y_test, proba, capital, umbral: float, umbral_f1: float) -> None:
    """Costo esperado en test (relativo a aprobar a todos) para cada umbral del ganador."""
    costos = costo_por_umbral(y_test, proba, capital)
    base = resumen_costo(y_test, proba, capital, umbral)["costo_sin_modelo"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(UMBRALES_CANDIDATOS, costos / base, color="steelblue")
    ax.axhline(1, color="gray", ls="--", lw=1, label="sin modelo (aprobar a todos)")
    ax.axvline(umbral, color="#d62728", ls="--", label=f"umbral de minimo costo ({umbral:.2f})")
    ax.axvline(umbral_f1, color="#ff7f0e", ls=":", label=f"umbral de maximo F1 ({umbral_f1:.2f})")
    ax.set(xlabel="Umbral de rechazo", ylabel="Costo relativo a aprobar a todos",
           title="Costo esperado por umbral en test (perdida por mora vs. margen perdido)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "costo_umbral.png", dpi=120)
    plt.close(fig)


def importancia_variables(pipe: Pipeline, X_test, y_test) -> pd.DataFrame:
    """Importancia por permutacion sobre las features ya transformadas (agnostica al modelo).

    Analisis post-hoc sobre test para explicar el ganador; no interviene en la seleccion.
    """
    x_transformado = pipe.named_steps["preprocesador"].transform(X_test)
    res = permutation_importance(
        pipe.named_steps["modelo"], x_transformado, y_test, scoring="roc_auc", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    imp = (pd.DataFrame({"feature": x_transformado.columns, "importancia": res.importances_mean, "std": res.importances_std})
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
def armar_metricas(fila_g: pd.Series, ganador: str, cm: np.ndarray, costo_test: dict, contexto: dict) -> dict:
    """Contenido de metrics.json: lo leen la API, la app, el monitoreo y el readme."""
    oot = {k.removeprefix("oot_"): float(v) for k, v in fila_g.items() if str(k).startswith("oot_")}
    return {
        "modelo_ganador": ganador,
        "criterio_seleccion": "mayor ROC-AUC medio en CV; desempate por PR-AUC y F1 en CV. Accuracy excluido por desbalance.",
        "umbral_decision": float(fila_g["umbral"]),
        "umbral_f1": float(fila_g["umbral_f1"]),
        "nota_umbral": "Umbral operativo = minimo costo esperado sobre probabilidades out-of-fold del train "
                       "(ver 'costos'). Metricas de decision de CV, test y out-of-time reportadas a este umbral. "
                       "umbral_f1 queda como referencia.",
        "costos": {
            "supuestos": {k: v for k, v in COSTOS.items() if k != "nota"},
            "test": costo_test,
        },
        "cv": {k: {"mean": float(fila_g[f"cv_{k}_mean"]), "std": float(fila_g[f"cv_{k}_std"])} for k in METRICAS},
        "test": {k: float(fila_g[f"test_{k}"]) for k in METRICAS},
        "out_of_time": oot,
        "matriz_confusion_test": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "nota_importancia": "Permutacion sobre test, analisis post-hoc; no participa de la seleccion.",
        "random_state": RANDOM_STATE,
        **contexto,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    crudo = cargar_datos()
    df = filtrar_censura(crudo)
    X, y = separar_target(df)
    X_train, X_test, y_train, y_test = dividir_train_test(X, y)
    oot = dividir_temporal(X, y)
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Censura: {len(crudo) - len(df):,} de {len(crudo):,} creditos excluidos (sin ventana de observacion)")
    print(f"Train {X_train.shape} | Test {X_test.shape} | mora train {y_train.mean():.2%} | ratio neg/pos {ratio:.1f}")
    print(f"Out-of-time: historico {len(oot[0]):,} (mora {oot[2].mean():.2%}) | reciente {len(oot[1]):,} "
          f"(mora {oot[3].mean():.2%}) desde {oot[1]['fecha_prestamo'].min():%Y-%m-%d}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    filas, pipelines, probas, umbrales = [], {}, {}, {}
    print(f"\nValidacion cruzada ({CV_FOLDS} folds) + test + out-of-time:")
    for nombre, clf in definir_modelos(ratio).items():
        fila, pipe, proba = evaluar_modelo(nombre, clf, X_train, y_train, X_test, y_test, cv, oot=oot)
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
    graficar_costo(y_test, probas[ganador], X_test["capital_prestado"], fila_g["umbral"], fila_g["umbral_f1"])

    cm = confusion_matrix(y_test, (probas[ganador] >= umbrales[ganador]).astype(int))
    costo_test = resumen_costo(y_test, probas[ganador], X_test["capital_prestado"], umbrales[ganador])
    features_modelo = list(pipe_g.named_steps["preprocesador"].transform(X_test.head(1)).columns)
    metricas = armar_metricas(fila_g, ganador, cm, costo_test, {
        "top_features": imp.head(10)["feature"].tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_excluidos_censura": int(len(crudo) - len(df)),
        "n_oot": int(len(oot[1])),
        "tasa_mora_train": float(y_train.mean()),
        "features_modelo": features_modelo,
    })
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(features_modelo, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(pipe_g, MODELS_DIR / "modelo_riesgo.joblib")
    _imprimir_resumen(tabla, metricas)


def _imprimir_resumen(tabla: pd.DataFrame, metricas: dict) -> None:
    print("\n=== Tabla comparativa ===")
    cols = ["modelo", "cv_roc_auc_mean", "cv_pr_auc_mean", "cv_ks_mean", "test_roc_auc", "test_pr_auc", "test_ks",
            "test_recall", "test_precision", "test_f1", "test_ahorro_pct", "test_tasa_rechazo",
            "oot_roc_auc", "oot_pr_auc", "oot_ahorro_pct", "umbral", "umbral_f1"]
    print(tabla[[c for c in cols if c in tabla.columns]].round(3).to_string(index=False))
    cm, costo = metricas["matriz_confusion_test"], metricas["costos"]["test"]
    print(f"\nGanador: {metricas['modelo_ganador']} | umbral {metricas['umbral_decision']:.2f} "
          f"(F1: {metricas['umbral_f1']:.2f}) | matriz test tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}")
    print(f"Costo test: {costo['costo']:,.0f} vs {costo['costo_sin_modelo']:,.0f} sin modelo "
          f"(ahorro {costo['ahorro_pct']:.1%}, rechazo {costo['tasa_rechazo']:.1%})")
    print("Top 10 features:", ", ".join(metricas["top_features"]))
    print(f"Modelo guardado en {MODELS_DIR / 'modelo_riesgo.joblib'} | reportes en {REPORTS_DIR}")


if __name__ == "__main__":
    main()
