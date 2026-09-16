"""
Monitoreo de data drift para el modelo de riesgo crediticio (PI M5).

Compara una ventana de datos nuevos contra la referencia de entrenamiento (el mismo
split ``random_state=42`` de ``ft_engineering``) sobre las 20 columnas crudas que
recibe la API, y devuelve un reporte con semaforo por variable y alerta global.

Tecnicas (implementacion propia con numpy / scipy, sin dependencias extra):

- Numericas: PSI (Population Stability Index) con bins por cuantiles de la referencia
  + test de Kolmogorov-Smirnov de dos muestras.
- Categoricas: PSI sobre frecuencias + test chi-cuadrado de independencia.
- Drift de prediccion: PSI de ``predict_proba`` del modelo y tasa de solicitudes
  marcadas como riesgo, referencia vs. actual.
- Drift del target (si viene en los datos nuevos): tasa de mora referencia vs. actual.

Semaforo: PSI < 0.1 ok | 0.1-0.25 alerta | > 0.25 critico. Alerta global si hay >= 1
critico o >= ``max_alertas`` alertas (config.json -> "drift").

Uso::

    python mlops_pipeline/src/model_monitoring.py                       # simula ambos escenarios
    python mlops_pipeline/src/model_monitoring.py --escenario temporal  # creditos desde la fecha de corte
    python mlops_pipeline/src/model_monitoring.py --escenario sintetico # drift inyectado (prueba del detector)
    python mlops_pipeline/src/model_monitoring.py --nuevos datos.csv    # ventana real de produccion
    python mlops_pipeline/src/model_monitoring.py --nuevos datos.csv --strict   # exit 1 si hay drift critico (CI)

Salidas en ``mlops_pipeline/reports/drift/``: ``drift_report_<escenario>.json``,
``drift_report_<escenario>.md``, ``drift_features_<escenario>.csv`` y figuras.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import (  # noqa: E402
    COLUMNAS_REQUERIDAS,
    CONFIG,
    MODELS_DIR,
    RAIZ,
    TARGET,
    cargar_datos,
    dividir_train_test,
    separar_target,
    validar_esquema,
)

DRIFT_DIR = RAIZ / "mlops_pipeline" / "reports" / "drift"
METRICS_PATH = RAIZ / "mlops_pipeline" / "reports" / "metrics.json"
CFG = CONFIG["drift"]
CATEGORIAS_TENDENCIA = CONFIG["categorias_tendencia"]

COLS_CATEGORICAS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]
COLS_NUMERICAS = [c for c in COLUMNAS_REQUERIDAS if c not in COLS_CATEGORICAS]
EPS = 1e-4  # evita log(0) y divisiones por cero en el PSI

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Estadisticos
# ---------------------------------------------------------------------------
def _proporciones(conteos: np.ndarray) -> np.ndarray:
    prop = conteos / max(conteos.sum(), 1)
    return np.clip(prop, EPS, None)


def calcular_psi_numerico(referencia: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """PSI con bins definidos por cuantiles de la referencia (robusto a colas largas)."""
    ref = referencia.dropna().to_numpy(dtype=float)
    act = actual.dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(act) == 0:
        return float("nan")
    bordes = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(bordes) < 3:  # variable casi constante (ej. saldo_mora): bins ok / distinto
        bordes = np.array([-np.inf, bordes[0], np.inf])
    else:
        bordes[0], bordes[-1] = -np.inf, np.inf
    p_ref = _proporciones(np.histogram(ref, bins=bordes)[0])
    p_act = _proporciones(np.histogram(act, bins=bordes)[0])
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def calcular_psi_categorico(referencia: pd.Series, actual: pd.Series) -> float:
    """PSI sobre la distribucion de frecuencias (union de categorias de ambas ventanas)."""
    categorias = sorted(set(referencia.dropna().unique()) | set(actual.dropna().unique()), key=str)
    p_ref = _proporciones(referencia.value_counts().reindex(categorias, fill_value=0).to_numpy(dtype=float))
    p_act = _proporciones(actual.value_counts().reindex(categorias, fill_value=0).to_numpy(dtype=float))
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def test_ks(referencia: pd.Series, actual: pd.Series) -> tuple[float, float]:
    res = stats.ks_2samp(referencia.dropna(), actual.dropna())
    return float(res.statistic), float(res.pvalue)


def test_chi2(referencia: pd.Series, actual: pd.Series) -> tuple[float, float]:
    tabla = pd.crosstab(
        pd.concat([referencia, actual], ignore_index=True),
        np.r_[np.zeros(len(referencia)), np.ones(len(actual))],
    )
    if tabla.shape[0] < 2:
        return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(tabla)
    return float(chi2), float(p)


def nivel_psi(psi: float) -> str:
    if np.isnan(psi):
        return "sin_datos"
    if psi >= CFG["psi_critico"]:
        return "critico"
    if psi >= CFG["psi_alerta"]:
        return "alerta"
    return "ok"


# ---------------------------------------------------------------------------
# Preparacion de datos
# ---------------------------------------------------------------------------
def normalizar_para_monitoreo(df: pd.DataFrame) -> pd.DataFrame:
    """Misma lectura que el pipeline: tendencia invalida -> 'Sin dato', tipo_credito como categoria."""
    df = df.copy()
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(CATEGORIAS_TENDENCIA), "Sin dato"
    )
    df["tipo_credito"] = df["tipo_credito"].astype("Int64").astype(str)
    return df


def datos_referencia() -> pd.DataFrame:
    """Train del split oficial, con target incluido para el drift de mora."""
    df = cargar_datos()
    X, y = separar_target(df)
    X_train, _, y_train, _ = dividir_train_test(X, y)
    return X_train.assign(**{TARGET: 1 - y_train})


def simular_escenario(nombre: str) -> pd.DataFrame:
    """Ventanas 'nuevas' para probar el monitoreo sin datos de produccion.

    - temporal: creditos desembolsados desde la fecha de corte (los mas recientes).
    - sintetico: muestra del dataset con drift inyectado a proposito (clientes mas
      jovenes, con mas consultas, peor score, menor salario, mas tendencia decreciente).
    """
    df = cargar_datos()
    if nombre == "temporal":
        return df[df["fecha_prestamo"] >= pd.Timestamp(CFG["fecha_corte_simulacion"])].copy()
    if nombre == "sintetico":
        rng = np.random.default_rng(CONFIG["random_state"])
        muestra = df.sample(2000, random_state=CONFIG["random_state"]).copy()
        muestra["edad_cliente"] = (muestra["edad_cliente"] - rng.integers(3, 9, len(muestra))).clip(lower=18)
        muestra["huella_consulta"] = muestra["huella_consulta"] + rng.integers(2, 6, len(muestra))
        muestra["puntaje_datacredito"] = muestra["puntaje_datacredito"] - rng.normal(45, 15, len(muestra))
        muestra["salario_cliente"] = (muestra["salario_cliente"] * rng.uniform(0.7, 0.9, len(muestra))).round()
        cambia = rng.random(len(muestra)) < 0.4
        muestra.loc[cambia, "tendencia_ingresos"] = "Decreciente"
        return muestra
    raise ValueError(f"Escenario desconocido: {nombre}")


# ---------------------------------------------------------------------------
# Deteccion
# ---------------------------------------------------------------------------
def detectar_drift(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Tabla con PSI, test estadistico y semaforo por variable."""
    validar_esquema(referencia)
    validar_esquema(actual)
    ref = normalizar_para_monitoreo(referencia)
    act = normalizar_para_monitoreo(actual)
    filas = []
    for col in COLS_NUMERICAS:
        psi = calcular_psi_numerico(ref[col], act[col], CFG["n_bins"])
        estad, p = test_ks(ref[col], act[col])
        filas.append({
            "feature": col, "tipo": "numerica", "psi": psi, "test": "KS", "estadistico": estad, "p_valor": p,
            "significativo": p < CFG["alpha"], "nivel": nivel_psi(psi),
            "ref_mediana": float(ref[col].median()), "act_mediana": float(act[col].median()),
            "ref_pct_nulos": float(ref[col].isna().mean()), "act_pct_nulos": float(act[col].isna().mean()),
        })
    for col in COLS_CATEGORICAS:
        psi = calcular_psi_categorico(ref[col], act[col])
        estad, p = test_chi2(ref[col], act[col])
        filas.append({
            "feature": col, "tipo": "categorica", "psi": psi, "test": "chi2", "estadistico": estad, "p_valor": p,
            "significativo": p < CFG["alpha"], "nivel": nivel_psi(psi),
            "ref_mediana": str(ref[col].mode().iat[0]), "act_mediana": str(act[col].mode().iat[0]),
            "ref_pct_nulos": 0.0, "act_pct_nulos": 0.0,
        })
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


def drift_prediccion(modelo, referencia: pd.DataFrame, actual: pd.DataFrame, umbral: float) -> dict:
    """El modelo ve el mundo distinto? PSI de las probabilidades y tasa de riesgo."""
    p_ref = modelo.predict_proba(referencia)[:, 1]
    p_act = modelo.predict_proba(actual)[:, 1]
    psi = calcular_psi_numerico(pd.Series(p_ref), pd.Series(p_act), CFG["n_bins"])
    return {
        "psi_probabilidad": psi,
        "nivel": nivel_psi(psi),
        "proba_media_ref": float(p_ref.mean()),
        "proba_media_act": float(p_act.mean()),
        "tasa_riesgo_ref": float((p_ref >= umbral).mean()),
        "tasa_riesgo_act": float((p_act >= umbral).mean()),
        "umbral": umbral,
    }


def drift_target(referencia: pd.DataFrame, actual: pd.DataFrame) -> dict | None:
    if TARGET not in actual.columns:
        return None
    mora_ref = float(1 - referencia[TARGET].mean())
    mora_act = float(1 - actual[TARGET].mean())
    return {
        "tasa_mora_ref": mora_ref, "tasa_mora_act": mora_act, "diferencia_pp": (mora_act - mora_ref) * 100,
        "nota": "Los creditos recientes pueden mostrar menos mora por censura (aun no vencieron), ver EDA 3.6.",
    }


def resumen_global(tabla: pd.DataFrame, pred: dict) -> dict:
    n_critico = int((tabla["nivel"] == "critico").sum())
    n_alerta = int((tabla["nivel"] == "alerta").sum())
    hay_drift = n_critico >= 1 or n_alerta >= CFG["max_alertas"] or pred["nivel"] == "critico"
    return {
        "hay_drift": bool(hay_drift),
        "estado": "DRIFT DETECTADO" if hay_drift else "SIN DRIFT RELEVANTE",
        "n_critico": n_critico,
        "n_alerta": n_alerta,
        "n_ok": int((tabla["nivel"] == "ok").sum()),
        "features_critico": tabla.loc[tabla["nivel"] == "critico", "feature"].tolist(),
        "features_alerta": tabla.loc[tabla["nivel"] == "alerta", "feature"].tolist(),
        "accion_recomendada": (
            "Revisar origen de datos y evaluar reentrenamiento" if hay_drift else "Ninguna, seguir monitoreando"
        ),
    }


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
def graficar(tabla: pd.DataFrame, referencia: pd.DataFrame, actual: pd.DataFrame, escenario: str) -> list[str]:
    colores = {"ok": "#2ca02c", "alerta": "#ff7f0e", "critico": "#d62728", "sin_datos": "gray"}
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(tabla["feature"], tabla["psi"], color=tabla["nivel"].map(colores))
    ax.axvline(CFG["psi_alerta"], ls="--", color="#ff7f0e", label=f"alerta ({CFG['psi_alerta']})")
    ax.axvline(CFG["psi_critico"], ls="--", color="#d62728", label=f"critico ({CFG['psi_critico']})")
    ax.invert_yaxis()
    ax.set_xlabel("PSI")
    ax.set_title(f"PSI por variable - escenario {escenario}")
    ax.legend()
    fig.tight_layout()
    ruta_psi = DRIFT_DIR / f"psi_{escenario}.png"
    fig.savefig(ruta_psi, dpi=120)
    plt.close(fig)

    top = [f for f in tabla["feature"].head(6) if f in COLS_NUMERICAS][:4]
    ref, act = normalizar_para_monitoreo(referencia), normalizar_para_monitoreo(actual)
    fig, axes = plt.subplots(1, len(top), figsize=(4.5 * len(top), 3.8))
    for ax, col in zip(np.atleast_1d(axes), top):
        tope = ref[col].quantile(0.99)
        sns.kdeplot(ref[col].clip(upper=tope).dropna(), ax=ax, label="referencia", fill=True, alpha=0.3)
        sns.kdeplot(act[col].clip(upper=tope).dropna(), ax=ax, label="actual", fill=True, alpha=0.3)
        psi = float(tabla.loc[tabla["feature"] == col, "psi"].iat[0])
        ax.set_title(f"{col}\nPSI {psi:.3f}", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle(f"Distribuciones referencia vs actual (mayor PSI) - {escenario}")
    fig.tight_layout()
    ruta_dist = DRIFT_DIR / f"distribuciones_{escenario}.png"
    fig.savefig(ruta_dist, dpi=120)
    plt.close(fig)
    return [ruta_psi.name, ruta_dist.name]


def reporte_markdown(rep: dict, tabla: pd.DataFrame) -> str:
    g = rep["resumen"]
    p = rep["prediccion"]
    icono = {"ok": "OK", "alerta": "ALERTA", "critico": "CRITICO", "sin_datos": "-"}
    lineas = [
        f"# Reporte de data drift - escenario `{rep['escenario']}`",
        "",
        f"Generado: {rep['generado']}  |  referencia: {rep['n_referencia']:,} filas  |  actual: {rep['n_actual']:,} filas",
        "",
        f"## Estado global: **{g['estado']}**",
        "",
        f"- Variables criticas (PSI >= {CFG['psi_critico']}): {g['n_critico']} {g['features_critico']}",
        f"- Variables en alerta (PSI >= {CFG['psi_alerta']}): {g['n_alerta']} {g['features_alerta']}",
        f"- Variables estables: {g['n_ok']}",
        f"- Drift de prediccion: PSI {p['psi_probabilidad']:.3f} ({icono[p['nivel']]}); tasa de riesgo "
        f"{p['tasa_riesgo_ref']:.1%} -> {p['tasa_riesgo_act']:.1%} al umbral {p['umbral']:.2f}",
        f"- Accion recomendada: {g['accion_recomendada']}",
        "",
    ]
    if rep.get("target"):
        t = rep["target"]
        lineas += [f"- Tasa de mora: {t['tasa_mora_ref']:.2%} -> {t['tasa_mora_act']:.2%} ({t['diferencia_pp']:+.2f} pp). {t['nota']}", ""]
    lineas += ["## Detalle por variable", "", "| Variable | Tipo | PSI | Nivel | Test | p-valor | Ref (mediana/moda) | Actual |", "|---|---|---|---|---|---|---|---|"]
    for _, r in tabla.iterrows():
        ref_v = f"{r['ref_mediana']:,.0f}" if r["tipo"] == "numerica" else r["ref_mediana"]
        act_v = f"{r['act_mediana']:,.0f}" if r["tipo"] == "numerica" else r["act_mediana"]
        lineas.append(f"| {r['feature']} | {r['tipo']} | {r['psi']:.3f} | {icono[r['nivel']]} | {r['test']} | {r['p_valor']:.3g} | {ref_v} | {act_v} |")
    lineas += ["", "Figuras: " + ", ".join(f"`{f}`" for f in rep["figuras"]), ""]
    return "\n".join(lineas)


def generar_reporte(referencia: pd.DataFrame, actual: pd.DataFrame, escenario: str, modelo, umbral: float) -> dict:
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    tabla = detectar_drift(referencia, actual)
    pred = drift_prediccion(modelo, referencia, actual, umbral)
    rep = {
        "escenario": escenario,
        "generado": datetime.now().isoformat(timespec="seconds"),
        "n_referencia": int(len(referencia)),
        "n_actual": int(len(actual)),
        "umbrales": {k: CFG[k] for k in ("psi_alerta", "psi_critico", "alpha", "max_alertas")},
        "resumen": resumen_global(tabla, pred),
        "prediccion": pred,
        "target": drift_target(referencia, actual),
        "features": tabla.to_dict(orient="records"),
        "figuras": graficar(tabla, referencia, actual, escenario),
    }
    (DRIFT_DIR / f"drift_report_{escenario}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    (DRIFT_DIR / f"drift_report_{escenario}.md").write_text(reporte_markdown(rep, tabla), encoding="utf-8")
    tabla.to_csv(DRIFT_DIR / f"drift_features_{escenario}.csv", index=False)
    return rep


def imprimir_resumen(rep: dict) -> None:
    g, p = rep["resumen"], rep["prediccion"]
    print(f"\n[{rep['escenario']}] {g['estado']} | ref {rep['n_referencia']:,} vs act {rep['n_actual']:,}")
    print(f"  critico={g['n_critico']} {g['features_critico']} | alerta={g['n_alerta']} {g['features_alerta']} | ok={g['n_ok']}")
    print(f"  prediccion: PSI {p['psi_probabilidad']:.3f} ({p['nivel']}), tasa riesgo {p['tasa_riesgo_ref']:.1%} -> {p['tasa_riesgo_act']:.1%}")
    if rep["target"]:
        t = rep["target"]
        print(f"  mora: {t['tasa_mora_ref']:.2%} -> {t['tasa_mora_act']:.2%} ({t['diferencia_pp']:+.2f} pp)")
    print(f"  -> {g['accion_recomendada']} | reporte: {DRIFT_DIR / ('drift_report_' + rep['escenario'] + '.md')}")


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitoreo de data drift del modelo de riesgo crediticio")
    parser.add_argument("--nuevos", type=Path, help="CSV con datos nuevos (mismas columnas crudas que Base_de_datos.csv)")
    parser.add_argument("--escenario", choices=["temporal", "sintetico", "ambos"], default="ambos",
                        help="Simulacion a usar cuando no se pasa --nuevos")
    parser.add_argument("--strict", action="store_true", help="Exit code 1 si se detecta drift (para CI / jobs)")
    args = parser.parse_args(argv)

    modelo = joblib.load(MODELS_DIR / "modelo_riesgo.joblib")
    umbral = float(json.loads(METRICS_PATH.read_text(encoding="utf-8"))["umbral_decision"])
    referencia = datos_referencia()

    if args.nuevos:
        ventanas = {"produccion": pd.read_csv(args.nuevos, parse_dates=["fecha_prestamo"]) if "fecha_prestamo" in
                    pd.read_csv(args.nuevos, nrows=0).columns else pd.read_csv(args.nuevos)}
    else:
        nombres = ["temporal", "sintetico"] if args.escenario == "ambos" else [args.escenario]
        ventanas = {n: simular_escenario(n) for n in nombres}

    hay_drift = False
    for nombre, actual in ventanas.items():
        ref = referencia
        if nombre == "temporal":
            # La referencia no debe contener la ventana que se evalua: solo train anterior al corte
            ref = referencia[referencia["fecha_prestamo"] < pd.Timestamp(CFG["fecha_corte_simulacion"])]
        rep = generar_reporte(ref, actual, nombre, modelo, umbral)
        imprimir_resumen(rep)
        hay_drift |= rep["resumen"]["hay_drift"]

    if "sintetico" in ventanas and not json.loads(
        (DRIFT_DIR / "drift_report_sintetico.json").read_text(encoding="utf-8"))["resumen"]["hay_drift"]:
        print("\nERROR: el detector no marco drift en el escenario sintetico con drift inyectado")
        return 2
    return 1 if (args.strict and hay_drift) else 0


if __name__ == "__main__":
    sys.exit(main())
