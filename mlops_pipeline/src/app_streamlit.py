"""
App Streamlit del modelo de riesgo crediticio (PI M5).

Pestanas:
- Prediccion: formulario con las 20 variables crudas del solicitante -> probabilidad de mora,
  clasificacion con el umbral operativo y comparacion contra el perfil tipico de un pagador.
- Explicacion: variables mas influyentes del modelo (importancia por permutacion).
- Lote: subir un CSV con solicitantes, predecir en batch y descargar.
- Monitoreo: semaforo de data drift a partir de los reportes de model_monitoring.py.

Uso::

    streamlit run mlops_pipeline/src/app_streamlit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import COLUMNAS_REQUERIDAS, CONFIG, MODELS_DIR, RAIZ, TARGET, cargar_datos, validar_esquema  # noqa: E402

REPORTS_DIR = RAIZ / "mlops_pipeline" / "reports"
DRIFT_DIR = REPORTS_DIR / "drift"
TENDENCIAS = CONFIG["categorias_tendencia"] + ["Sin dato"]

st.set_page_config(page_title="Riesgo crediticio - PI M5", page_icon=":bank:", layout="wide")


# ---------------------------------------------------------------------------
# Carga de artefactos (cacheada)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELS_DIR / "modelo_riesgo.joblib")


@st.cache_data
def cargar_metricas() -> dict:
    return json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data
def cargar_importancias() -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / "importancia_variables.csv")


@st.cache_data
def cargar_referencia() -> tuple[pd.DataFrame, pd.Series, float]:
    """Dataset historico: defaults del formulario, perfil tipico de pagadores y probabilidad media del modelo.

    El modelo entrena con clases balanceadas, asi que sus probabilidades no son la tasa real de mora (4,75%):
    la referencia justa para un solicitante es la probabilidad media que el modelo asigna al historico.
    """
    df = cargar_datos()
    pagadores = df[df[TARGET] == 1]
    proba_media = float(cargar_modelo().predict_proba(df[COLUMNAS_REQUERIDAS])[:, 1].mean())
    return df, pagadores[COLUMNAS_REQUERIDAS].select_dtypes("number").median(), proba_media


@st.cache_data
def cargar_drift() -> dict[str, dict]:
    reportes = {}
    for ruta in sorted(DRIFT_DIR.glob("drift_report_*.json")):
        reportes[ruta.stem.replace("drift_report_", "")] = json.loads(ruta.read_text(encoding="utf-8"))
    return reportes


modelo = cargar_modelo()
metricas = cargar_metricas()
df_hist, mediana_pagadores, PROBA_MEDIA_HIST = cargar_referencia()
UMBRAL_MODELO = float(metricas["umbral_decision"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def predecir(df: pd.DataFrame) -> np.ndarray:
    """Probabilidad de mora. El pipeline recibe datos crudos y hace toda la limpieza."""
    return modelo.predict_proba(df[COLUMNAS_REQUERIDAS])[:, 1]


def nivel_riesgo(proba: float, umbral: float) -> tuple[str, str]:
    if proba >= umbral:
        return "ALTO RIESGO - revisar / rechazar", "red"
    if proba >= umbral * 0.6:
        return "RIESGO MEDIO - analisis adicional", "orange"
    return "BAJO RIESGO - aprobar", "green"


def formulario_solicitante() -> pd.DataFrame:
    """Formulario en 3 grupos; defaults = mediana / moda del historico."""
    med = df_hist[COLUMNAS_REQUERIDAS].select_dtypes("number").median()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Solicitante")
        edad = st.number_input("Edad", 18, 99, int(med["edad_cliente"]))
        tipo_laboral = st.selectbox("Tipo laboral", ["Empleado", "Independiente"])
        salario = st.number_input("Salario mensual", 0, 500_000_000, int(med["salario_cliente"]), step=100_000)
        otros_prestamos = st.number_input("Total otros prestamos", 0, 5_000_000_000, int(med["total_otros_prestamos"]), step=100_000)
    with c2:
        st.subheader("Credito solicitado")
        tipo_credito = st.selectbox("Tipo de credito (codigo)", [4, 9, 10, 6], format_func=lambda v: f"Producto {v}")
        capital = st.number_input("Capital solicitado", 100_000, 100_000_000, int(med["capital_prestado"]), step=100_000)
        plazo = st.select_slider("Plazo (meses)", options=sorted(df_hist["plazo_meses"].unique().tolist()), value=int(med["plazo_meses"]))
        cuota = st.number_input("Cuota pactada", 10_000, 10_000_000, int(med["cuota_pactada"]), step=10_000)
    with c3:
        st.subheader("Central de riesgo")
        score = st.slider("Puntaje Datacredito", 150, 950, int(med["puntaje_datacredito"]))
        huella = st.slider("Consultas recientes (huella)", 0, 30, int(med["huella_consulta"]))
        vigentes = st.slider("Creditos vigentes", 0, 60, int(med["cant_creditosvigentes"]))
        sec_fin = st.slider("Creditos sector financiero", 0, 50, int(med["creditos_sectorFinanciero"]))
        sec_coop = st.slider("Creditos sector cooperativo", 0, 15, int(med["creditos_sectorCooperativo"]))
        sec_real = st.slider("Creditos sector real", 0, 25, int(med["creditos_sectorReal"]))
        saldo_total = st.number_input("Saldo total en central", 0, 10_000_000, int(med["saldo_total"]), step=1_000)
        saldo_principal = st.number_input("Saldo principal", 0, 10_000_000, int(med["saldo_principal"]), step=1_000)
        saldo_mora = st.number_input("Saldo en mora", 0, 1_000_000, 0, step=1_000)
        saldo_mora_cod = st.number_input("Saldo en mora como codeudor", 0, 1_000_000, 0, step=1_000)
        ingresos_central = st.number_input("Ingreso promedio estimado (central)", 0, 100_000_000, int(med["promedio_ingresos_datacredito"]), step=100_000)
        sin_ingreso_central = st.checkbox("La central no tiene estimacion de ingresos")
        tendencia = st.selectbox("Tendencia de ingresos", TENDENCIAS, index=0)

    return pd.DataFrame([{
        "tipo_credito": tipo_credito, "capital_prestado": float(capital), "plazo_meses": plazo, "edad_cliente": edad,
        "tipo_laboral": tipo_laboral, "salario_cliente": salario, "total_otros_prestamos": otros_prestamos,
        "cuota_pactada": cuota, "puntaje_datacredito": float(score), "cant_creditosvigentes": vigentes,
        "huella_consulta": huella, "saldo_mora": float(saldo_mora), "saldo_total": float(saldo_total),
        "saldo_principal": float(saldo_principal), "saldo_mora_codeudor": float(saldo_mora_cod),
        "creditos_sectorFinanciero": sec_fin, "creditos_sectorCooperativo": sec_coop, "creditos_sectorReal": sec_real,
        "promedio_ingresos_datacredito": np.nan if sin_ingreso_central else float(ingresos_central),
        "tendencia_ingresos": np.nan if (sin_ingreso_central or tendencia == "Sin dato") else tendencia,
    }])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Modelo de riesgo crediticio")
st.caption(
    f"Modelo en produccion: **{metricas['modelo_ganador']}** | ROC-AUC test {metricas['test']['roc_auc']:.3f} | "
    f"PR-AUC {metricas['test']['pr_auc']:.3f} | umbral operativo {UMBRAL_MODELO:.2f} | "
    f"tasa de mora historica {metricas['tasa_mora_train']:.2%}"
)

tab_pred, tab_expl, tab_lote, tab_mon = st.tabs(["Prediccion", "Explicacion del modelo", "Lote (CSV)", "Monitoreo de drift"])

# --- Prediccion ---------------------------------------------------------------
with tab_pred:
    solicitante = formulario_solicitante()
    umbral = st.slider("Umbral de decision", 0.05, 0.95, UMBRAL_MODELO, 0.01,
                       help="Por defecto el umbral optimizado por F1 en entrenamiento. Bajarlo detecta mas moras a costa de mas rechazos.")
    if st.button("Evaluar solicitud", type="primary"):
        proba = float(predecir(solicitante)[0])
        etiqueta, color = nivel_riesgo(proba, umbral)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidad de mora", f"{proba:.1%}",
                      delta=f"{(proba - PROBA_MEDIA_HIST) * 100:+.1f} pp vs. solicitante promedio ({PROBA_MEDIA_HIST:.1%})",
                      delta_color="inverse")
            st.caption("El modelo entrena con clases balanceadas: la probabilidad es un score de riesgo relativo, "
                       f"no la tasa real de mora ({metricas['tasa_mora_train']:.2%}). Compara contra el umbral.")
            st.markdown(f"### :{color}[{etiqueta}]")
            st.progress(min(proba / max(umbral * 2, 1e-6), 1.0), text=f"umbral {umbral:.2f}")
        with c2:
            st.markdown("**Solicitante vs. perfil tipico de un pagador (mediana)**")
            claves = ["puntaje_datacredito", "huella_consulta", "edad_cliente", "plazo_meses", "capital_prestado",
                      "salario_cliente", "total_otros_prestamos", "cant_creditosvigentes"]
            comp = pd.DataFrame({
                "variable": claves,
                "solicitante": [float(solicitante[k].iat[0]) for k in claves],
                "pagador tipico": [float(mediana_pagadores[k]) for k in claves],
            })
            comp["ratio vs. tipico"] = (comp["solicitante"] / comp["pagador tipico"].replace(0, np.nan)).round(2)
            st.dataframe(comp.style.format({"solicitante": "{:,.0f}", "pagador tipico": "{:,.0f}"}), hide_index=True, width="stretch")

# --- Explicacion --------------------------------------------------------------
with tab_expl:
    imp = cargar_importancias().head(15)
    st.markdown("**Importancia por permutacion** (caida de ROC-AUC en test al mezclar cada variable). "
                "Las variables de arriba son las que mas usa el modelo para separar pagadores de morosos.")
    chart = (alt.Chart(imp).mark_bar().encode(
        x=alt.X("importancia:Q", title="caida de ROC-AUC"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", alt.Tooltip("importancia:Q", format=".4f")],
    ).properties(height=420))
    st.altair_chart(chart, width="stretch")
    st.markdown(
        "- `puntaje_datacredito` y `huella_consulta` explican la mayor parte: score bajo y muchas consultas recientes elevan la mora.\n"
        "- `plazo_meses` y `plazo_largo`: creditos a 24-36 meses triplican la mora de los de 9-12.\n"
        "- `edad_cliente`: los menores de 31 duplican la mora de los de 40-55.\n"
        "- `ratio_deuda_salario` y `promedio_ingresos_datacredito`: capacidad de pago e historial en la central.\n"
        "- `puntaje` (score interno) fue **excluido** por fuga de informacion: se calculaba despues del resultado."
    )

# --- Lote ---------------------------------------------------------------------
with tab_lote:
    st.markdown("Subi un CSV con las mismas columnas crudas que `Base_de_datos.csv` (con o sin `Pago_atiempo`).")
    archivo = st.file_uploader("CSV de solicitantes", type="csv")
    if archivo is not None:
        lote = pd.read_csv(archivo)
        try:
            validar_esquema(lote)
        except (ValueError, TypeError) as e:
            st.error(str(e))
        else:
            lote["probabilidad_mora"] = predecir(lote)
            lote["decision"] = np.where(lote["probabilidad_mora"] >= UMBRAL_MODELO, "riesgo", "aprobar")
            c1, c2, c3 = st.columns(3)
            c1.metric("Solicitudes", f"{len(lote):,}")
            c2.metric("Marcadas como riesgo", f"{(lote['decision'] == 'riesgo').mean():.1%}")
            c3.metric("Probabilidad media", f"{lote['probabilidad_mora'].mean():.1%}")
            st.dataframe(lote.sort_values("probabilidad_mora", ascending=False).head(200), width="stretch")
            st.download_button("Descargar resultados", lote.to_csv(index=False).encode("utf-8"),
                               "predicciones.csv", "text/csv")

# --- Monitoreo ----------------------------------------------------------------
with tab_mon:
    reportes = cargar_drift()
    if not reportes:
        st.warning("No hay reportes. Ejecuta `python mlops_pipeline/src/model_monitoring.py`.")
    else:
        escenario = st.selectbox("Ventana evaluada", list(reportes), format_func=lambda k: f"{k} ({reportes[k]['n_actual']:,} filas)")
        rep = reportes[escenario]
        g, p = rep["resumen"], rep["prediccion"]
        color = "red" if g["hay_drift"] else "green"
        st.markdown(f"### :{color}[{g['estado']}]  \n{g['accion_recomendada']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variables criticas", g["n_critico"])
        c2.metric("Variables en alerta", g["n_alerta"])
        c3.metric("PSI de la prediccion", f"{p['psi_probabilidad']:.3f}", p["nivel"])
        c4.metric("Tasa de riesgo", f"{p['tasa_riesgo_act']:.1%}", f"{(p['tasa_riesgo_act'] - p['tasa_riesgo_ref']) * 100:+.1f} pp", delta_color="inverse")
        if rep.get("target"):
            t = rep["target"]
            st.caption(f"Tasa de mora: {t['tasa_mora_ref']:.2%} -> {t['tasa_mora_act']:.2%} ({t['diferencia_pp']:+.2f} pp). {t['nota']}")
        feats = pd.DataFrame(rep["features"])
        # Columnas mixtas (numero / moda textual) -> texto para que Arrow las serialice
        for col in ("ref_mediana", "act_mediana"):
            feats[col] = feats[col].map(lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) else str(v))
        escala = alt.Scale(domain=["ok", "alerta", "critico", "sin_datos"], range=["#2ca02c", "#ff7f0e", "#d62728", "gray"])
        chart = (alt.Chart(feats).mark_bar().encode(
            x=alt.X("psi:Q", title="PSI"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color("nivel:N", scale=escala, legend=alt.Legend(title="nivel")),
            tooltip=["feature", "tipo", alt.Tooltip("psi:Q", format=".3f"), "test", alt.Tooltip("p_valor:Q", format=".3g"), "nivel"],
        ).properties(height=520))
        st.altair_chart(chart, width="stretch")
        st.dataframe(feats[["feature", "tipo", "psi", "nivel", "test", "p_valor", "ref_mediana", "act_mediana"]],
                     hide_index=True, width="stretch")
