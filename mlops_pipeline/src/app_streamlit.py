"""
App Streamlit del modelo de riesgo crediticio (PI M5).

Pestanas:
- Prediccion: formulario con las 20 variables crudas del solicitante -> probabilidad de mora,
  clasificacion con el umbral operativo y comparacion contra el perfil tipico de un pagador.
- Explicacion: variables mas influyentes del modelo (importancia por permutacion).
- Lote: subir un CSV con solicitantes, predecir en batch y descargar.
- Monitoreo: semaforo de data drift a partir de los reportes de model_monitoring.py.

Capa visual: tema claro / oscuro con toggle en el header (st.session_state + CSS con
variables de color), ancho maximo acotado, tarjetas con borde, resultado destacado.

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
# Tema (claro / oscuro) y CSS
# ---------------------------------------------------------------------------
PALETAS = {
    "claro": {
        "bg": "#f5f7fb", "surface": "#ffffff", "surface2": "#eef2f7", "text": "#0f172a", "muted": "#64748b",
        "border": "#e2e8f0", "primary": "#2563eb", "primary_text": "#ffffff",
        "ok": "#16a34a", "warn": "#d97706", "danger": "#dc2626", "shadow": "0 1px 2px rgba(15,23,42,.06)",
    },
    "oscuro": {
        "bg": "#0b1220", "surface": "#111a2e", "surface2": "#182238", "text": "#e5e7eb", "muted": "#94a3b8",
        "border": "#243047", "primary": "#3b82f6", "primary_text": "#ffffff",
        "ok": "#22c55e", "warn": "#f59e0b", "danger": "#ef4444", "shadow": "0 1px 2px rgba(0,0,0,.4)",
    },
}


def tema_actual() -> str:
    """Estado del toggle en session_state; 'claro' por defecto."""
    if "modo_oscuro" not in st.session_state:
        st.session_state["modo_oscuro"] = False
    return "oscuro" if st.session_state["modo_oscuro"] else "claro"


def aplicar_css(tema: str) -> None:
    """Inyecta las variables de color del tema y el estilo corporativo (ancho, tarjetas, inputs)."""
    p = PALETAS[tema]
    st.markdown(
        f"""
<style>
:root {{
  --bg: {p['bg']}; --surface: {p['surface']}; --surface2: {p['surface2']}; --text: {p['text']};
  --muted: {p['muted']}; --border: {p['border']}; --primary: {p['primary']}; --primary-text: {p['primary_text']};
  --ok: {p['ok']}; --warn: {p['warn']}; --danger: {p['danger']}; --shadow: {p['shadow']}; --radius: 16px;
}}
/* Lienzo y tipografia */
.stApp, [data-testid="stHeader"] {{ background: var(--bg); color: var(--text); }}
[data-testid="stHeader"] {{ border-bottom: 1px solid var(--border); }}
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {{ color: var(--text); }}
.stApp [data-testid="stCaptionContainer"] p, .stApp small {{ color: var(--muted); }}
/* Ancho maximo: layout wide sin oceanos de espacio */
.block-container {{ max-width: 1180px; padding-top: 1.25rem; padding-bottom: 2.5rem; }}
/* Ocultar cromo de Streamlit */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {{ display: none !important; }}
/* Tarjetas: containers con borde y expanders */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1rem 1.25rem; box-shadow: var(--shadow);
}}
[data-testid="stExpander"] {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; }}
[data-testid="stExpander"] summary {{ color: var(--text); font-weight: 600; }}
[data-testid="stExpander"] details {{ border: none; }}
/* Inputs */
.stApp label p {{ font-size: .85rem; font-weight: 600; color: var(--muted); }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stNumberInputContainer"] {{
  background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important;
  color: var(--text) !important;
}}
.stSelectbox > div > div, [data-baseweb="select"] > div > div {{
  background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important;
}}
.stSelectbox > div > div * {{ color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; }}
[data-baseweb="input"] input, [data-baseweb="select"] input, [data-baseweb="select"] div,
[data-testid="stNumberInputContainer"] input {{ color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; }}
[data-baseweb="select"] svg, [data-testid="stNumberInputContainer"] svg {{ fill: var(--muted); color: var(--muted); }}
[data-testid="stNumberInputContainer"] button {{ background: var(--surface2); color: var(--text); border-left: 1px solid var(--border); }}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {{ background: var(--surface) !important; }}
[data-baseweb="menu"] li {{ color: var(--text) !important; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--primary); border-color: var(--primary); }}
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {{ color: var(--muted); }}
[data-testid="stWidgetLabel"] {{ margin-bottom: .1rem; }}
.stNumberInput, .stSelectbox, .stSlider, .stCheckbox, .stToggle {{ margin-bottom: .35rem; }}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ padding: .6rem 1rem; border-radius: 10px 10px 0 0; color: var(--muted); }}
.stTabs [aria-selected="true"] {{ color: var(--primary); border-bottom: 2px solid var(--primary); }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--primary); }}
/* Botones */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
  border-radius: 12px; font-weight: 600; padding: .6rem 1.2rem; border: 1px solid var(--border);
  background: var(--surface2); color: var(--text);
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background: var(--primary); color: var(--primary-text); border-color: var(--primary);
}}
/* Metricas como tarjetas */
[data-testid="stMetric"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: .9rem 1.1rem; box-shadow: var(--shadow);
}}
[data-testid="stMetricLabel"] p {{
  color: var(--muted) !important; font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .03em;
}}
[data-testid="stMetricValue"] {{ color: var(--text); font-weight: 700; }}
/* Alertas y dataframes */
[data-testid="stAlert"] {{ border-radius: 12px; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
/* Banner de decision y barra de riesgo */
.banner {{ border-radius: var(--radius); padding: 1rem 1.25rem; font-weight: 700; font-size: 1.15rem; color: #fff; }}
.banner small {{ display: block; font-weight: 500; font-size: .9rem; opacity: .9; margin-top: .25rem; color: #fff; }}
.gauge {{ position: relative; height: 14px; border-radius: 999px; margin: 1.6rem 0 .4rem 0;
  background: linear-gradient(90deg, var(--ok) 0%, var(--warn) 50%, var(--danger) 100%); }}
.gauge .marker {{
  position: absolute; top: -7px; width: 4px; height: 28px; background: var(--text); border-radius: 2px; transform: translateX(-2px);
}}
.gauge .umbral {{ position: absolute; top: -12px; width: 0; height: 38px; border-left: 2px dashed var(--muted); }}
.gauge .lbl {{ position: absolute; top: -32px; transform: translateX(-50%); font-size: .75rem; color: var(--muted); white-space: nowrap; }}
.subtle {{ color: var(--muted); font-size: .85rem; }}
</style>
""",
        unsafe_allow_html=True,
    )


def estilizar(chart: alt.Chart, tema: str) -> alt.Chart:
    """Fondo transparente y colores de texto acordes al tema para los charts Altair."""
    p = PALETAS[tema]
    return (chart.configure(background="transparent")
            .configure_axis(labelColor=p["muted"], titleColor=p["muted"], gridColor=p["border"], domainColor=p["border"])
            .configure_legend(labelColor=p["text"], titleColor=p["muted"])
            .configure_view(strokeWidth=0))


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


# ---------------------------------------------------------------------------
# Logica
# ---------------------------------------------------------------------------
def predecir(modelo, df: pd.DataFrame) -> np.ndarray:
    """Probabilidad de mora. El pipeline recibe datos crudos y hace toda la limpieza."""
    return modelo.predict_proba(df[COLUMNAS_REQUERIDAS])[:, 1]


def nivel_riesgo(proba: float, umbral: float) -> tuple[str, str, str]:
    """(etiqueta, clave de color de la paleta, recomendacion)."""
    if proba >= umbral:
        return "ALTO RIESGO", "danger", "Rechazar o escalar a analisis manual con garantias adicionales."
    if proba >= umbral * 0.6:
        return "RIESGO MEDIO", "warn", "Aprobar con analisis adicional: verificar ingresos y reducir monto o plazo."
    return "BAJO RIESGO", "ok", "Aprobar segun politica vigente."


# ---------------------------------------------------------------------------
# Componentes de UI
# ---------------------------------------------------------------------------
def header(metricas: dict, umbral_modelo: float) -> str:
    """Titulo, resumen del modelo y toggle de tema. Devuelve el tema vigente tras el toggle."""
    c1, c2 = st.columns([5, 1], vertical_alignment="center")
    with c1:
        st.title("Modelo de riesgo crediticio")
        st.caption(
            f"Modelo en produccion: **{metricas['modelo_ganador']}** | ROC-AUC test {metricas['test']['roc_auc']:.3f} | "
            f"PR-AUC {metricas['test']['pr_auc']:.3f} | umbral operativo {umbral_modelo:.2f} | "
            f"tasa de mora historica {metricas['tasa_mora_train']:.2%}"
        )
    with c2:
        st.toggle("Modo oscuro", key="modo_oscuro")
    return tema_actual()


def formulario_solicitante(df_hist: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Formulario en 3 tarjetas + expander avanzado; defaults = mediana historica. Devuelve (solicitante, enviado)."""
    med = df_hist[COLUMNAS_REQUERIDAS].select_dtypes("number").median()
    with st.form("solicitud", border=False):
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1, st.container(border=True):
            st.markdown("#### Solicitante")
            edad = st.number_input("Edad", 18, 99, int(med["edad_cliente"]))
            tipo_laboral = st.selectbox("Tipo laboral", ["Empleado", "Independiente"])
            salario = st.number_input("Salario mensual", 0, 500_000_000, int(med["salario_cliente"]), step=100_000)
            otros_prestamos = st.number_input("Total otros prestamos", 0, 5_000_000_000, int(med["total_otros_prestamos"]), step=100_000)
        with c2, st.container(border=True):
            st.markdown("#### Credito solicitado")
            tipo_credito = st.selectbox("Tipo de credito", [4, 9, 10, 6], format_func=lambda v: f"Producto {v}")
            capital = st.number_input("Capital solicitado", 100_000, 100_000_000, int(med["capital_prestado"]), step=100_000)
            plazos = sorted(df_hist["plazo_meses"].unique().tolist())
            plazo = st.select_slider("Plazo (meses)", options=plazos, value=int(med["plazo_meses"]))
            cuota = st.number_input("Cuota pactada", 10_000, 10_000_000, int(med["cuota_pactada"]), step=10_000)
        with c3, st.container(border=True):
            st.markdown("#### Central de riesgo")
            score = st.slider("Puntaje Datacredito", 150, 950, int(med["puntaje_datacredito"]))
            huella = st.slider("Consultas recientes (huella)", 0, 30, int(med["huella_consulta"]))
            vigentes = st.slider("Creditos vigentes", 0, 60, int(med["cant_creditosvigentes"]))
            ingresos_central = st.number_input(
                "Ingreso promedio estimado", 0, 100_000_000, int(med["promedio_ingresos_datacredito"]), step=100_000
            )
            tendencia = st.selectbox("Tendencia de ingresos", TENDENCIAS, index=0)
            sin_ingreso_central = st.checkbox("La central no tiene estimacion de ingresos")
        with st.expander("Saldos y creditos por sector (avanzado)"):
            e1, e2, e3 = st.columns(3, gap="medium")
            with e1:
                saldo_total = st.number_input("Saldo total en central", 0, 10_000_000, int(med["saldo_total"]), step=1_000)
                saldo_principal = st.number_input("Saldo principal", 0, 10_000_000, int(med["saldo_principal"]), step=1_000)
            with e2:
                saldo_mora = st.number_input("Saldo en mora", 0, 1_000_000, 0, step=1_000)
                saldo_mora_cod = st.number_input("Saldo en mora como codeudor", 0, 1_000_000, 0, step=1_000)
            with e3:
                sec_fin = st.slider("Creditos sector financiero", 0, 50, int(med["creditos_sectorFinanciero"]))
                sec_coop = st.slider("Creditos sector cooperativo", 0, 15, int(med["creditos_sectorCooperativo"]))
                sec_real = st.slider("Creditos sector real", 0, 25, int(med["creditos_sectorReal"]))
        b1, _ = st.columns([1, 2])
        enviado = b1.form_submit_button("Evaluar solicitud", type="primary", width="stretch")

    solicitante = pd.DataFrame([{
        "tipo_credito": tipo_credito, "capital_prestado": float(capital), "plazo_meses": plazo, "edad_cliente": edad,
        "tipo_laboral": tipo_laboral, "salario_cliente": salario, "total_otros_prestamos": otros_prestamos,
        "cuota_pactada": cuota, "puntaje_datacredito": float(score), "cant_creditosvigentes": vigentes,
        "huella_consulta": huella, "saldo_mora": float(saldo_mora), "saldo_total": float(saldo_total),
        "saldo_principal": float(saldo_principal), "saldo_mora_codeudor": float(saldo_mora_cod),
        "creditos_sectorFinanciero": sec_fin, "creditos_sectorCooperativo": sec_coop, "creditos_sectorReal": sec_real,
        "promedio_ingresos_datacredito": np.nan if sin_ingreso_central else float(ingresos_central),
        "tendencia_ingresos": np.nan if (sin_ingreso_central or tendencia == "Sin dato") else tendencia,
    }])
    return solicitante, enviado


def barra_riesgo(proba: float, umbral: float) -> None:
    """Barra verde -> rojo con marcador del solicitante y linea punteada del umbral."""
    pos, pos_u = proba * 100, umbral * 100
    st.markdown(
        f'<div class="gauge"><span class="lbl" style="left:{pos:.1f}%">solicitante {proba:.1%}</span>'
        f'<div class="marker" style="left:{pos:.1f}%"></div>'
        f'<div class="umbral" style="left:{pos_u:.1f}%" title="umbral {umbral:.2f}"></div></div>'
        f'<div class="subtle">0% &nbsp;&middot;&nbsp; umbral de decision {umbral:.2f} (linea punteada) &nbsp;&middot;&nbsp; 100%</div>',
        unsafe_allow_html=True,
    )


def panel_resultado(res: dict, umbral: float, proba_media: float, mediana_pagadores: pd.Series, tema: str, tasa_mora: float) -> None:
    proba, solicitante = res["proba"], res["solicitante"]
    etiqueta, color, recomendacion = nivel_riesgo(proba, umbral)
    p = PALETAS[tema]

    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilidad de mora", f"{proba:.1%}")
    m2.metric("Vs. solicitante promedio", f"{(proba - proba_media) * 100:+.1f} pp", f"promedio {proba_media:.1%}", delta_color="off")
    m3.metric("Umbral aplicado", f"{umbral:.2f}",
              "minimo costo esperado" if abs(umbral - res["umbral_modelo"]) < 1e-9 else "ajustado manualmente", delta_color="off")

    st.markdown(
        f'<div class="banner" style="background:{p[color]}">{etiqueta} &middot; probabilidad de mora {proba:.1%}'
        f'<small>{recomendacion}</small></div>',
        unsafe_allow_html=True,
    )
    barra_riesgo(proba, umbral)
    st.caption(f"El modelo entrena con clases balanceadas: la probabilidad es un score de riesgo relativo, no la tasa real de mora "
               f"({tasa_mora:.2%}). Compara contra el umbral y contra el solicitante promedio.")

    with st.container(border=True):
        st.markdown("#### Solicitante vs. perfil tipico de un pagador")
        claves = ["puntaje_datacredito", "huella_consulta", "edad_cliente", "plazo_meses", "capital_prestado",
                  "salario_cliente", "total_otros_prestamos", "cant_creditosvigentes"]
        comp = pd.DataFrame({
            "variable": claves,
            "solicitante": [float(solicitante[k].iat[0]) for k in claves],
            "pagador_tipico": [float(mediana_pagadores[k]) for k in claves],
        })
        comp["ratio"] = (comp["solicitante"] / comp["pagador_tipico"].replace(0, np.nan)).fillna(0).clip(upper=3)
        st.dataframe(
            comp, hide_index=True, width="stretch",
            column_config={
                "variable": st.column_config.TextColumn("Variable"),
                "solicitante": st.column_config.NumberColumn("Solicitante", format="%,.0f"),
                "pagador_tipico": st.column_config.NumberColumn("Pagador tipico (mediana)", format="%,.0f"),
                "ratio": st.column_config.ProgressColumn("Ratio vs. tipico (tope 3x)", min_value=0, max_value=3, format="%.2fx"),
            },
        )


def tab_prediccion(modelo, df_hist, mediana_pagadores, proba_media, umbral_modelo, tasa_mora, tema):
    solicitante, enviado = formulario_solicitante(df_hist)
    u1, u2 = st.columns([2, 1], vertical_alignment="center")
    umbral = u1.slider("Umbral de decision", 0.01, 0.99, umbral_modelo, 0.01,
                       help="Por defecto el umbral de minimo costo esperado (perdida por mora vs. margen perdido). "
                            "Bajarlo detecta mas moras a costa de mas rechazos.")
    u2.markdown(f'<div class="subtle">Umbral del modelo: <b>{umbral_modelo:.2f}</b>. Se aplica sobre la ultima evaluacion.</div>',
                unsafe_allow_html=True)
    if enviado:
        st.session_state["resultado"] = {
            "proba": float(predecir(modelo, solicitante)[0]), "solicitante": solicitante, "umbral_modelo": umbral_modelo,
        }
    if "resultado" in st.session_state:
        panel_resultado(st.session_state["resultado"], umbral, proba_media, mediana_pagadores, tema, tasa_mora)
    else:
        st.info("Completa el formulario y presiona **Evaluar solicitud**.")


def tab_explicacion(tema: str):
    imp = cargar_importancias().head(15)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1, st.container(border=True):
        st.markdown("#### Importancia por permutacion")
        st.caption("Caida de ROC-AUC en test al mezclar cada variable. Las de arriba son las que mas usa el modelo.")
        chart = alt.Chart(imp).mark_bar(color=PALETAS[tema]["primary"], cornerRadiusEnd=4).encode(
            x=alt.X("importancia:Q", title="caida de ROC-AUC"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importancia:Q", format=".4f")],
        ).properties(height=440)
        st.altair_chart(estilizar(chart, tema), width="stretch", theme=None)
    with c2, st.container(border=True):
        st.markdown("#### Lectura de negocio")
        st.markdown(
            "- **puntaje_datacredito** y **huella_consulta** explican la mayor parte: "
            "score bajo y muchas consultas recientes elevan la mora.\n"
            "- **plazo_meses** / **plazo_largo**: creditos a 24-36 meses triplican la mora de los de 9-12.\n"
            "- **edad_cliente**: los menores de 31 duplican la mora de los de 40-55.\n"
            "- **ratio_deuda_salario** y **promedio_ingresos_datacredito**: capacidad de pago e historial en la central.\n"
            "- **puntaje** (score interno) fue excluido por fuga de informacion: se calculaba despues del resultado."
        )


def tab_lote(modelo, umbral_modelo: float):
    with st.container(border=True):
        st.markdown("#### Prediccion en lote")
        st.caption("CSV con las mismas columnas crudas que `Base_de_datos.csv` (con o sin `Pago_atiempo`).")
        archivo = st.file_uploader("CSV de solicitantes", type="csv", label_visibility="collapsed")
    if archivo is None:
        return
    lote = pd.read_csv(archivo)
    try:
        validar_esquema(lote)
    except (ValueError, TypeError) as e:
        st.error(str(e))
        return
    lote["probabilidad_mora"] = predecir(modelo, lote)
    lote["decision"] = np.where(lote["probabilidad_mora"] >= umbral_modelo, "riesgo", "aprobar")
    c1, c2, c3 = st.columns(3)
    c1.metric("Solicitudes", f"{len(lote):,}")
    c2.metric("Marcadas como riesgo", f"{(lote['decision'] == 'riesgo').mean():.1%}")
    c3.metric("Probabilidad media", f"{lote['probabilidad_mora'].mean():.1%}")
    st.dataframe(
        lote.sort_values("probabilidad_mora", ascending=False).head(200), width="stretch", hide_index=True,
        column_config={"probabilidad_mora": st.column_config.ProgressColumn("Prob. mora", min_value=0, max_value=1, format="%.1%")},
    )
    st.download_button("Descargar resultados (CSV)", lote.to_csv(index=False).encode("utf-8"), "predicciones.csv", "text/csv")


def tab_monitoreo(tema: str):
    reportes = cargar_drift()
    if not reportes:
        st.warning("No hay reportes. Ejecuta `python mlops_pipeline/src/model_monitoring.py`.")
        return
    s1, _ = st.columns([1, 2])
    escenario = s1.selectbox("Ventana evaluada", list(reportes), format_func=lambda k: f"{k} ({reportes[k]['n_actual']:,} filas)")
    rep = reportes[escenario]
    g, pr = rep["resumen"], rep["prediccion"]
    (st.error if g["hay_drift"] else st.success)(f"**{g['estado']}** - {g['accion_recomendada']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variables criticas", g["n_critico"])
    c2.metric("Variables en alerta", g["n_alerta"])
    c3.metric("PSI de la prediccion", f"{pr['psi_probabilidad']:.3f}", pr["nivel"], delta_color="off")
    c4.metric("Tasa de riesgo", f"{pr['tasa_riesgo_act']:.1%}", f"{(pr['tasa_riesgo_act'] - pr['tasa_riesgo_ref']) * 100:+.1f} pp",
              delta_color="inverse")
    if rep.get("target"):
        t = rep["target"]
        st.caption(f"Tasa de mora: {t['tasa_mora_ref']:.2%} -> {t['tasa_mora_act']:.2%} ({t['diferencia_pp']:+.2f} pp). {t['nota']}")

    feats = pd.DataFrame(rep["features"])
    p = PALETAS[tema]
    escala = alt.Scale(domain=["ok", "alerta", "critico", "sin_datos"], range=[p["ok"], p["warn"], p["danger"], p["muted"]])
    with st.container(border=True):
        st.markdown("#### PSI por variable")
        chart = alt.Chart(feats).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("psi:Q", title="PSI"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color("nivel:N", scale=escala, legend=alt.Legend(title="nivel")),
            tooltip=["feature", "tipo", alt.Tooltip("psi:Q", format=".3f"), "test", alt.Tooltip("p_valor:Q", format=".2e"), "nivel"],
        ).properties(height=520)
        st.altair_chart(estilizar(chart, tema), width="stretch", theme=None)

    with st.container(border=True):
        st.markdown("#### Detalle por variable")
        tabla = feats[["feature", "tipo", "psi", "nivel", "test", "estadistico", "p_valor", "ref_mediana", "act_mediana"]].copy()
        # Medianas: numero para numericas, moda textual para categoricas -> texto formateado (Arrow no mezcla tipos)
        for col in ("ref_mediana", "act_mediana"):
            tabla[col] = tabla[col].map(lambda v: f"{v:,.2f}" if isinstance(v, (int, float)) else str(v))
        st.dataframe(
            tabla, hide_index=True, width="stretch", height=600,
            column_config={
                "feature": st.column_config.TextColumn("Variable"),
                "tipo": st.column_config.TextColumn("Tipo"),
                "psi": st.column_config.NumberColumn("PSI", format="%.4f"),
                "nivel": st.column_config.TextColumn("Nivel"),
                "test": st.column_config.TextColumn("Test"),
                "estadistico": st.column_config.NumberColumn("Estadistico", format="%.4f"),
                "p_valor": st.column_config.NumberColumn("p-valor", format="%.2e"),
                "ref_mediana": st.column_config.TextColumn("Referencia (mediana / moda)"),
                "act_mediana": st.column_config.TextColumn("Actual (mediana / moda)"),
            },
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    aplicar_css(tema_actual())
    modelo = cargar_modelo()
    metricas = cargar_metricas()
    df_hist, mediana_pagadores, proba_media = cargar_referencia()
    umbral_modelo = float(metricas["umbral_decision"])

    tema = header(metricas, umbral_modelo)
    if tema != st.session_state.get("_tema_aplicado"):
        aplicar_css(tema)  # el toggle cambio en este rerun: reinyectar la paleta correcta
    st.session_state["_tema_aplicado"] = tema

    t_pred, t_expl, t_lote, t_mon = st.tabs(["Prediccion", "Explicacion del modelo", "Lote (CSV)", "Monitoreo de drift"])
    with t_pred:
        tab_prediccion(modelo, df_hist, mediana_pagadores, proba_media, umbral_modelo, metricas["tasa_mora_train"], tema)
    with t_expl:
        tab_explicacion(tema)
    with t_lote:
        tab_lote(modelo, umbral_modelo)
    with t_mon:
        tab_monitoreo(tema)


if __name__ == "__main__":
    main()
