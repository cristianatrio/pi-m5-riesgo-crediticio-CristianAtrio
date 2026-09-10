# PI M5 - Modelo predictivo de riesgo crediticio

Proyecto Integrador del Modulo 5 (Data Science, Henry). Modelo de machine learning
que anticipa si un nuevo solicitante de credito pagara a tiempo, desplegado como API
(FastAPI + Docker) con monitoreo de data drift y una app Streamlit.

> Estado actual: **V1.1.0** - ingenieria de caracteristicas (`ft_engineering.py`) y primeros modelos
> entrenados, evaluados y comparados (`model_training_evaluation.py`). Modelo ganador: **Random Forest**,
> ROC-AUC 0,70 en test, serializado en `mlops_pipeline/models/modelo_riesgo.joblib`.

## Estructura del repositorio (no modificar: los pipelines de Jenkins dependen de ella)

```
mlops_pipeline/
├── src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   ├── model_monitoring.py
│   └── config.json            # parametros del proyecto (target, exclusiones, rangos, seed)
├── models/                    # modelo_riesgo.joblib (preprocesador + modelo), feature_names.json
├── reports/                   # metrics.json, comparacion_modelos.csv, importancia_variables.csv
│   └── figures/               # curvas ROC/PR, matrices de confusion, comparacion, importancias
└── data/                      # splits transformados (parquet, ignorados por git)
Base_de_datos.csv
requirements.txt
set_up.bat                     # crea el venv, instala requirements y registra el kernel
.gitignore
readme.md
```

## Ramas y versionado

| Rama | Rol |
|---|---|
| `developer` | Todo el desarrollo nuevo sale de aca |
| `certification` | Validacion antes de produccion (PR desde `developer`, aprobado por un par) |
| `master` | Produccion (PR desde `certification`) |

| Version | Contenido |
|---|---|
| V1.0.0 | Estructura de carpetas identica en las 3 ramas |
| V1.0.1 | `Cargar_datos.ipynb` y `comprension_eda.ipynb` |
| V1.0.2 | Correcciones del review del EDA (rango completo de `puntaje_datacredito`, etiquetas y conteos) |
| V1.1.0 | Ingenieria de caracteristicas + entrenamiento, evaluacion y seleccion de modelos |
| V1.2.0+ | Monitoreo, Streamlit, API, Docker |

## Setup local

```bat
set_up.bat
```

o manualmente:

```bash
py -3.12 -m venv pim5_riesgo_crediticio-venv
pim5_riesgo_crediticio-venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` fija versiones exactas (Python 3.12). Si se actualiza una dependencia, correr los dos scripts del
Avance 2 y verificar que `mlops_pipeline/reports/metrics.json` no cambie.

## Avance 1 - EDA (resumen)

- 10.763 creditos, 23 variables, target `Pago_atiempo` (1 = pago a tiempo). Se modela **mora = 1 - Pago_atiempo**.
- **Desbalance fuerte**: 4,75% de mora (511 casos). El accuracy no sirve; se evalua con ROC-AUC, PR-AUC, recall y F1 de la clase mora.
- **`puntaje` es fuga de informacion**: separa perfectamente las clases (todo moroso <= 62,7; todo cumplidor >= 63,8). Excluida.
- Senales legitimas: `puntaje_datacredito` (-), `huella_consulta` (+), `edad_cliente` (-), `plazo_meses` largo (+), ingresos estimados por la central (-), tendencia de ingresos decreciente (+), ausencia de dato en la central (+).
- Calidad: edades 121-123, salarios de miles de millones, `tendencia_ingresos` con numeros, `puntaje_datacredito` fuera de 150-950. Decisiones en `comprension_eda.ipynb`, seccion 5.

## Avance 2 - Ingenieria de caracteristicas y modelado

### `ft_engineering.py` (pipeline sklearn + feature-engine)

| Paso | Tecnica | Variables |
|---|---|---|
| Limpieza (`LimpiezaCredito`) | Valores imposibles -> nulo; drop `puntaje` y `fecha_prestamo` | edad >= 100, salario 0, score fuera de 150-950, tendencia no valida |
| Indicadores de faltante | `AddMissingIndicator` | promedio_ingresos_datacredito, saldo_mora, saldo_principal, saldo_mora_codeudor |
| Imputacion saldos | `ArbitraryNumberImputer(0)` (sin deuda reportada) | saldo_* |
| Imputacion numericas | `MeanMedianImputer(median)` (robusta a outliers) | edad, salario, puntaje_datacredito, promedio_ingresos |
| Imputacion categorica | `CategoricalImputer("Sin dato")` (el nulo es informativo) | tendencia_ingresos |
| Outliers | `Winsorizer(p99, cola derecha)` | montos y saldos |
| Features nuevas (`FeaturesCredito`) | ratios y flags de negocio | ratio_cuota_salario, ratio_deuda_salario, ratio_capital_salario, plazo_largo, tiene_mora_previa, sin_historial_financiero, creditos_total_sectores |
| Asimetria | `LogCpTransformer(log(x+1))` | montos, saldos y ratios |
| Categorias raras | `RareLabelEncoder(tol=0.3%)` | tipo_credito (6, 7, 68 -> Rare) |
| Encoding | `OneHotEncoder` | tipo_credito, tipo_laboral, tendencia_ingresos |
| Multicolinealidad | `DropCorrelatedFeatures(0.9)` | descarta 6 redundantes (saldo_total, creditos_total_sectores, dummies complementarias, etc.) |
| Escalado | `StandardScaler` | todas (necesario para la regresion logistica) |

Salida: 32 features. El preprocesador viaja dentro del pipeline del modelo (unico artefacto de inferencia:
`modelo_riesgo.joblib`), asi la API recibe datos crudos y no hay dos versiones que puedan divergir.
`LimpiezaCredito` valida el esquema de entrada (`COLUMNAS_REQUERIDAS`) y falla con un mensaje operativo si falta una columna.

```bash
python mlops_pipeline/src/ft_engineering.py
```

### `model_training_evaluation.py`

- Split estratificado 80/20 (`random_state=42`), `StratifiedKFold(5)`, un solo ajuste por fold.
- El preprocesador se ajusta dentro de cada fold (`Pipeline(preprocesador, modelo)`): sin fuga de imputaciones.
- Umbral de decision elegido con probabilidades out-of-fold del train (maximo F1). CV y test se reportan al mismo umbral.
  Limitacion: el umbral se elige sobre el mismo OOF con el que se reporta F1 en CV (sesgo optimista leve); el test, que
  nunca interviene en la eleccion, es la estimacion honesta.
- Seleccion por ROC-AUC en CV (calidad del ranking), desempate por PR-AUC y F1. El umbral es una decision operativa posterior.
- Importancia por permutacion sobre test: analisis post-hoc del ganador, no evidencia adicional de performance.
- Modelos: Dummy (piso), Regresion Logistica, Random Forest, HistGradientBoosting, XGBoost. Todos con compensacion del desbalance.
- LightGBM se descarto: la version 4.7 falla con numpy 2.5 en Windows (access violation). HistGradientBoosting es el
  gradient boosting con histogramas nativo de sklearn, mismo enfoque pero sin GOSS, sin soporte nativo de categoricas
  en este pipeline (ya van one-hot) y con menos hiperparametros. `requirements.txt` fija versiones para que la
  combinacion incompatible no vuelva a instalarse.

```bash
python mlops_pipeline/src/model_training_evaluation.py
```

### Resultados

CV y test al umbral optimizado de cada modelo. Clase positiva = mora.

| Modelo | CV ROC-AUC | CV PR-AUC | CV F1 | Test ROC-AUC | Test PR-AUC | Test recall | Test precision | Test F1 | Umbral |
|---|---|---|---|---|---|---|---|---|---|
| Dummy (siempre paga) | 0,500 | 0,048 | 0,00 | 0,500 | 0,047 | 0,00 | 0,00 | 0,00 | 0,50 |
| Regresion Logistica | 0,675 +- 0,043 | 0,146 | 0,19 | 0,687 | 0,161 | 0,26 | 0,19 | 0,22 | 0,69 |
| **Random Forest** | **0,678 +- 0,048** | 0,139 | 0,19 | **0,699** | 0,143 | 0,31 | 0,18 | **0,23** | 0,50 |
| HistGradientBoosting | 0,641 +- 0,037 | 0,126 | 0,18 | 0,699 | 0,167 | 0,32 | 0,15 | 0,21 | 0,56 |
| XGBoost | 0,657 +- 0,034 | 0,132 | 0,17 | 0,689 | 0,149 | 0,31 | 0,13 | 0,18 | 0,53 |

**Modelo elegido: Random Forest.** Criterio: mayor ROC-AUC medio en validacion cruzada (ranking robusto con
desbalance), desempate por PR-AUC y F1 en CV. No se usa accuracy porque el Dummy tendria 95%. Los cuatro modelos estan
dentro del margen de error entre si (std ~0,04); se prefiere Random Forest por ser el mas estable entre CV y
test, tener el mejor F1 en test y un umbral natural (0,50). La regresion logistica queda como alternativa
interpretable a un punto de distancia.

Matriz de confusion en test (umbral 0,50): TN 1.907 | FP 144 | FN 70 | TP 32. El modelo detecta 31% de las
moras marcando el 8% de los solicitantes; el PR-AUC de 0,14 triplica el azar (0,047).

Variables mas influyentes (importancia por permutacion sobre test, analisis post-hoc): `puntaje_datacredito`, `huella_consulta`,
`promedio_ingresos_datacredito`, `plazo_meses`, `edad_cliente`, `total_otros_prestamos`, `ratio_deuda_salario`.
Coinciden con el bivariable del EDA.

Figuras en `mlops_pipeline/reports/figures/`: `curvas_roc_pr.png`, `matrices_confusion.png`,
`comparacion_modelos.png`, `importancia_variables.png`. Metricas completas en `mlops_pipeline/reports/metrics.json`.
