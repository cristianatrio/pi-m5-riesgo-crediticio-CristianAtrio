# PI M5 - Modelo predictivo de riesgo crediticio

Proyecto Integrador del Modulo 5 (Data Science, Henry). Modelo de machine learning
que anticipa si un nuevo solicitante de credito pagara a tiempo, desplegado como API
(FastAPI + Docker) con monitoreo de data drift y una app Streamlit.

## Caso de negocio

Una financiera otorga creditos de consumo de corto plazo (mediana 1,9 M a 10 meses). El 4,75% de los
creditos no se paga a tiempo y cada mora cuesta capital, cobranza y provisiones. Hoy la decision se apoya en
el score de la central de riesgo y en reglas manuales. El objetivo es un modelo que, con los datos disponibles
**en el momento de la solicitud** (solicitante, producto e historial en la central), estime la probabilidad de
mora de cada nuevo cliente para priorizar analisis, ajustar montos o rechazar.

Restricciones: estructura de carpetas fija (pipelines de Jenkins), flujo de ramas `developer -> certification ->
master` con PR aprobado por un par, y todo reproducible desde el repo.

Resultado: el modelo detecta el 31% de las moras marcando el 8% de las solicitudes (test), con PR-AUC que
triplica el azar. Las variables que mas pesan son el score de la central, las consultas recientes, el plazo
y la edad. Un score interno (`puntaje`) fue descartado por contener el resultado.

> Estado actual: **V1.3.0** - API REST con FastAPI (`model_deploy.py`), imagen Docker y pruebas
> automatizadas. Los 4 avances completos. Modelo en produccion: **Random Forest**, ROC-AUC 0,70 en test.

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
│   ├── model_deploy.py        # API FastAPI: /health, /model/info, /predict, /predict/batch
│   ├── app_streamlit.py       # app de prediccion, explicacion, lote y monitoreo
│   └── config.json            # parametros del proyecto (target, exclusiones, rangos, seed, umbrales de drift)
├── models/                    # modelo_riesgo.joblib (preprocesador + modelo), feature_names.json
├── reports/                   # metrics.json, comparacion_modelos.csv, importancia_variables.csv
│   ├── figures/               # curvas ROC/PR, matrices de confusion, comparacion, importancias
│   └── drift/                 # reportes de drift (json, md, csv) y figuras PSI por escenario
└── data/                      # splits transformados (parquet, ignorados por git)
.github/workflows/ci.yml       # CI: compila, feature engineering, monitoreo, pytest, build y prueba de la imagen Docker
tests/test_api.py              # pruebas de la API y del contrato del pipeline (pytest)
Dockerfile                     # imagen de la API (python:3.12-slim, usuario no root, healthcheck)
.dockerignore
requirements-api.txt           # dependencias minimas de la imagen
Base_de_datos.csv
requirements.txt               # dependencias completas de desarrollo y CI
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
| V1.2.0 | Monitoreo de data drift, app Streamlit, CI con GitHub Actions |
| V1.3.0 | API FastAPI, Dockerfile, pruebas con pytest, CI con build de la imagen |

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

## Avance 3 - Monitoreo de data drift, app Streamlit y CI

### `model_monitoring.py`

Compara una ventana de datos nuevos contra la referencia de entrenamiento (train del split oficial) sobre las
20 columnas crudas que recibe la API. Implementacion propia con numpy / scipy (sin `evidently`: dependencias
pesadas, incompatible con pandas 3 y menos transparente para auditar).

| Tipo de variable | Metrica | Test estadistico |
|---|---|---|
| Numericas (17) | PSI con 10 bins por cuantiles de la referencia | Kolmogorov-Smirnov 2 muestras |
| Categoricas (3) | PSI sobre frecuencias | Chi-cuadrado |
| Prediccion | PSI de `predict_proba` + tasa de solicitudes marcadas como riesgo | - |
| Target (si existe) | Tasa de mora referencia vs. actual | - |

Semaforo por variable: PSI < 0,1 ok, 0,1 a 0,25 alerta, > 0,25 critico. Alerta global si hay >= 1 critico o
>= 3 alertas. Salida: `reports/drift/drift_report_<escenario>.{json,md}`, `drift_features_<escenario>.csv`,
figuras PSI y distribuciones.

```bash
python mlops_pipeline/src/model_monitoring.py                        # simula ambos escenarios
python mlops_pipeline/src/model_monitoring.py --nuevos ventana.csv   # datos reales
python mlops_pipeline/src/model_monitoring.py --nuevos ventana.csv --strict   # exit 1 si hay drift (jobs / CI)
```

Escenarios simulados (no hay datos de produccion todavia):

| Escenario | Ventana | Resultado |
|---|---|---|
| `temporal` | Creditos desembolsados desde 2025-10-01 (825) vs. train anterior al corte | **Drift detectado**: `promedio_ingresos_datacredito` critico (PSI 0,58), `capital_prestado`, `salario_cliente` y `total_otros_prestamos` en alerta. El mix de productos cambio (capital mediano +33%). La prediccion se mantiene estable (PSI 0,04) y la mora baja a 3,5% por censura. |
| `sintetico` | 2.000 creditos con drift inyectado (clientes mas jovenes, mas consultas, score -45, salario -20%, mas tendencia decreciente) | **Drift detectado**: 3 criticos, 2 alertas, PSI de la prediccion 0,68, tasa de riesgo 9% -> 24%. Es el autotest del detector: el script devuelve exit 2 si no lo marca. |

### `app_streamlit.py`

```bash
streamlit run mlops_pipeline/src/app_streamlit.py
```

- **Prediccion**: formulario con las 20 variables crudas (solicitante, credito, central de riesgo), valores por
  defecto = mediana historica, opcion "la central no tiene dato". Devuelve probabilidad de mora, clasificacion
  (bajo / medio / alto) con el umbral operativo de `metrics.json` ajustable, y comparacion contra el pagador tipico.
- **Explicacion**: importancia por permutacion del modelo y lectura de negocio de cada variable.
- **Lote**: subir un CSV de solicitantes, validacion de esquema, prediccion batch y descarga.
- **Monitoreo**: semaforo de drift por variable y drift de prediccion desde los reportes de `model_monitoring.py`.

### CI/CD (`.github/workflows/ci.yml`)

En cada push a `developer` / `certification` / `master` y en cada PR: instala `requirements.txt` fijado,
compila los scripts, corre `ft_engineering.py`, corre el monitoreo en ambos escenarios (falla si el detector no
marca el drift sintetico), hace un smoke test del modelo serializado y publica los reportes de drift como
artefacto. Es la validacion automatica previa al merge; el despliegue (Docker) se agrega en el Avance 4.

## Avance 4 - Despliegue: API FastAPI y Docker

### `model_deploy.py`

| Metodo | Ruta | Que hace |
|---|---|---|
| GET | `/health` | Estado del servicio y del modelo (lo usa el `HEALTHCHECK` de Docker y el CI) |
| GET | `/model/info` | Modelo, umbral, metricas de CV y test, columnas requeridas, features, top features |
| POST | `/predict` | Un solicitante -> `probabilidad_mora`, `clase`, `nivel`, `decision`, `umbral` |
| POST | `/predict/batch` | Hasta 1.000 solicitantes -> predicciones + resumen |
| GET | `/docs` | Swagger UI con el ejemplo cargado |

- Entrada validada con Pydantic (`Solicitante`): 20 campos con los mismos nombres que `Base_de_datos.csv`, rangos,
  `tipo_laboral` y `tendencia_ingresos` como literales, `extra="forbid"` (un campo desconocido devuelve 422).
  `puntaje_datacredito`, saldos, `promedio_ingresos_datacredito` y `tendencia_ingresos` aceptan `null`: el pipeline imputa.
- El modelo y el umbral se cargan una vez en el `lifespan`. Si faltan artefactos la API responde 503.
- Errores: 422 (validacion), 400 (esquema rechazado por el pipeline), 500 (fallo de inferencia, sin traza al cliente).
- La API no duplica reglas de limpieza: manda datos crudos al pipeline serializado (`modelo_riesgo.joblib`).

```bash
python -m uvicorn model_deploy:app --app-dir mlops_pipeline/src --reload --port 8000
```

### Prueba del endpoint (resultado real)

```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","modelo":"Random Forest","version_api":"1.3.0","umbral":0.4962}
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"tipo_credito":4,"capital_prestado":1921920,"plazo_meses":10,"edad_cliente":42,"tipo_laboral":"Empleado","salario_cliente":3000000,"total_otros_prestamos":1000000,"cuota_pactada":182863,"puntaje_datacredito":791,"cant_creditosvigentes":5,"huella_consulta":4,"saldo_mora":0,"saldo_total":16178,"saldo_principal":14442,"saldo_mora_codeudor":0,"creditos_sectorFinanciero":2,"creditos_sectorCooperativo":0,"creditos_sectorReal":1,"promedio_ingresos_datacredito":1204496,"tendencia_ingresos":"Creciente"}'
```
```json
{"probabilidad_mora":0.2942,"clase":0,"nivel":"bajo","decision":"aprobar","umbral":0.4962}
```

Perfil riesgoso (23 anios, independiente, score 610, 14 consultas, 36 meses, tendencia decreciente, sin ingreso en la central):
```json
{"probabilidad_mora":0.7867,"clase":1,"nivel":"alto","decision":"rechazar","umbral":0.4962}
```

Campo faltante -> `HTTP 422` con el detalle de Pydantic. Lote de 2 solicitantes:
```json
{"n":2,"n_riesgo":0,"tasa_riesgo":0.0,"probabilidad_media":0.2519,"predicciones":[...]}
```

### Pruebas (`tests/test_api.py`)

11 pruebas con `TestClient`: health, info, prediccion, monotonicidad (perfil peor -> mayor probabilidad),
422 por campo faltante / extra / valor invalido, nulos de la central, batch, batch vacio y contrato de esquema.

```bash
python -m pytest tests -q
```

### Docker

`Dockerfile`: `python:3.12-slim`, `libgomp1`, dependencias minimas (`requirements-api.txt`, sin jupyter / streamlit /
xgboost), copia solo `ft_engineering.py`, `model_deploy.py`, `config.json`, el modelo y `metrics.json`, usuario no root,
`EXPOSE 8000`, `HEALTHCHECK` sobre `/health`, `CMD uvicorn`. `.dockerignore` excluye venv, dataset, notebooks y reportes.

```bash
docker build -t riesgo-api:1.3.0 .
```
```bash
docker run -d --name riesgo-api -p 8000:8000 riesgo-api:1.3.0
```
```bash
curl http://localhost:8000/health
```
```bash
docker logs riesgo-api
```
```bash
docker stop riesgo-api && docker rm riesgo-api
```

Resultado local (Docker Desktop 29.7, backend WSL 2):

```
Imagen: riesgo-api:1.3.0 | 831MB
riesgo-api | Up 6 seconds (healthy) | 0.0.0.0:8000->8000/tcp
$ docker exec riesgo-api whoami        -> api
$ curl http://localhost:8000/health    -> {"status":"ok","modelo":"Random Forest","version_api":"1.3.0","umbral":0.4962}
$ curl -X POST .../predict (ejemplo)   -> {"probabilidad_mora":0.2942,"clase":0,"nivel":"bajo","decision":"aprobar","umbral":0.4962}
```

El job `docker` del CI repite lo mismo en cada push: construye la imagen, levanta el contenedor, espera el `/health`
y ejecuta un `/predict` real. Es la evidencia de que la imagen buildea y la API responde en un entorno limpio.

### Despliegue

1. Merge por PR `developer -> certification -> master`; el CI valida pipeline, pruebas e imagen en cada paso.
2. En el servidor: `docker build` + `docker run` con el tag de la version (o `docker pull` desde un registry si se publica).
3. Operacion: `/health` para el balanceador, `model_monitoring.py --nuevos ventana.csv --strict` como job periodico
   sobre las solicitudes recibidas; si detecta drift, reentrenar con `model_training_evaluation.py` y reconstruir la imagen.
