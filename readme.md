# PI M5 - Modelo predictivo de riesgo crediticio

Proyecto Integrador del Modulo 5 (Data Science, Henry). Modelo de machine learning
que anticipa si un nuevo solicitante de credito pagara a tiempo, desplegado como API
(FastAPI + Docker) con monitoreo de data drift y una app Streamlit.

> Estado actual: **V1.0.0** - estructura de carpetas y ramas (punto de partida).

## Estructura del repositorio (no modificar: los pipelines de Jenkins dependen de ella)

```
mlops_pipeline/
└── src/
    ├── Cargar_datos.ipynb
    ├── comprension_eda.ipynb
    ├── ft_engineering.py
    ├── model_training_evaluation.py
    ├── model_deploy.py
    ├── model_monitoring.py
    └── config.json            # codigo de proyecto usado por set_up.bat
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
| `certification` | Validacion antes de produccion (merge desde `developer`) |
| `master` | Produccion (merge desde `certification`) |

| Version | Contenido |
|---|---|
| V1.0.0 | Estructura de carpetas identica en las 3 ramas |
| V1.0.1 | `Cargar_datos.ipynb` y `comprension_eda.ipynb` |
| V1.1.0 | Ingenieria de caracteristicas + primeros modelos |
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
