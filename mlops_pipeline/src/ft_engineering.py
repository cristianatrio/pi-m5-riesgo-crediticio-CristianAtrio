"""
Ingenieria de caracteristicas para el modelo de riesgo crediticio (PI M5).

Implementa como pipeline reutilizable de scikit-learn + feature-engine todas las
decisiones documentadas en ``comprension_eda.ipynb`` (seccion 5):

1. Limpieza de valores imposibles (edad >= 100, salario 0, score fuera de rango,
   tendencia_ingresos con numeros) -> pasan a nulo.
2. Indicadores de faltante para los nulos informativos de la central de riesgo.
3. Imputacion: 0 para saldos, mediana para numericas, "Sin dato" para tendencia.
4. Winsorizacion al p99 de los montos con colas extremas.
5. Features nuevas de capacidad de pago y endeudamiento (ratios, flags).
6. Log de montos, agrupacion de categorias raras, one-hot, drop de correlacionadas
   y escalado.

Uso como modulo::

    from ft_engineering import cargar_datos, separar_target, construir_preprocesador
    X, y = separar_target(cargar_datos())
    prep = construir_preprocesador().fit(X_train, y_train)

Uso como script (genera los splits transformados para inspeccion y tests)::

    python mlops_pipeline/src/ft_engineering.py

El preprocesador ajustado NO se guarda por separado: el unico artefacto de inferencia
es ``mlops_pipeline/models/modelo_riesgo.joblib`` (preprocesador + modelo en un solo
Pipeline, generado por ``model_training_evaluation.py``). Asi no hay dos versiones
que puedan divergir.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    ArbitraryNumberImputer,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropCorrelatedFeatures
from feature_engine.transformation import LogCpTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Rutas y configuracion
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((SRC_DIR / "config.json").read_text(encoding="utf-8"))


def buscar_raiz(nombre: str = "Base_de_datos.csv") -> Path:
    """Raiz del repo: variable RAIZ_PROYECTO, o la carpeta que contiene el dataset, o
    <raiz>/mlops_pipeline/src -> <raiz> (contenedor de la API, sin dataset)."""
    if os.environ.get("RAIZ_PROYECTO"):
        return Path(os.environ["RAIZ_PROYECTO"]).resolve()
    for carpeta in [SRC_DIR, *SRC_DIR.parents]:
        if (carpeta / nombre).exists():
            return carpeta
    return SRC_DIR.parents[1]


RAIZ = buscar_raiz(CONFIG["dataset_path"])
DATA_DIR = RAIZ / "mlops_pipeline" / "data"
MODELS_DIR = RAIZ / "mlops_pipeline" / "models"

TARGET = CONFIG["target"]
RANDOM_STATE = CONFIG["random_state"]

# ---------------------------------------------------------------------------
# Grupos de variables (los nombres son los del CSV original)
# ---------------------------------------------------------------------------
COLS_EXCLUIDAS = CONFIG["features_excluidas"]  # puntaje (leakage) y fecha_prestamo

COLS_SALDOS = ["saldo_mora", "saldo_total", "saldo_principal", "saldo_mora_codeudor"]
COLS_NA_INDICADOR = ["promedio_ingresos_datacredito", "saldo_mora", "saldo_principal", "saldo_mora_codeudor"]
COLS_IMPUTAR_MEDIANA = ["edad_cliente", "salario_cliente", "puntaje_datacredito", "promedio_ingresos_datacredito"]
COLS_WINSORIZAR = [
    "salario_cliente", "total_otros_prestamos", "capital_prestado", "cuota_pactada",
    "saldo_total", "saldo_principal", "promedio_ingresos_datacredito",
]
# Los ratios se crean dentro del pipeline (FeaturesCredito) y tambien tienen colas largas
COLS_RATIOS = ["ratio_cuota_salario", "ratio_deuda_salario", "ratio_capital_salario"]
COLS_LOG = COLS_WINSORIZAR + ["saldo_mora"] + COLS_RATIOS
COLS_CATEGORICAS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]

# Contrato de entrada del pipeline (y de la API): columnas crudas que el modelo necesita.
# No incluye puntaje ni fecha_prestamo (excluidas) ni el target.
COLUMNAS_REQUERIDAS = [
    "tipo_credito", "capital_prestado", "plazo_meses", "edad_cliente", "tipo_laboral",
    "salario_cliente", "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total", "saldo_principal",
    "saldo_mora_codeudor", "creditos_sectorFinanciero", "creditos_sectorCooperativo",
    "creditos_sectorReal", "promedio_ingresos_datacredito", "tendencia_ingresos",
]


def validar_esquema(X: pd.DataFrame, columnas: Sequence[str] = COLUMNAS_REQUERIDAS) -> None:
    """Falla con un mensaje operativo si faltan columnas, en vez de un KeyError generico."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Se esperaba un pandas.DataFrame, llego {type(X).__name__}")
    faltantes = [c for c in columnas if c not in X.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas por el pipeline: {faltantes}")


# ---------------------------------------------------------------------------
# Transformadores propios
# ---------------------------------------------------------------------------
class LimpiezaCredito(BaseEstimator, TransformerMixin):
    """Valida el esquema, convierte valores imposibles en nulos y elimina las columnas excluidas.

    No aprende nada del train (stateless): las reglas salen del EDA y estan en
    config.json. Se deja dentro del pipeline para que la API reciba datos crudos.
    """

    def __init__(self, columnas_excluidas=None, rango_valido=None, categorias_tendencia=None):
        self.columnas_excluidas = columnas_excluidas
        self.rango_valido = rango_valido
        self.categorias_tendencia = categorias_tendencia

    def fit(self, X, y=None):
        validar_esquema(X)
        self.columnas_excluidas_ = list(self.columnas_excluidas or CONFIG["features_excluidas"])
        self.rango_valido_ = dict(self.rango_valido or CONFIG["rango_valido"])
        self.categorias_tendencia_ = list(self.categorias_tendencia or CONFIG["categorias_tendencia"])
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        validar_esquema(X)
        X = X.copy()
        X = X.drop(columns=[c for c in self.columnas_excluidas_ + [TARGET] if c in X.columns])

        # Valores fuera de rango -> nulo (los imputa el paso siguiente)
        for col, (minimo, maximo) in self.rango_valido_.items():
            if col in X.columns:
                X[col] = X[col].where(X[col].between(minimo, maximo))

        # Salario 0 es "no informado", no un ingreso real
        X["salario_cliente"] = X["salario_cliente"].replace(0, np.nan)

        # tendencia_ingresos trae numeros mezclados: todo lo que no es categoria valida -> nulo
        X["tendencia_ingresos"] = X["tendencia_ingresos"].where(
            X["tendencia_ingresos"].isin(self.categorias_tendencia_)
        )

        # tipo_credito es un codigo de producto, no una cantidad
        X["tipo_credito"] = X["tipo_credito"].astype("Int64").astype(str)
        X["tipo_laboral"] = X["tipo_laboral"].astype(str)
        return X

    def get_feature_names_out(self, input_features=None):
        features = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        return np.asarray([c for c in features if c not in self.columnas_excluidas_ + [TARGET]])


class FeaturesCredito(BaseEstimator, TransformerMixin):
    """Crea las variables de negocio definidas en el EDA (seccion 4.5 y 5).

    Se aplica despues de imputar y winsorizar, asi los ratios no reciben nulos,
    ceros ni salarios de miles de millones.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        salario = X["salario_cliente"].clip(lower=1)

        # Capacidad de pago y endeudamiento
        X["ratio_cuota_salario"] = X["cuota_pactada"] / salario
        X["ratio_deuda_salario"] = X["total_otros_prestamos"] / salario
        X["ratio_capital_salario"] = X["capital_prestado"] / salario

        # Flags de riesgo detectados en el bivariable
        X["plazo_largo"] = (X["plazo_meses"] >= 24).astype(int)
        X["tiene_mora_previa"] = (X["saldo_mora"] > 0).astype(int)
        X["sin_historial_financiero"] = (X["creditos_sectorFinanciero"] == 0).astype(int)
        X["creditos_total_sectores"] = (
            X["creditos_sectorFinanciero"] + X["creditos_sectorCooperativo"] + X["creditos_sectorReal"]
        )
        return X

    def get_feature_names_out(self, input_features=None):
        nuevas = [
            "ratio_cuota_salario", "ratio_deuda_salario", "ratio_capital_salario", "plazo_largo",
            "tiene_mora_previa", "sin_historial_financiero", "creditos_total_sectores",
        ]
        features = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        return np.asarray(list(features) + nuevas)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def construir_preprocesador(escalar: bool = True) -> Pipeline:
    """Devuelve el pipeline de ingenieria de caracteristicas, sin ajustar.

    Parameters
    ----------
    escalar : bool
        Si True agrega StandardScaler al final (necesario para la regresion
        logistica; inocuo para los arboles).
    """
    pasos = [
        ("limpieza", LimpiezaCredito()),
        # Nulos informativos (EDA 3.7): quien no tiene dato en la central tiene mas mora
        ("indicador_nulos", AddMissingIndicator(variables=COLS_NA_INDICADOR)),
        # Saldo faltante = sin deuda reportada
        ("imputar_saldos", ArbitraryNumberImputer(arbitrary_number=0, variables=COLS_SALDOS)),
        # Numericas con valores imposibles ya convertidos a nulo -> mediana (robusta a outliers)
        ("imputar_mediana", MeanMedianImputer(imputation_method="median", variables=COLS_IMPUTAR_MEDIANA)),
        # Categoria explicita para el 27% sin tendencia
        ("imputar_tendencia", CategoricalImputer(imputation_method="missing", fill_value="Sin dato",
                                                 variables=["tendencia_ingresos"], ignore_format=True)),
        # Colas extremas de los montos (salarios de miles de millones) -> tope en el p99
        ("winsorizar", Winsorizer(capping_method="quantiles", tail="right", fold=0.01, variables=COLS_WINSORIZAR)),
        ("features_negocio", FeaturesCredito()),
        # Montos y ratios muy asimetricos -> log(x + 1); ayuda al modelo lineal y evita
        # que un par de outliers dominen la correlacion de Pearson del paso siguiente
        ("log_montos", LogCpTransformer(C=1, variables=COLS_LOG)),
        # Codigos de producto con < 0.3% de casos (6, 7, 68) -> "Rare"
        ("categorias_raras", RareLabelEncoder(tol=0.003, n_categories=2, variables=["tipo_credito"],
                                              ignore_format=True)),
        ("one_hot", OneHotEncoder(variables=COLS_CATEGORICAS, drop_last=False, ignore_format=True)),
        # Multicolinealidad (EDA 4.2): descarta una de cada par con |r| > 0.9
        ("drop_correlacionadas", DropCorrelatedFeatures(threshold=0.9, method="pearson")),
    ]
    if escalar:
        pasos.append(("escalado", StandardScaler().set_output(transform="pandas")))
    return Pipeline(pasos)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_datos(ruta: Path | str | None = None) -> pd.DataFrame:
    """Lee Base_de_datos.csv con la fecha parseada."""
    ruta = Path(ruta) if ruta else RAIZ / CONFIG["dataset_path"]
    return pd.read_csv(ruta, parse_dates=["fecha_prestamo"])


def separar_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve X (crudo, sin el target) e y = mora (1 = no pago a tiempo)."""
    if TARGET not in df.columns:
        raise ValueError(f"No se encontro la columna target requerida: {TARGET}")
    y = (1 - df[TARGET]).rename("mora").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def dividir_train_test(X, y, test_size: float | None = None, random_state: int | None = None):
    """Split estratificado: con 4,75% de positivos hay que preservar la proporcion."""
    return train_test_split(
        X, y,
        test_size=CONFIG["test_size"] if test_size is None else test_size,
        random_state=random_state if random_state is not None else RANDOM_STATE,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = cargar_datos()
    X, y = separar_target(df)
    X_train, X_test, y_train, y_test = dividir_train_test(X, y)
    print(f"Train: {X_train.shape} | mora {y_train.mean():.2%}   Test: {X_test.shape} | mora {y_test.mean():.2%}")

    # El preprocesador se ajusta SOLO con train: medianas, percentiles y categorias raras
    # se aprenden ahi y se reutilizan tal cual en test y en produccion.
    preprocesador = construir_preprocesador()
    X_train_t = preprocesador.fit_transform(X_train, y_train)
    X_test_t = preprocesador.transform(X_test)

    descartadas = preprocesador.named_steps["drop_correlacionadas"].features_to_drop_
    print(f"Features de salida: {X_train_t.shape[1]} | descartadas por correlacion: {sorted(descartadas)}")
    if X_train_t.isna().any().any():
        raise RuntimeError("Quedaron nulos despues del pipeline")

    # Splits transformados: solo para inspeccion y tests (ignorados por git, regenerables)
    X_train_t.to_parquet(DATA_DIR / "X_train.parquet")
    X_test_t.to_parquet(DATA_DIR / "X_test.parquet")
    y_train.to_frame().to_parquet(DATA_DIR / "y_train.parquet")
    y_test.to_frame().to_parquet(DATA_DIR / "y_test.parquet")
    print(f"Splits transformados guardados en {DATA_DIR}")
    print(X_train_t.describe().T[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
