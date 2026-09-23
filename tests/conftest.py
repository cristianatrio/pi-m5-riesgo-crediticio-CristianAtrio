"""Fixtures compartidas: ruta a los scripts del pipeline, dataset y modelo cargados una sola vez."""

import sys
from pathlib import Path

import joblib
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "mlops_pipeline" / "src"
sys.path.insert(0, str(SRC_DIR))

from ft_engineering import MODELS_DIR, cargar_datos  # noqa: E402


@pytest.fixture(scope="session")
def datos():
    """Base_de_datos.csv completo (solo lectura: cada prueba trabaja sobre copias)."""
    return cargar_datos()


@pytest.fixture(scope="session")
def muestra(datos):
    """Muestra estratificada chica para entrenar rapido en las pruebas (con ~50 moras)."""
    return datos.groupby("Pago_atiempo", group_keys=False).sample(frac=0.1, random_state=0)


@pytest.fixture(scope="session")
def modelo():
    return joblib.load(MODELS_DIR / "modelo_riesgo.joblib")
