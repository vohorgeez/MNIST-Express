"""Façade de compatibilité vers l'adaptateur `JoblibModelRepository`.

L'API historiquue (`save_model`, `load_model`) est conservée pour que les tests de caractérisation écrits avant le refactor restent inchangés. Le nouveau code passe par `mnist_express.bootstrap.build_model_repository`."""
from .adapters.joblib_model_repository import JoblibModelRepository
from .config import Settings

def save_model(model, filename: str, settings: Settings) -> str:
    return JoblibModelRepository(settings.model_dir, filename).save(model)

def load_model(filename: str, settings: Settings):
    return JoblibModelRepository(settings.model_dir, filename).load()