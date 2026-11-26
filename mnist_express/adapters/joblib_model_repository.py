"""Adaptateur du port `ModelRepository` : le modèle est un fichier joblib."""
import logging
from pathlib import Path
import joblib
from ..core.ports import Classifier

logger = logging.getLogger(__name__)

class JoblibModelRepository:
    def __init__(self, model_dir: str | Path, filename: str):
        self.path = Path(model_dir) / filename

    def save(self, model: Classifier) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.path)
        logger.info("model_saved | path=%s", self.path)
        return str(self.path)

    def load(self) -> Classifier:
        if not self.path.exists():
            raise FileNotFoundError(f"Model file not found: {self.path}")
        model = joblib.load(self.path)
        logger.info("model_loaded | path=%s", self.path)
        return model