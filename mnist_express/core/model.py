"""Objets du domaine : ce que l'application manipule, sans aucune I/O."""
from dataclasses import dataclass
import numpy as np

class EmptyDrawingError(ValueError):
    """Le dessin ne contient aucun trait : il n'y a rien à prédire."""

@dataclass(frozen=True)
class Prediction:
    """Résultat d'une prédiction sur un dessin utilisateur."""
    predicted_class: int
    confidence: float
    top_k: list[tuple[int, float]]
    inference_time_ms: float
    preprocessed_image: np.ndarray
    input_shape: tuple[int, ...]

@dataclass
class UsageStats:
    """Compteurs d'usage de l'application et règles de mise à jour."""
    total_predictions: int = 0
    total_feedback: int = 0
    total_user_errors: int = 0
    cumulative_inference_time_ms: float = 0.0
    avg_inference_time_ms: float = 0.0
    last_prediction_at: str | None = None

    @property
    def user_error_rate(self) -> float:
        if self.total_feedback == 0:
            return 0.0
        return self.total_user_errors / self.total_feedback

    def record_prediction(self, inference_time_ms: float, at: str) -> None:
        self.total_predictions += 1
        self.cumulative_inference_time_ms += inference_time_ms
        self.avg_inference_time_ms = (
            self.cumulative_inference_time_ms / self.total_predictions
        )
        self.last_prediction_at = at

    def record_feedback(self, is_error: bool) -> None:
        self.total_feedback += 1
        if is_error:
            self.total_user_errors += 1