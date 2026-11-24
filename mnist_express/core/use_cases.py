"""Cas d'usage : ce que l'extérieur (Streamlit, scripts, tests) peut demander au coeur.

Ces classes forment le port primaire de l'application. Elles orchestrent le pré-traitement, l'inférence et le monitoring en ne parlant qu'aux ports secondaires : aucune ne sait où ni comment les données sont stockées.
"""
import logging
from collections.abc import Callable
import numpy as np
from ..config import Settings
from ..inference import extract_prediction_summary, predict_knn
from ..preprocessing import preprocess_user_drawing
from .model import EmptyDrawingError, Prediction, UsageStats
from .ports import Classifier, UsageStatsRepository

logger = logging.getLogger(__name__)

N_FEATURES = 28*28
TOP_K = 3

class PredictDigit:
    """Prédire le chiffre représenté par un dessin en niveaux de gris"""

    def __init__(
            self,
            classifier: Classifier,
            settings: Settings,
            usage_stats: UsageStatsRepository,
            clock: Callable[[], str],
    ):
        self._classifier = classifier
        self._settings = settings
        self._usage_stats = usage_stats
        self._clock = clock

    def execute(self, drawing: np.ndarray) -> Prediction:
        if drawing is None or np.max(drawing) == 0:
            logger.warning("prediction_skipped | reason=empty_canvas")
            raise EmptyDrawingError("The drawing is empty.")

        preprocessed_img = preprocess_user_drawing(drawing)
        X_input = preprocessed_img.reshape(1, N_FEATURES).astype(np.float64)

        predictions, probabilities, predict_duration = predict_knn(
            model = self._classifier,
            X=X_input,
            settings=self._settings,
        )
        summary = extract_prediction_summary(
            model=self._classifier,
            predictions=predictions,
            probabilities=probabilities,
            sample_index=0,
            top_k=TOP_K,
        )
        inference_time_ms = predict_duration * 1000.0

        if self._settings.enable_monitoring:
            stats = self._usage_stats.load()
            stats.record_prediction(inference_time_ms, at=self._clock())
            self._usage_stats.save(stats)

        logger.info(
            "prediction_completed | predicted_class=%d | confidence=%.4f | inference_time_ms=%.2f",
            summary["predicted_class"],
            summary["confidence"],
            inference_time_ms,
        )

        return Prediction(
            predicted_class=summary["predicted_class"],
            confidence=summary["confidence"],
            top_k=summary["top_k"],
            inference_time_ms=inference_time_ms,
            preprocessed_image=preprocessed_img,
            input_shape=X_input.shape,
        )

class RecordFeedback:
    """Enregistrer l'avis de l'utilisateur sur la dernière prédiction."""

    def __init__(self, usage_stats: UsageStatsRepository):
        self._usage_stats = usage_stats

    def execute(self, is_error: bool) -> UsageStats:
        stats = self._usage_stats.load()
        stats.record_feedback(is_error)
        self._usage_stats.save(stats)
        logger.info("feedback_recorded | is_error=%s", is_error)
        return stats

class GetUsageStats:
    """Lire les statistiques d'usage à afficher."""

    def __init__(self, usage_stats: UsageStatsRepository):
        self._usage_stats = usage_stats

    def execute(self) -> UsageStats:
        return self._usage_stats.load()