"""Façade de compatibilité vers le domaine (`UsageStats`) et l'adaptateur JSON.

L'API historique est conservée pour que les tests de caractérisation écrits avant le refactor restent inchangés. Les règles de calcul vivent désormais dans `core.model.UsageStats`, la lecture et l'écriture du fichier dans `adapters.json_usage_stats_repository`.
"""
import logging
from pathlib import Path
from .adapters.clock import utc_now_iso
from .adapters.json_usage_stats_repository import JsonUsageStatsRepository
from .config import Settings
from .core.model import UsageStats

__all__ = [
    "UsageStats",
    "get_monitoring_path",
    "ensure_monitoring_dir",
    "load_usage_stats",
    "save_usage_stats",
    "record_prediction",
    "record_user_feedback",
]

logger = logging.getLogger(__name__)

def get_monitoring_path(settings: Settings) -> Path:
    return Path(settings.monitoring_dir) / settings.monitoring_filename

def ensure_monitoring_dir(settings: Settings) -> None:
    Path(settings.monitoring_dir).mkdir(parents=True, exist_ok=True)

def _repository(settings: Settings) -> JsonUsageStatsRepository:
    return JsonUsageStatsRepository(get_monitoring_path(settings))

def load_usage_stats(settings: Settings) -> UsageStats:
    return _repository(settings).load()

def save_usage_stats(settings: Settings, stats: UsageStats) -> None:
    _repository(settings).save(stats)

def record_prediction(settings: Settings, inference_time_ms: float) -> UsageStats:
    repository = _repository(settings)
    stats = repository.load()
    stats.record_prediction(inference_time_ms, at=utc_now_iso())
    repository.save(stats)

    logger.info(
        "Prediction recorded | total_predictions=%d | inference_time_ms=%.2f | avg_inference_time_ms=%.2f",
        stats.total_predictions,
        inference_time_ms,
        stats.avg_inference_time_ms,
    )

    return stats

def record_user_feedback(settings: Settings, is_error: bool) -> UsageStats:
    repository = _repository(settings)
    stats = repository.load()
    stats.record_feedback(is_error)
    repository.save(stats)

    logger.info(
        "User feedback recorded | total_feedback=%d | total_user_errors=%d | user_error_rate=%.4f",
        stats.total_feedback,
        stats.total_user_errors,
        stats.user_error_rate,
    )