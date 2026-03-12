import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path

from .config import Settings

logger = logging.getLogger(__name__)

@dataclass
class UsageStats:
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
    
def get_monitoring_path(settings: Settings) -> Path:
    return Path(settings.monitoring_dir) / settings.monitoring_filename

def ensure_monitoring_dir(settings: Settings) -> None:
    Path(settings.monitoring_dir).mkdir(parents=True, exist_ok=True)

def load_usage_stats(settings: Settings) -> UsageStats:
    path = get_monitoring_path(settings)

    if not path.exists():
        logger.info("Monitoring file does not exist yes: %s", path)
        return UsageStats()
    
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return UsageStats(**raw)

def save_usage_stats(settings: Settings, stats: UsageStats) -> None:
    ensure_monitoring_dir(settings)
    path = get_monitoring_path(settings)

    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    logger.info("Monitoring stats saved to %s", path)

def record_prediction(
        settings: Settings,
        inference_time_ms: float,
) -> UsageStats:
    stats = load_usage_stats(settings)

    stats.total_predictions += 1
    stats.cumulative_inference_time_ms += inference_time_ms
    stats.avg_inference_time_ms = (
        stats.cumulative_inference_time_ms / stats.total_predictions
    )
    stats.last_prediction_at = datetime.now(UTC).isoformat()

    save_usage_stats(settings, stats)

    logger.info(
        "Prediction recorded | total_predictions=%d | inference_time_ms=%.2f | avg_inference_time_ms=%.2f",
        stats.total_predictions,
        inference_time_ms,
        stats.avg_inference_time_ms,
    )

    return stats

def record_user_feedback(
        settings: Settings,
        is_error: bool,
) -> UsageStats:
    stats = load_usage_stats(settings)

    stats.total_feedback += 1
    if is_error:
        stats.total_user_errors += 1

    save_usage_stats(settings, stats)

    logger.info(
        "User feedback recorded | total_feedback=%d | total_user_errors=%d | user_error_rate=%.4f",
        stats.total_feedback,
        stats.total_user_errors,
        stats.user_error_rate,
    )

    return stats