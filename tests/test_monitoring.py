"""Tests de caractérisation du monitoring d'usage."""
import json
import pytest
from mnist_express.config import Settings
from mnist_express.monitoring import (
    UsageStats,
    get_monitoring_path,
    load_usage_stats,
    record_prediction,
    record_user_feedback,
)

def _settings(tmp_path) -> Settings:
    return Settings(monitoring_dir=str(tmp_path / "monitoring"))

def test_missing_file_returns_default_stats(tmp_path):
    assert load_usage_stats(_settings(tmp_path)) == UsageStats()

def test_record_prediction_updates_counters_and_average(tmp_path):
    settings = _settings(tmp_path)

    record_prediction(settings, inference_time_ms=10.0)
    stats = record_prediction(settings, inference_time_ms=30.0)

    assert stats.total_predictions == 2
    assert stats.cumulative_inference_time_ms == pytest.approx(40.0)
    assert stats.avg_inference_time_ms == pytest.approx(20.0)
    assert stats.last_prediction_at is not None
    assert load_usage_stats(settings) == stats # bien persisté

def test_record_user_feedback_tracks_error_rate(tmp_path):
    settings = _settings(tmp_path)
    
    record_user_feedback(settings, is_error=False)
    record_user_feedback(settings, is_error=True)
    stats = load_usage_stats(settings)

    assert stats.total_feedback == 2
    assert stats.total_user_errors == 1
    assert stats.user_error_rate == pytest.approx(0.5)

def test_corrupted_file_falls_back_to_default_stats(tmp_path):
    settings = _settings(tmp_path)
    path = get_monitoring_path(settings)
    path.parent.mkdir(parents=True)
    path.write_text("{ pas du json", encoding="utf-8")

    assert load_usage_stats(settings) == UsageStats()

def test_stats_file_format_is_stable(tmp_path):
    settings = _settings(tmp_path)

    record_prediction(settings, inference_time_ms=12.5)
    raw = json.loads(get_monitoring_path(settings).read_text(encoding="utf-8"))

    assert set(raw) == {
        "total_predictions",
        "total_feedback",
        "total_user_errors",
        "cumulative_inference_time_ms",
        "avg_inference_time_ms",
        "last_prediction_at",
    }
    assert raw["total_predictions"] == 1
    assert raw["avg_inference_time_ms"] == pytest.approx(12.5)