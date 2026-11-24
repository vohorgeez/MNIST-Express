"""Tests des cas d'usage avec un adaptateur en mémoire.

C'est le bénéfice direct de l'architecture hexagonale : le coeur se teste sans Streamlit, sans fichier joblib et sans rien écrire sur le disque.
"""
from dataclasses import replace
import numpy as np
import pytest
from mnist_express.config import Settings
from mnist_express.core.model import EmptyDrawingError, UsageStats
from mnist_express.core.use_cases import GetUsageStats, PredictDigit, RecordFeedback
from test_prediction_flow import fitted_model, ring, run_legacy_flow, vertical_bar

FIXED_TIME = "2026-11-20T10:00:00+00:00"

class InMemoryUsageStatsRepository:
    """Adaptateur de test du port UsageStatsRepository."""

    def __init__(self, stats: UsageStats | None = None):
        self.stats = stats or UsageStats()
        self.saves = 0

    def load(self) -> UsageStats:
        return replace(self.stats) # une copie, comme une relecture de fichier

    def save(self, stats: UsageStats) -> None:
        self.stats = replace(stats)
        self.saves += 1

def make_predict_digit(settings=None, repository=None, model=None):
    repository = repository or InMemoryUsageStatsRepository()
    use_case = PredictDigit(
        classifier=model or fitted_model(),
        settings=settings or Settings(),
        usage_stats=repository,
        clock=lambda: FIXED_TIME,
    )
    return use_case, repository

def test_predict_digit_recognises_a_one():
    use_case, _ = make_predict_digit()

    prediction = use_case.execute(vertical_bar(x0=135))

    assert prediction.predicted_class == 1
    assert prediction.confidence == pytest.approx(1.0)

@pytest.mark.parametrize(
    "drawing",
    [vertical_bar(x0=135), vertical_bar(x0=110), ring(thickness=22), ring(thickness=28)],
)
def test_predict_digit_matches_the_legacy_flow(drawing):
    model, settings = fitted_model(), Settings()
    use_case, _ = make_predict_digit(settings=settings, model=model)

    legacy = run_legacy_flow(model, settings, drawing)
    prediction = use_case.execute(drawing)

    assert prediction.predicted_class == legacy["predicted_class"]
    assert prediction.confidence == pytest.approx(legacy["confidence"])
    assert prediction.top_k == legacy["top_k"]

def test_predict_digit_returns_what_the_ui_displays():
    use_case, _ = make_predict_digit()

    prediction = use_case.execute(vertical_bar())

    assert prediction.preprocessed_image.shape == (28, 28)
    assert prediction.input_shape == (1, 784)
    assert prediction.inference_time_ms >= 0.0
    assert len(prediction.top_k) == 2 # deux classes seulement dans ce modèle

def test_predict_digit_records_usage_when_monitoring_is_enabled():
    use_case, repository = make_predict_digit(settings=Settings(enable_monitoring=True))

    use_case.execute(vertical_bar())

    assert repository.stats.total_predictions == 1
    assert repository.stats.last_prediction_at == FIXED_TIME

def test_predict_digit_records_nothing_when_monitoring_is_disabled():
    use_case, repository = make_predict_digit(settings=Settings(enable_monitoring=False))

    use_case.execute(vertical_bar())

    assert repository.saves == 0

def test_empty_drawing_raises_and_records_nothing():
    use_case, repository = make_predict_digit()

    with pytest.raises(EmptyDrawingError):
        use_case.execute(np.zeros((280, 280), dtype=np.uint8))

    assert repository.saves == 0

def test_record_feedback_updates_the_error_rate():
    repository = InMemoryUsageStatsRepository()
    record_feedback = RecordFeedback(repository)

    record_feedback.execute(is_error=False)
    stats = record_feedback.execute(is_error=True)

    assert stats.total_feedback == 2
    assert stats.user_error_rate == pytest.approx(0.5)
    assert repository.stats == stats

def test_get_usage_stats_reads_the_repository():
    stored = UsageStats(total_predictions=7, total_feedback=2, total_user_errors=1)

    stats = GetUsageStats(InMemoryUsageStatsRepository(stored)).execute()

    assert stats == stored